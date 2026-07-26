/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Runtime Builder - fully_distributed_within_core host binding
 *
 * Stages per-run tensor arguments, prepares the direct distributed runtime
 * arena, and copies mutable outputs back to host at finalize.
 *
 * init_runtime_impl:
 *   - Converts host tensor pointers to device pointers (all inputs copied H2D;
 *     only OUTPUT/INOUT tensors are copied back D2H)
 *   - Registers callable-side kernel metadata
 *   - Uploads the prebuilt direct distributed runtime header
 *
 * validate_runtime_impl:
 *   - Copies OUTPUT/INOUT tensors back from device to host (read-only inputs
 *     are skipped)
 *   - Frees device memory
 */

#include <stddef.h>
#include <stdint.h>

#include <cinttypes>
#include <cstdlib>

#include "../runtime/pto_runtime2.h"
#include "../runtime/runtime.h"
#include "../../../../common/task_interface/call_config.h"
#include "utils/device_arena.h"
#include "callable.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "prepare_callable_common.h"

/**
 * Stage the per-callable resources (kernel binaries + orchestration entry) into
 * the supplied runtime so a subsequent bind_callable_to_runtime_impl can use
 * them. This is the cacheable half of init_runtime_impl: nothing here depends
 * on per-run argument values, so the prepare_callable / run_prepared split
 * lets us run this once per callable_id and amortize across runs.
 *
 * @param runtime   Pointer to pre-constructed Runtime (host_api populated)
 * @param callable  ChipCallable carrying child kernel binaries
 * @return 0 on success, -1 on failure
 */
extern "C" int
prepare_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out) {
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return -1;
    }
    if (upload_fn == nullptr || out == nullptr) {
        LOG_ERROR("upload_fn or out is null");
        return -1;
    }
    *out = CallableArtifacts{};
    out->signature.assign(callable->signature_, callable->signature_ + callable->sig_count());

    LOG_INFO_V0("Registering %d kernel(s) in prepare_callable_impl", callable->child_count());
    if (upload_and_collect_child_addrs(callable, upload_fn, &out->kernel_addrs) != 0) {
        LOG_ERROR("Failed to upload ChipCallable buffer");
        return -1;
    }
    for (const ChildKernelAddr &c : out->kernel_addrs) {
        if (c.func_id < 0 || c.func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d)", c.func_id, RUNTIME_MAX_FUNC_ID);
            return -1;
        }
    }

#if defined(SIMPLER_AICORE_LINKED_ORCH)
    out->aicore_linked_orch = true;
    LOG_INFO_V0("AICore-linked orchestration staged; standalone orch SO is not registered");
#else
    const uint8_t *orch_so_binary = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();
    if (orch_so_binary == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration SO binary is required for device orchestration");
        return -1;
    }
    out->orch_so_data = orch_so_binary;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();
    LOG_INFO_V0("Orchestration SO: %zu bytes staged (host-only)", orch_so_size);
#endif
    return 0;
}

/**
 * Per-run binding: build device-side argument storage (tensor copy-out, GM
 * heap, runtime arena) and publish it to the runtime. Assumes the
 * callable-side state (kernel binaries and orchestration entry metadata) is
 * already populated by prepare_callable_impl.
 *
 * Splitting this from prepare_callable_impl matches the per-callable_id
 * design: register/run_prepared invokes this every call, while the prep
 * half runs only once per callable_id.
 *
 * @param runtime    Pointer to pre-constructed Runtime (host_api populated)
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr, const ArgDirection *signature,
    int sig_count, uint64_t ring_task_window, uint64_t ring_heap, uint64_t ring_dep_pool,
    const uint64_t *ring_task_windows, const uint64_t *ring_heaps, const uint64_t *ring_dep_pools
) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }
    if (!fdwic_build_identity_matches(runtime->fdwic_build_identity, static_cast<uint32_t>(sizeof(Runtime)))) {
        LOG_ERROR(
            "FDWIC host Runtime build identity is corrupt or stale "
            "(abi=%u, mode=%u, ring_cap=%u, runtime_bytes=%u)",
            runtime->fdwic_build_identity.abi_version, runtime->fdwic_build_identity.tensor_map_mode,
            runtime->fdwic_build_identity.tensor_map_ring_cap, runtime->fdwic_build_identity.runtime_bytes
        );
        return -1;
    }
    runtime->fdwic_build_identity.error_bits = FdwicBuildErrorNone;
    if (orch_args == nullptr) {
        LOG_ERROR("orch_args pointer is null");
        return -1;
    }
    // trb runs orchestration on the device — there is no host-side orch
    // function pointer to invoke. The c_api signature accepts one for
    // symmetry with hbg; assert the trb-side invariant here.
    if (host_orch_func_ptr != nullptr) {
        LOG_ERROR("bind_callable_to_runtime_impl: trb does not accept a host_orch_func_ptr");
        return -1;
    }
    (void)ring_task_window;
    (void)ring_dep_pool;
    (void)ring_task_windows;
    (void)ring_dep_pools;

    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO_V0("fdwic bind: %d tensors + %d scalars", tensor_count, scalar_count);

    // Build device args: copy from input, replace host tensor pointers with device pointers
    ChipStorageTaskArgs device_args;

    for (int i = 0; i < tensor_count; i++) {
        Tensor t = orch_args->tensor(i);

        if (t.is_child_memory()) {
            LOG_INFO_V0("  Tensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
            device_args.add_tensor(t);
            continue;
        }

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());

        void *dev_ptr = runtime->host_api.device_malloc(size);
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return -1;
        }

        // Pure write-only OUTPUT buffers carry no meaningful host content, so
        // the H2D copy-in is wasted. Zero them on-device instead (cheap HBM
        // memset, no PCIe) so any region the kernel leaves unwritten reads as 0
        // rather than pooled-allocator garbage. INOUT (read-before-write)
        // and IN keep the H2D copy. Falls back to copy_to_device if a backend
        // did not wire device_memset.
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        int rc;
        if (is_pure_output && runtime->host_api.device_memset != nullptr) {
            rc = runtime->host_api.device_memset(dev_ptr, 0, size);
        } else {
            rc = runtime->host_api.copy_to_device(dev_ptr, host_ptr, size);
        }
        if (rc != 0) {
            LOG_ERROR("Failed to stage tensor %d to device", i);
            runtime->host_api.device_free(dev_ptr);
            return -1;
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (child_memory tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_pairs_.push_back({host_ptr, dev_ptr, size, needs_copy_back});
        LOG_INFO_V0("  Tensor %d: %zu bytes at %p", i, size, dev_ptr);

        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
        device_args.add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        device_args.add_scalar(orch_args->scalar(i));
    }

    // Lay out the per-Worker static device arena. Direct distributed runtime
    // uses one GM output heap plus one prebuilt runtime arena; the old PTO2
    // shared-memory region is not part of this path.
    // Owned by DeviceRunner across runs — do NOT record in tensor_pairs_; the
    // free is deferred to DeviceRunner::finalize(). The runtime-arena size is
    // determined by replaying the reserve sequence on a host-side arena.
    uint64_t total_heap_size = PTO2_HEAP_SIZE;
    if (ring_heap != 0) {
        if (ring_heap < 1024) {
            LOG_ERROR("runtime_env.ring_heap=%" PRIu64 " must be >= 1024", ring_heap);
            return -1;
        }
        total_heap_size = ring_heap;
    }
    if (ring_heaps != nullptr && ring_heaps[0] != 0) {
        if (ring_heaps[0] < 1024) {
            LOG_ERROR("ring_heaps[0]=%" PRIu64 " must be >= 1024", ring_heaps[0]);
            return -1;
        }
        total_heap_size = ring_heaps[0];
    }
    if (const char *e = std::getenv("PTO_DIST_HEAP_MB")) {
        const long mb = std::atol(e);
        if (mb > 0) {
            total_heap_size = static_cast<uint64_t>(mb) << 20;
        }
    }
    LOG_INFO_V0("Dist heap size: %" PRIu64 " bytes", total_heap_size);

    DeviceArena host_arena;  // libc malloc backend by default
    PTO2RuntimeArenaLayout layout = runtime_reserve_layout(host_arena);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return -1;
    }

    if (runtime->host_api.setup_static_arena(total_heap_size, 0, layout.arena_size) != 0) {
        LOG_ERROR("Failed to setup pooled static arena");
        return -1;
    }

    void *gm_heap = runtime->host_api.acquire_pooled_gm_heap();
    if (gm_heap == nullptr) {
        LOG_ERROR("Failed to acquire pooled GM heap");
        return -1;
    }

    void *runtime_arena_dev = runtime->host_api.acquire_pooled_runtime_arena();
    if (runtime_arena_dev == nullptr) {
        LOG_ERROR("Failed to acquire pooled runtime arena");
        return -1;
    }

    runtime->set_orch_args(device_args);

    // -------------------------------------------------------------------------
    // Build the prebuilt runtime-arena image on host.
    //
    // We pre-compute the runtime header and dist-global image on host, then
    // rtMemcpy it into the pooled runtime-arena region that DeviceRunner keeps
    // alive across runs. The direct distributed AICPU setup path only reads the
    // PTO2Runtime header and the DistGlobal pointed to by rt->dist_global.
    // -------------------------------------------------------------------------
    PTO2Runtime *rt = runtime_init_data_from_layout(host_arena, layout, PTO2_MODE_EXECUTE, gm_heap, total_heap_size);
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return -1;
    }
    rt->dist_global = static_cast<char *>(runtime_arena_dev) + layout.off_dist_global;

    // Keep the layout in the PTO2Runtime image for compatibility with the
    // prebuilt image format. The direct distributed AICPU setup path only uses
    // set_prebuilt_arena below to locate the PTO2Runtime header.
    rt->prebuilt_layout = layout;

    int rc_upload = runtime->host_api.copy_to_device(runtime_arena_dev, host_arena.base(), layout.arena_size);
    if (rc_upload != 0) {
        LOG_ERROR("Failed to rtMemcpy prebuilt runtime arena to device (rc=%d)", rc_upload);
        return -1;
    }
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);

    LOG_INFO_V0("fdwic runtime ready: %d tensors + %d scalars", tensor_count, scalar_count);

    return 0;
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Frees device memory for recorded tensors
 * 3. Clears tensor pair state
 *
 * @param runtime  Pointer to Runtime
 * @return 0 on success, -1 on failure
 */
extern "C" int validate_runtime_impl(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return -1;
    }

    int rc = 0;

    LOG_INFO_V0("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    TensorPair *tensor_pairs = runtime->tensor_pairs_.data();
    int tensor_pair_count = static_cast<int>(runtime->tensor_pairs_.size());

    LOG_INFO_V0("Tensor pairs to process: %d", tensor_pair_count);

    for (int i = 0; i < tensor_pair_count; i++) {
        const TensorPair &pair = tensor_pairs[i];

        // Skip if device pointer is null
        if (pair.dev_ptr == nullptr) {
            LOG_WARN("Tensor %d has null device pointer, skipping", i);
            continue;
        }

        // If host pointer is null, this is a device-only allocation (no copy-back)
        if (pair.host_ptr == nullptr) {
            LOG_INFO_V0("Tensor %d: device-only allocation (no copy-back)", i);
            continue;
        }

        // Read-only INPUT tensors were uploaded H2D but the kernel never
        // wrote them — copying them back (potentially ~GB) is pure waste.
        // They are still device_free'd in the cleanup loop below.
        if (!pair.needs_copy_back) {
            LOG_INFO_V0("Tensor %d: read-only input, skipping copy-back", i);
            continue;
        }

        int copy_rc = runtime->host_api.copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
        if (copy_rc != 0) {
            LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
            rc = copy_rc;
        } else {
            LOG_INFO_V0("Tensor %d: %zu bytes copied to host", i, pair.size);
        }
    }

    // Cleanup device tensors
    LOG_INFO_V0("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr != nullptr) {
            runtime->host_api.device_free(tensor_pairs[i].dev_ptr);
        }
    }
    LOG_INFO_V0("Freed %d device allocations", tensor_pair_count);

    // Clear the per-run dispatch-table entries staged by prepare_callable_impl.
    // The underlying chip-callable device buffer is pool-managed by
    // DeviceRunner (keyed by content hash) and bulk-freed in
    // DeviceRunner::finalize().
    int kernel_count = runtime->get_registered_kernel_count();
    for (int i = 0; i < kernel_count; i++) {
        int func_id = runtime->get_registered_kernel_func_id(i);
        runtime->set_function_bin_addr(func_id, 0);
    }
    if (kernel_count > 0) {
        LOG_INFO_V0("Cleared %d kernel dispatch-table entries", kernel_count);
    }
    runtime->clear_registered_kernels();

    // Clear tensor pairs
    runtime->tensor_pairs_.clear();

    LOG_INFO_V0("=== Finalize Complete ===");

    return rc;
}

// Strong override of the weak runtime_apply_example_exec_time hook declared in
// pto_runtime_c_api.h. fully_distributed_within_core is the only runtime that
// implements the sim-only trace-driven replay feature: stash the per-func
// reference durations on the Runtime so execute_slot busy-waits
// example_exec_time_ns_[func_id] in place of the real incore kernel. A func
// left at 0 (or func_id beyond the table) still runs for real. See
// call_config.h::use_example_exec_time.
extern "C" void
runtime_apply_example_exec_time(void *runtime, int use_example_exec_time, const int32_t *example_exec_time_ns) {
    Runtime *rt = static_cast<Runtime *>(runtime);
    rt->use_example_exec_time_ = (use_example_exec_time != 0);
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; ++i) {
        rt->example_exec_time_ns_[i] =
            (use_example_exec_time != 0 && example_exec_time_ns != nullptr && i < CALLCONFIG_MAX_EXAMPLE_FUNCS) ?
                example_exec_time_ns[i] :
                0;
    }
}
