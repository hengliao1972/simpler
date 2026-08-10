/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#include "aicpu_plan_backend_abi.h"

#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "../../../common/aicpu_plan_protocol.h"
#include "pto_orchestration_api.h"

namespace {

using namespace pa_scheduler::aicpu_plan;

template <typename Function>
Function Load(void *handle, const char *name)
{
    dlerror();
    auto function = reinterpret_cast<Function>(dlsym(handle, name));
    const char *error = dlerror();
    if (error != nullptr || function == nullptr) {
        std::cerr << "dlsym(" << name << ") failed: "
                  << (error == nullptr ? "null" : error) << '\n';
        std::exit(2);
    }
    return function;
}

bool Check(bool condition, const char *message)
{
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

Tensor External(
    void *address, const uint32_t *shape, uint32_t rank,
    DataType dtype
)
{
    Tensor tensor{};
    tensor.buffer.addr = reinterpret_cast<uintptr_t>(address);
    tensor.owner_task_id = PTO2TaskId::invalid();
    tensor.start_offset = 0U;
    tensor.version = 0;
    tensor.ndims = rank;
    tensor.dtype = dtype;
    tensor.manual_dep = false;
    tensor.child_memory = 0U;
    uint64_t elements = 1U;
    for (uint32_t dim = rank; dim > 0U; --dim) {
        const uint32_t index = dim - 1U;
        tensor.shapes[index] = shape[index];
        tensor.strides[index] = static_cast<uint32_t>(elements);
        elements *= shape[index];
    }
    tensor.is_contiguous = true;
    tensor.extent_elem_cache = elements;
    tensor.buffer.size = elements * get_element_size(dtype);
    return tensor;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " libpaged_attention_aicpu_plan.so\n";
        return 2;
    }
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "dlopen failed: " << dlerror() << '\n';
        return 2;
    }

    using Bind = int32_t (*)(const AicpuPlanBackendConfig *);
    using Close = int32_t (*)();
    using Result = AicpuPlanBackendResult (*)();
    using Entry = void (*)(const L2TaskArgs &);
    using Config = PTO2OrchestrationConfig (*)(const L2TaskArgs &);
    const Bind bind = Load<Bind>(handle, "aicpu_plan_backend_bind");
    const Close close = Load<Close>(handle, "aicpu_plan_backend_close");
    const Result result = Load<Result>(handle, "aicpu_plan_backend_result");
    const Entry entry = Load<Entry>(handle, "aicpu_orchestration_entry");
    const Config orch_config =
        Load<Config>(handle, "aicpu_orchestration_config");

    // 两个 batch 分别产生 G1 和 G2，真实 callback 序列为 5+9=14 个 task。
    // 这样一次 smoke 同时覆盖 UP->Alloc 的 batch 尾和 UP->QK 的下一组。
    constexpr uint32_t kBatch = 2U;
    constexpr uint32_t kTotalBlocks = 128U;
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kHeads = 16U;
    constexpr uint32_t kHeadDim = 128U;
    constexpr uint32_t kBlockTableWidth = 128U;
    constexpr uint32_t kExpectedTasks = 14U;
    alignas(64) uint64_t dummy_buffers[6]{};
    alignas(64) int32_t context_lens[kBatch] = {4096, 9000};

    const uint32_t query_shape[3] = {kBatch, kHeads, kHeadDim};
    const uint32_t cache_shape[4] = {
        kTotalBlocks, kBlockSize, 1U, kHeadDim,
    };
    const uint32_t block_table_shape[2] = {kBatch, kBlockTableWidth};
    const uint32_t context_shape[1] = {kBatch};
    const uint32_t output_shape[3] = {kBatch, kHeads, kHeadDim};
    Tensor query = External(
        &dummy_buffers[0], query_shape, 3U, DataType::BFLOAT16
    );
    Tensor key = External(
        &dummy_buffers[1], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor value = External(
        &dummy_buffers[2], cache_shape, 4U, DataType::BFLOAT16
    );
    Tensor block_table = External(
        &dummy_buffers[3], block_table_shape, 2U, DataType::INT32
    );
    Tensor context = External(
        context_lens, context_shape, 1U, DataType::INT32
    );
    Tensor output = External(
        &dummy_buffers[4], output_shape, 3U, DataType::FLOAT32
    );
    L2TaskArgs args;
    args.reset();
    args.add_input(query, key, value, block_table, context, output);
    args.add_scalar(uint64_t{0x3f800000U});

    bool ok = Check(!args.has_error, "failed to construct real L2TaskArgs");
    ok &= Check(
        orch_config(args).expected_arg_count == 7,
        "real PA orchestration config expected_arg_count changed"
    );

    void *control_memory = nullptr;
    void *cells_memory = nullptr;
    if (posix_memalign(
            &control_memory, kAtomicIsolationBytes,
            sizeof(RuntimePlanControl)
        ) != 0 ||
        posix_memalign(
            &cells_memory, kAtomicIsolationBytes,
            kExpectedTasks * sizeof(RuntimeTaskPlanCell)
        ) != 0) {
        std::cerr << "aligned allocation failed\n";
        return 2;
    }
    std::memset(control_memory, 0, sizeof(RuntimePlanControl));
    std::memset(
        cells_memory, 0,
        kExpectedTasks * sizeof(RuntimeTaskPlanCell)
    );
    const AicpuPlanBackendConfig backend_config{
        control_memory, cells_memory, kExpectedTasks, 0U,
    };
    ok &= Check(bind(&backend_config) == 0, "backend bind failed");
    if (ok) entry(args);
    ok &= Check(close() == 0, "backend close failed");

    const AicpuPlanBackendResult backend_result = result();
    ok &= Check(backend_result.status == 0, "backend reported failure");
    ok &= Check(
        backend_result.task_count == kExpectedTasks &&
        backend_result.begin_count == kExpectedTasks &&
        backend_result.finish_count == kExpectedTasks &&
        backend_result.published_count == kExpectedTasks,
        "actual callback count did not close at 14"
    );
    ok &= Check(
        backend_result.alloc_count == 2U &&
        backend_result.aic_count == 6U &&
        backend_result.aiv_count == 6U,
        "actual Alloc/AIC/AIV callback distribution changed"
    );

    auto *control = static_cast<RuntimePlanControl *>(control_memory);
    auto *cells = static_cast<RuntimeTaskPlanCell *>(cells_memory);
    RuntimePlanStorageRef storage{};
    storage.cells_base = reinterpret_cast<uintptr_t>(cells);
    storage.cells_bytes =
        kExpectedTasks * sizeof(RuntimeTaskPlanCell);
    storage.capacity = kExpectedTasks;
    storage.cell_bytes = sizeof(RuntimeTaskPlanCell);
    storage.abi_version = kRuntimePlanAbiVersion;
    RuntimePlanView view{};
    ok &= Check(
        MakeRuntimePlanView(control, storage, view),
        "published storage reference is invalid"
    );
    ok &= Check(
        control->planned_frontier.value == kExpectedTasks &&
        control->closed_task_count.value == kExpectedTasks &&
        control->fatal.value == 0,
        "Plan close/frontier/fatal control mismatch"
    );

    // 期望来自真实 G1+G2 callback 顺序；backend 本身没有这张 task 表。
    const std::array<uint8_t, kExpectedTasks> expected_kind{
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4, 1, 2, 3, 4,
    };
    const std::array<uint8_t, kExpectedTasks> expected_group{
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 1, 1,
    };
    const std::array<uint16_t, 5> expected_outputs{3, 1, 3, 1, 0};
    for (uint32_t task = 0U; task < kExpectedTasks; ++task) {
        const DecodedPlanCellControl decoded =
            DecodePlanCellControl(cells[task].control.value);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        ok &= Check(
            decoded.valid && decoded.phase == PlanCellPhase::Published &&
            decoded.task_id == task,
            "cell control is not a canonical published identity"
        );
        ok &= Check(
            ValidateRuntimeTaskPlanPayload(
                cells[task].payload, task,
                decoded.payload_lines, header, layout
            ),
            "published payload failed canonical validation"
        );
        const uint8_t kind = expected_kind[task];
        const uint8_t group = expected_group[task];
        const bool has_following = task == 9U;
        const bool last_in_batch = task == 4U || task == 13U;
        const uint8_t expected_flags = static_cast<uint8_t>(
            0x80U | (last_in_batch ? 0x40U : 0U) |
            (has_following ? 0x20U : 0U) |
            (group << 3U) | kind
        );
        ok &= Check(
            header.task_id == task &&
            header.adapter_flags == expected_flags &&
            header.output_count == expected_outputs[kind],
            "task identity/flags/output arity mismatch"
        );
        const uint8_t expected_engine =
            kind == 0U ? 0U : ((kind == 1U || kind == 3U) ? 1U : 2U);
        const uint32_t expected_function =
            kind == 0U ? kInvalidFunctionId : kind - 1U;
        ok &= Check(
            header.engine_class == expected_engine &&
            header.function_id == expected_function,
            "engine/function registry mapping mismatch"
        );
    }

    std::free(cells_memory);
    std::free(control_memory);
    dlclose(handle);
    if (!ok) return 1;
    std::cout
        << "PASS real paged_attention orchestration SO: tasks="
        << backend_result.task_count
        << " alloc/aic/aiv=" << backend_result.alloc_count << '/'
        << backend_result.aic_count << '/' << backend_result.aiv_count
        << '\n';
    return 0;
}
