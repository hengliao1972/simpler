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
 * Paged Attention Orchestration Function V2 - N_UNROLL=64, 4 Compute Tasks Per Group
 *
 * Batches up to N_UNROLL blocks per group. Each group submits exactly 4 tasks:
 *   1. QK matmul:  qi @ K^T for n_blocks → sij_buf (q_tile, n_blocks * block_size)
 *   2. Softmax:    two-pass over sij_buf → pij_buf, mi, li
 *   3. PV matmul:  SplitK accumulated P @ V → oi_new (q_tile, head_dim)
 *   4. Update:     online softmax accumulation with group-level mi, li, oi_new
 *
 * Memory Layout:
 *   Query: (batch * num_heads, head_dim) bf16
 *   Key:   (total_blocks, block_size, head_dim) bf16 (stored as K^T for QK)
 *   Value: (total_blocks, block_size, head_dim) bf16
 */

#include <cstdint>
#include <cstring>

#include "dist_engine/common/target.h"

// `shared TensorMap` 是后端存储合同，不能等同于 legacy same-core PA
// 的专用回放协议。只有 same_core + shared 才使用固定 Case1、
// SharedOutputRef 和角色特化的快路径；跨核调度器必须走通用 Submit
// 接口，才能让构建 owner、执行 owner 和输出描述符协议保持算子无关。
#if PTO_FDWIC_SHARED_MAP && PTO_FDWIC_SCHEDULER_MODE == 0
#define PTO_FDWIC_SAME_CORE_SHARED_PA_PATH 1
#else
#define PTO_FDWIC_SAME_CORE_SHARED_PA_PATH 0
#endif

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
#if PTO_FDWIC_SHARED_PA_BUILD_ROLE == 0
#define PTO_FDWIC_SHARED_PA_EMIT_ORCHESTRATION 1
#else
#define PTO_FDWIC_SHARED_PA_EMIT_ORCHESTRATION 0
#endif
#else
#define PTO_FDWIC_SHARED_PA_EMIT_ORCHESTRATION 1
#endif

#if PTO_FDWIC_SHARED_PA_UNITY && PTO_FDWIC_SHARED_PA_EMIT_ORCHESTRATION
#define PTO_FDWIC_SHARED_PA_UNITY_IMPLEMENTATION 1
#include "dist_engine/aicore/dist_engine.cpp"
#undef PTO_FDWIC_SHARED_PA_UNITY_IMPLEMENTATION
#endif

#if PTO_FDWIC_SHARED_PA_EMIT_ORCHESTRATION

#include "pto_orchestration_api.h"

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
#define PTO_FDWIC_SHARED_PA_ALLOC_CALL(name) name<ReplayRole>
#define PTO_FDWIC_SHARED_PA_TASK_CALL(name, kind) name<ReplayRole, kind>
#define PTO_FDWIC_SHARED_PA_ALLOC_IDENTITY(expected_task_id) expected_task_id,
#define PTO_FDWIC_SHARED_PA_TASK_IDENTITY(kind, expected_task_id) expected_task_id,
#else
#define PTO_FDWIC_SHARED_PA_ALLOC_CALL(name) name
#define PTO_FDWIC_SHARED_PA_TASK_CALL(name, kind) name
#define PTO_FDWIC_SHARED_PA_ALLOC_IDENTITY(expected_task_id)
#define PTO_FDWIC_SHARED_PA_TASK_IDENTITY(kind, expected_task_id) kind,
#endif

#define N_UNROLL 64

#define FUNC_QK_MATMUL 0
#define FUNC_SOFTMAX_PREPARE 1
#define FUNC_PV_MATMUL 2
#define FUNC_ONLINE_UPDATE 3
constexpr uint64_t kPaOrchestrationProfSysCntFreq = 50000000;  // 50 MHz

PTO_DEVICE_FUNC inline uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
PTO_DEVICE_FUNC inline void init_shared_pa_create_info(
    TensorCreateInfo &info, const uint32_t shapes[], uint32_t ndims, DataType dtype
) {
    always_assert(ndims > 0 && ndims <= MAX_TENSOR_DIMS);
    info.initial_value = 0;
    info.has_initial_value = false;
    info.__pad2__ = 0;
    info.start_offset = 0;
    info.version = 0;
    info.ndims = ndims;
    info.dtype = dtype;
    info.manual_dep = false;
    info.is_contiguous = true;
    info.__pad_flags__ = 0;
    for (uint32_t index = 0; index < ndims; ++index) info.shapes[index] = shapes[index];
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
/**
 * Replay-local output symbols retained by each compiled AICore role.
 *
 * Every actor still reconstructs the authoritative SharedTaskOutputs
 * task-id/count pair after every Submit.  Only these plain symbols have a
 * later argument-builder consumer on the same actor: AIC builds PV from the
 * SF probabilities, while AIV builds SF/UP from the other seven outputs.
 */
template <CoreType ReplayRole>
struct SharedPaRoleOutputRefs;

template <>
struct SharedPaRoleOutputRefs<CoreType::AIC> {
    FdwicOutputRef sf_probs;
};

template <>
struct SharedPaRoleOutputRefs<CoreType::AIV> {
    FdwicOutputRef accumulated_output;
    FdwicOutputRef accumulated_sum;
    FdwicOutputRef accumulated_max;
    FdwicOutputRef qk_scores;
    FdwicOutputRef sf_max;
    FdwicOutputRef sf_sum;
    FdwicOutputRef pv_output;
};

static_assert(
    sizeof(SharedPaRoleOutputRefs<CoreType::AIC>) == sizeof(FdwicOutputRef),
    "AIC shared PA replay must retain exactly one output symbol"
);
static_assert(
    sizeof(SharedPaRoleOutputRefs<CoreType::AIV>) == 7 * sizeof(FdwicOutputRef),
    "AIV shared PA replay must retain exactly seven output symbols"
);

PTO_DEVICE_FUNC inline void shared_pa_report_unexpected_winner_role() {
    rt_report_fatal(
        PTO2_ERROR_TENSORMAP_PROTOCOL,
        "shared PA Claim selected a winner whose compiled role cannot execute the task"
    );
}
#endif
#endif

inline double pa_orchestration_cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / kPaOrchestrationProfSysCntFreq) * 1000000.0;
}

inline uint64_t get_sys_cnt_aicpu() {
    uint64_t ticks;
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
}

#ifdef ENABLE_PROFILING
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#else
#define CYCLE_COUNT_START() (void)0
#define CYCLE_COUNT_LAP(acc) (void)0
#endif

/**
 * Orchestration config — the executor reads these values to set up
 * shared memory and runtime before calling aicpu_orchestration_entry.
 */
extern "C" __attribute__((visibility("default"), weak)) PTO2OrchestrationConfig
aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 7,
    };
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <CoreType ReplayRole>
PTO_DEVICE_FUNC void pa_orchestration_entry_impl(const L2TaskArgs &orch_args) {
#else
extern "C" __attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void
aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
#endif
#ifdef ENABLE_PROFILING
    uint64_t prof_param_extract = 0;
    uint64_t prof_ext_tensor = 0;
    uint64_t prof_make_tensor = 0;
    uint64_t prof_tensor_view = 0;
    uint64_t prof_param_setup = 0;
    uint64_t prof_submit_task = 0;
    uint64_t prof_scope_and_loop = 0;
    int prof_submit_count = 0;
    int prof_make_count = 0;
    int prof_view_count = 0;
#endif

    CYCLE_COUNT_START();

#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
    // Runtime attach owns the authoritative AIC/AIV identity. Snapshot it
    // exactly once per replay; all 1,280 shared Claim calls reuse these
    // scalar fields instead of loading DistCore identity from GM.
    const DistSharedPaReplayContext replay = dist_shared_pa_replay_context();
    if (!replay.ready()) {
        rt_report_fatal(
            PTO2_ERROR_DIST_CONFIG_INVALID, "shared PA replay has no valid attached AICore identity"
        );
        return;
    }
#endif

    // Read dimensions from tensor metadata
    // query: shape=[batch, num_heads, head_dim]
    uint64_t batch = orch_args.tensor(0).ref().shapes[0];
#if PTO_FDWIC_PERF_CLOCK || PTO_FDWIC_SUBMIT_PMU
    // perf-clock 与 submit-pmu-none 当前只服务与 Case1 同构的 PA：每个 batch 恰好一次
    // Alloc 和一组 QK/SF/PV/UP，共 5 次 Submit。Case2/3 的 block 分组数
    // 不同；若误用该诊断构建，最终实际 count 会超过 expected，host 必须
    // fail closed，不能把中途第 5*batch 次 Submit 冒充为末次。
    rt_perf_clock_expect_submits(static_cast<uint32_t>(5 * batch));
#endif
    uint64_t num_heads = orch_args.tensor(0).ref().shapes[1];
    uint64_t head_dim = orch_args.tensor(0).ref().shapes[2];
    DataType data_type = orch_args.tensor(0).ref().dtype;

    // key_cache: shape=[total_blocks, block_size, kv_head_num, head_dim]
    uint64_t block_size = orch_args.tensor(1).ref().shapes[1];
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
    // The build/scene gate rejects every non-Case1 selection before compile.
    // Repeat the exact data contract here so a custom caller cannot bypass
    // that gate and run this specialized image with an unsupported shape.
    if (batch != kFdwicSharedPaBatches || num_heads != 16 || head_dim != 128 ||
        block_size != 128 || data_type != DataType::BFLOAT16) {
        rt_report_fatal(
            PTO2_ERROR_DIST_CONFIG_INVALID,
            "shared PA phase 1 requires Case1: batch=256, num_heads=16, "
            "head_dim=128, block_size=128, bfloat16"
        );
        return;
    }
#endif

    // block_table: shape=[batch, max_num_blocks_per_req]
    uint64_t block_num = orch_args.tensor(3).ref().shapes[1];

    // scale from scalar arg
    uint64_t scale_value = orch_args.scalar(0);
    uint64_t q_head_num = num_heads;
    uint64_t q_tile = min_u64(num_heads, static_cast<uint64_t>(128));
    uint64_t q_loop = (q_head_num + q_tile - 1) / q_tile;
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
    // Phase-1 shared PA has one fixed five-task group per batch:
    // Alloc -> QK -> SF -> PV -> UP.  Reject shapes that would submit a
    // second q-head group or exceed the fixed shared task table.
    if (q_loop != 1) {
        rt_report_fatal(
            PTO2_ERROR_DIST_CONFIG_INVALID, "shared PA phase 1 requires exactly one q-head group"
        );
        return;
    }
#endif
    CYCLE_COUNT_LAP(prof_param_extract);

    // Reshape tensors for kernel consumption (2D flattened)
    void *query_ptr = orch_args.tensor(0).ref().data_as<void>();
    void *kc_ptr = orch_args.tensor(1).ref().data_as<void>();
    void *vc_ptr = orch_args.tensor(2).ref().data_as<void>();
    void *out_ptr = orch_args.tensor(5).ref().data_as<void>();

    uint64_t total_blocks_count = orch_args.tensor(1).ref().shapes[0];

    uint32_t query_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    uint32_t key_cache_shapes[2] = {
        static_cast<uint32_t>(total_blocks_count * block_size), static_cast<uint32_t>(head_dim)
    };
    uint32_t value_cache_shapes[2] = {
        static_cast<uint32_t>(total_blocks_count * block_size), static_cast<uint32_t>(head_dim)
    };
    uint32_t out_shapes[2] = {static_cast<uint32_t>(batch * num_heads), static_cast<uint32_t>(head_dim)};
    Tensor query = make_tensor_external(query_ptr, query_shapes, 2, data_type, false);
    Tensor key_cache = make_tensor_external(kc_ptr, key_cache_shapes, 2, data_type, false);
    Tensor value_cache = make_tensor_external(vc_ptr, value_cache_shapes, 2, data_type, false);
    Tensor out = make_tensor_external(out_ptr, out_shapes, 2, DataType::FLOAT32);

    uint32_t bt_shapes[2] = {static_cast<uint32_t>(batch), static_cast<uint32_t>(block_num)};
    Tensor block_table =
        make_tensor_external(orch_args.tensor(3).ref().data_as<void>(), bt_shapes, 2, DataType::INT32, false);
    uint32_t cl_shapes[1] = {static_cast<uint32_t>(batch)};
    Tensor context_lens =
        make_tensor_external(orch_args.tensor(4).ref().data_as<void>(), cl_shapes, 1, DataType::INT32, false);
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
    // The standalone shared fast path carries the same immutable contiguous
    // backing pointer explicitly. Avoid rebuilding a Tensor scalar-access
    // request on every batch and every replay actor; each iteration still
    // performs one volatile GM load of the real context length.
    __gm__ const volatile int32_t *context_lens_data =
        reinterpret_cast<__gm__ const volatile int32_t *>(context_lens.buffer.addr);
#endif

#ifdef ENABLE_PROFILING
    CYCLE_COUNT_LAP(prof_ext_tensor);
#endif

    // Create infos are loop-invariant — shapes depend only on q_tile/head_dim
    uint32_t oi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
    uint32_t li_shapes[1] = {static_cast<uint32_t>(q_tile)};
    TensorCreateInfo tile2d_ci(oi_shapes, 2, DataType::FLOAT32);
    TensorCreateInfo scalar_ci(li_shapes, 1, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
    prof_make_count += 2;
    CYCLE_COUNT_LAP(prof_make_tensor);
#endif

#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
    // Shared callbacks consume this object synchronously and every winner
    // resets it before building args. Keep one replay-local instance instead
    // of value-initializing the fixed tag array once per batch on all 96
    // actors. Private scope/lifetime behavior remains unchanged below.
    L0TaskArgs params;
#endif
    for (uint64_t b_idx = 0; b_idx < batch; b_idx++) {
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
        uint64_t cur_seq = static_cast<uint64_t>(context_lens_data[b_idx]);
#else
        uint32_t cl_idx[1] = {static_cast<uint32_t>(b_idx)};
        uint64_t cur_seq = static_cast<uint64_t>(get_tensor_data<int32_t>(context_lens, 1, cl_idx));
#endif
        uint64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
        // N_UNROLL is 64, so this also guarantees exactly one block group and
        // therefore exactly five shared tasks for this batch.
        if (bn_this_batch == 0 || bn_this_batch > N_UNROLL) {
            rt_report_fatal(
                PTO2_ERROR_DIST_CONFIG_INVALID, "shared PA phase 1 requires 1..64 KV-cache blocks per batch"
            );
            return;
        }
#endif
        for (uint64_t q_idx = 0; q_idx < q_loop; q_idx++) {
            CYCLE_COUNT_LAP(prof_scope_and_loop);
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
            // The phase-1 shared backend owns its complete task lifetime and
            // its AICore scope hooks are deliberate no-ops. Keep only the C++
            // lifetime block instead of crossing the empty scope ABI 256 times
            // per replay actor.
            {
#else
            PTO2_SCOPE() {
#endif
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                // Shared PA constructs task-local views only in the Claim
                // winner callback. These descriptors remain in the enclosing
                // scope until the synchronous Finish consumes them.
                Tensor qi;
                Tensor out_view;
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                // The Case1 gate proves one five-task group per batch. Carry
                // its exact task-id prefix into each typed Submit so all 96
                // replay actors do an equality check instead of rebuilding
                // task kind from task_id%5 on the Claim hot path.
                const int32_t shared_batch_task_start =
                    static_cast<int32_t>(b_idx * kFdwicSharedPaTasksPerBatch);
                // Deliberately uninitialized: each retained symbol is
                // assigned by an earlier successful Submit in this task
                // group before its role-matched winner callback consumes it.
                SharedPaRoleOutputRefs<ReplayRole> role_outputs;
#endif
#else
                uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;
                uint32_t qi_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor qi = Tensor::view(query, qi_shapes, qi_offsets);
                uint32_t out_view_shapes[2] = {static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)};
                uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                Tensor out_view = Tensor::view(out, out_view_shapes, out_view_offsets, true);
#endif
#ifdef ENABLE_PROFILING
#if !PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                prof_view_count += 2;
#endif
                CYCLE_COUNT_LAP(prof_tensor_view);
#endif
                // Compete-first helpers synchronously run Claim before invoking
                // the eager argument builder, then consume params in Finish.
                // Private mode retains its replay-wide argument construction;
                // shared mode invokes the builder only on the Claim winner.
#if !PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                L0TaskArgs params;
#endif
                CYCLE_COUNT_LAP(prof_param_setup);
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                SharedTaskOutputs alloc_outs =
                    PTO_FDWIC_SHARED_PA_ALLOC_CALL(shared_pa_alloc_tensors_compete_first)(
                        replay, PTO_FDWIC_SHARED_PA_ALLOC_IDENTITY(shared_batch_task_start)
                        params, [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
                            // Alloc has no executable lane.  Its two-level
                            // tournament deliberately admits all 96 workers,
                            // so either compiled role must be able to build
                            // the same three output descriptors when it wins.
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            submit_args.add_output(tile2d_ci);
                            submit_args.add_output(scalar_ci);
                            submit_args.add_output(scalar_ci);
                            CYCLE_COUNT_LAP(prof_param_setup);
                        }
                    );
                if (alloc_outs.size() != 3) return;
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                if constexpr (ReplayRole == CoreType::AIV) {
                    role_outputs.accumulated_output = alloc_outs.output_ref(0);
                    role_outputs.accumulated_sum = alloc_outs.output_ref(1);
                    role_outputs.accumulated_max = alloc_outs.output_ref(2);
                }
#else
                FdwicOutputRef oi = alloc_outs.output_ref(0);
                FdwicOutputRef li_update = alloc_outs.output_ref(1);
                FdwicOutputRef mi_update = alloc_outs.output_ref(2);
#endif
#else
                TaskOutputTensors alloc_outs = alloc_tensors_compete_first(
                    params,
                    [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
                        CYCLE_COUNT_LAP(prof_submit_task);
                        submit_args.add_output(tile2d_ci);
                        submit_args.add_output(scalar_ci);
                        submit_args.add_output(scalar_ci);
                        CYCLE_COUNT_LAP(prof_param_setup);
                    }
                );
                __gm__ const Tensor &oi = alloc_outs.get_ref(0);
                __gm__ const Tensor &li_update = alloc_outs.get_ref(1);
                __gm__ const Tensor &mi_update = alloc_outs.get_ref(2);
#endif
#ifdef ENABLE_PROFILING
                prof_submit_count++;
                CYCLE_COUNT_LAP(prof_submit_task);
#endif

                for (uint64_t bn = 0; bn < bn_this_batch; bn += N_UNROLL) {
#if !PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    uint64_t n_blocks = min_u64(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);

                    // Valid length for last block in this group
                    uint64_t last_block_seq_start = (bn + n_blocks - 1) * block_size;
                    uint64_t valid_len_last = min_u64(block_size, cur_seq - last_block_seq_start);
                    CYCLE_COUNT_LAP(prof_param_extract);
#endif

                    // === Task 1: Batched QK matmul ===
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    TensorCreateInfo sij_buf_ci;
#else
                    uint32_t sij_buf_shapes[2] = {
                        static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)
                    };
                    TensorCreateInfo sij_buf_ci(sij_buf_shapes, 2, DataType::FLOAT32);
#ifdef ENABLE_PROFILING
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);
#endif
#endif

#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    SharedTaskOutputs qk_outs = PTO_FDWIC_SHARED_PA_TASK_CALL(
                        shared_pa_submit_aic_compete_first, DistSharedPaTaskKind::Qk
                    )(
                        replay, PTO_FDWIC_SHARED_PA_TASK_IDENTITY(
                            DistSharedPaTaskKind::Qk, shared_batch_task_start + 1
                        ) FUNC_QK_MATMUL, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            if constexpr (ReplayRole == CoreType::AIC) {
#endif
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            const uint64_t n_blocks =
                                min_u64(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);
                            const uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;
                            uint32_t sij_buf_shapes[2] = {
                                static_cast<uint32_t>(q_tile),
                                static_cast<uint32_t>(n_blocks * block_size)
                            };
                            CYCLE_COUNT_LAP(prof_param_extract);
                            uint32_t qi_shapes[2] = {
                                static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)
                            };
                            uint32_t qi_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                            qi = Tensor::view(query, qi_shapes, qi_offsets);
                            init_shared_pa_create_info(
                                sij_buf_ci, sij_buf_shapes, 2, DataType::FLOAT32
                            );
                            submit_args.add_input(qi, key_cache, block_table);
                            submit_args.add_output(sij_buf_ci);
                            submit_args.add_scalar(n_blocks, b_idx * block_num + bn);
#ifdef ENABLE_PROFILING
                            ++prof_view_count;
                            ++prof_make_count;
#endif
                            CYCLE_COUNT_LAP(prof_param_setup);
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            } else {
                                shared_pa_report_unexpected_winner_role();
                            }
#endif
                        }
                    );
                    if (qk_outs.size() != 1) return;
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                    if constexpr (ReplayRole == CoreType::AIV) {
                        role_outputs.qk_scores = qk_outs.output_ref(0);
                    }
#else
                    FdwicOutputRef sij_buf = qk_outs.output_ref(0);
#endif
#else
                    TaskOutputTensors qk_outs = rt_submit_aic_task_compete_first(
                        FUNC_QK_MATMUL, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            submit_args.add_input(qi, key_cache, block_table);
                            submit_args.add_output(sij_buf_ci);
                            submit_args.add_scalar(n_blocks, b_idx * block_num + bn);
                            CYCLE_COUNT_LAP(prof_param_setup);
                        }
                    );
                    __gm__ const Tensor &sij_buf = qk_outs.get_ref(0);
#endif
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 2: Two-pass softmax over all blocks in group ===
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    TensorCreateInfo pij_buf_ci;
#else
                    uint32_t pij_buf_shapes[2] = {
                        static_cast<uint32_t>(q_tile), static_cast<uint32_t>(n_blocks * block_size)
                    };
                    TensorCreateInfo pij_buf_ci(pij_buf_shapes, 2, data_type);
#ifdef ENABLE_PROFILING
                    prof_make_count += 1;
                    CYCLE_COUNT_LAP(prof_make_tensor);
#endif
#endif

#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    SharedTaskOutputs sf_outs = PTO_FDWIC_SHARED_PA_TASK_CALL(
                        shared_pa_submit_aiv_compete_first, DistSharedPaTaskKind::Sf
                    )(
                        replay, PTO_FDWIC_SHARED_PA_TASK_IDENTITY(
                            DistSharedPaTaskKind::Sf, shared_batch_task_start + 2
                        ) FUNC_SOFTMAX_PREPARE, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            if constexpr (ReplayRole == CoreType::AIV) {
#endif
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            const uint64_t n_blocks =
                                min_u64(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);
                            const uint64_t last_block_seq_start =
                                (bn + n_blocks - 1) * block_size;
                            const uint64_t valid_len_last =
                                min_u64(block_size, cur_seq - last_block_seq_start);
                            uint32_t pij_buf_shapes[2] = {
                                static_cast<uint32_t>(q_tile),
                                static_cast<uint32_t>(n_blocks * block_size)
                            };
                            CYCLE_COUNT_LAP(prof_param_extract);
                            init_shared_pa_create_info(
                                pij_buf_ci, pij_buf_shapes, 2, data_type
                            );
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            submit_args.add_input(role_outputs.qk_scores);
#else
                            submit_args.add_input(sij_buf);
#endif
                            submit_args.add_output(pij_buf_ci, scalar_ci, scalar_ci);
                            submit_args.add_scalar(scale_value, n_blocks, valid_len_last);
#ifdef ENABLE_PROFILING
                            ++prof_make_count;
#endif
                            CYCLE_COUNT_LAP(prof_param_setup);
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            } else {
                                shared_pa_report_unexpected_winner_role();
                            }
#endif
                        }
                    );
                    if (sf_outs.size() != 3) return;
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                    if constexpr (ReplayRole == CoreType::AIC) {
                        role_outputs.sf_probs = sf_outs.output_ref(0);
                    } else {
                        role_outputs.sf_max = sf_outs.output_ref(1);
                        role_outputs.sf_sum = sf_outs.output_ref(2);
                    }
#else
                    FdwicOutputRef pij_buf = sf_outs.output_ref(0);
                    FdwicOutputRef mi = sf_outs.output_ref(1);
                    FdwicOutputRef li = sf_outs.output_ref(2);
#endif
#else
                    TaskOutputTensors sf_outs = rt_submit_aiv_task_compete_first(
                        FUNC_SOFTMAX_PREPARE, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            submit_args.add_input(sij_buf);
                            submit_args.add_output(pij_buf_ci, scalar_ci, scalar_ci);
                            submit_args.add_scalar(scale_value, n_blocks, valid_len_last);
                            CYCLE_COUNT_LAP(prof_param_setup);
                        }
                    );
                    __gm__ const Tensor &pij_buf = sf_outs.get_ref(0);
                    __gm__ const Tensor &mi = sf_outs.get_ref(1);
                    __gm__ const Tensor &li = sf_outs.get_ref(2);
#endif
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 3: SplitK PV matmul (accumulated P @ V) ===
#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    SharedTaskOutputs pv_outs = PTO_FDWIC_SHARED_PA_TASK_CALL(
                        shared_pa_submit_aic_compete_first, DistSharedPaTaskKind::Pv
                    )(
                        replay, PTO_FDWIC_SHARED_PA_TASK_IDENTITY(
                            DistSharedPaTaskKind::Pv, shared_batch_task_start + 3
                        ) FUNC_PV_MATMUL, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            if constexpr (ReplayRole == CoreType::AIC) {
#endif
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            const uint64_t n_blocks =
                                min_u64(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);
                            CYCLE_COUNT_LAP(prof_param_extract);
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            submit_args.add_input(
                                role_outputs.sf_probs, value_cache, block_table
                            );
#else
                            submit_args.add_input(pij_buf, value_cache, block_table);
#endif
                            submit_args.add_output(tile2d_ci);
                            submit_args.add_scalar(n_blocks, b_idx * block_num + bn);
                            CYCLE_COUNT_LAP(prof_param_setup);
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            } else {
                                shared_pa_report_unexpected_winner_role();
                            }
#endif
                        }
                    );
                    if (pv_outs.size() != 1) return;
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                    if constexpr (ReplayRole == CoreType::AIV) {
                        role_outputs.pv_output = pv_outs.output_ref(0);
                    }
#else
                    FdwicOutputRef oi_new = pv_outs.output_ref(0);
#endif
#else
                    TaskOutputTensors pv_outs = rt_submit_aic_task_compete_first(
                        FUNC_PV_MATMUL, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            submit_args.add_input(pij_buf, value_cache, block_table);
                            submit_args.add_output(tile2d_ci);
                            submit_args.add_scalar(n_blocks, b_idx * block_num + bn);
                            CYCLE_COUNT_LAP(prof_param_setup);
                        }
                    );
                    __gm__ const Tensor &oi_new = pv_outs.get_ref(0);
#endif
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif

                    // === Task 4: Online update (per-group) ===
#if !PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    uint64_t is_first = (bn == 0) ? 1 : 0;
                    uint64_t is_last = (bn + n_blocks >= bn_this_batch) ? 1 : 0;
                    CYCLE_COUNT_LAP(prof_param_setup);
#endif

#if PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
                    SharedTaskOutputs up_outs = PTO_FDWIC_SHARED_PA_TASK_CALL(
                        shared_pa_submit_aiv_compete_first, DistSharedPaTaskKind::Up
                    )(
                        replay, PTO_FDWIC_SHARED_PA_TASK_IDENTITY(
                            DistSharedPaTaskKind::Up, shared_batch_task_start + 4
                        ) FUNC_ONLINE_UPDATE, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            if constexpr (ReplayRole == CoreType::AIV) {
#endif
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            const uint64_t n_blocks =
                                min_u64(static_cast<uint64_t>(N_UNROLL), bn_this_batch - bn);
                            const uint64_t cur_offset = b_idx * q_head_num + q_idx * q_tile;
                            const uint64_t is_first = (bn == 0) ? 1 : 0;
                            const uint64_t is_last =
                                (bn + n_blocks >= bn_this_batch) ? 1 : 0;
                            CYCLE_COUNT_LAP(prof_param_extract);
                            uint32_t out_view_shapes[2] = {
                                static_cast<uint32_t>(q_tile), static_cast<uint32_t>(head_dim)
                            };
                            uint32_t out_view_offsets[2] = {static_cast<uint32_t>(cur_offset), 0};
                            out_view = Tensor::view(out, out_view_shapes, out_view_offsets, true);
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            submit_args.add_input(
                                role_outputs.sf_max, role_outputs.sf_sum,
                                role_outputs.pv_output
                            );
                            submit_args.add_inout(
                                role_outputs.accumulated_max,
                                role_outputs.accumulated_sum,
                                role_outputs.accumulated_output, out_view
                            );
#else
                            submit_args.add_input(mi, li, oi_new);
                            submit_args.add_inout(mi_update, li_update, oi, out_view);
#endif
                            submit_args.add_scalar(is_first, is_last);
#ifdef ENABLE_PROFILING
                            ++prof_view_count;
#endif
                            CYCLE_COUNT_LAP(prof_param_setup);
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
                            } else {
                                shared_pa_report_unexpected_winner_role();
                            }
#endif
                        }
                    );
                    if (up_outs.producer_task_id < 0 || !up_outs.empty()) return;
#else
                    rt_submit_aiv_task_compete_first(
                        FUNC_ONLINE_UPDATE, params,
                        [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
                            CYCLE_COUNT_LAP(prof_submit_task);
                            submit_args.reset();
                            submit_args.add_input(mi, li, oi_new);
                            submit_args.add_inout(mi_update, li_update, oi, out_view);
                            submit_args.add_scalar(is_first, is_last);
                            CYCLE_COUNT_LAP(prof_param_setup);
                        }
                    );
#endif
#ifdef ENABLE_PROFILING
                    prof_submit_count++;
                    CYCLE_COUNT_LAP(prof_submit_task);
#endif
                }
            }
            CYCLE_COUNT_LAP(prof_scope_and_loop);
        }
    }
    CYCLE_COUNT_LAP(prof_scope_and_loop);

#ifdef ENABLE_PROFILING
    uint64_t total = prof_param_extract + prof_ext_tensor + prof_make_tensor + prof_tensor_view + prof_param_setup +
                     prof_submit_task + prof_scope_and_loop;
    LOG_INFO_V9(
        "=== PagedAttn Orch Profiling: %d submits, %d makes, %d views, total=%.3fus ===", prof_submit_count,
        prof_make_count, prof_view_count, pa_orchestration_cycles_to_us(total)
    );
    if (total > 0) {
        LOG_INFO_V9(
            "  param_extract    : %7.3fus (%5.1f%%)", pa_orchestration_cycles_to_us(prof_param_extract),
            prof_param_extract * 100.0 / total
        );
        LOG_INFO_V9(
            "  ext_tensor(x4)   : %7.3fus (%5.1f%%)", pa_orchestration_cycles_to_us(prof_ext_tensor),
            prof_ext_tensor * 100.0 / total
        );
        LOG_INFO_V9(
            "  create_info(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", prof_make_count,
            pa_orchestration_cycles_to_us(prof_make_tensor),
            prof_make_tensor * 100.0 / total,
            prof_make_count > 0 ? pa_orchestration_cycles_to_us(prof_make_tensor) / prof_make_count : 0.0
        );
        LOG_INFO_V9(
            "  tensor_view(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", prof_view_count,
            pa_orchestration_cycles_to_us(prof_tensor_view),
            prof_tensor_view * 100.0 / total,
            prof_view_count > 0 ? pa_orchestration_cycles_to_us(prof_tensor_view) / prof_view_count : 0.0
        );
        LOG_INFO_V9(
            "  param_setup      : %7.3fus (%5.1f%%)", pa_orchestration_cycles_to_us(prof_param_setup),
            prof_param_setup * 100.0 / total
        );
        LOG_INFO_V9(
            "  submit_task(x%d) : %7.3fus (%5.1f%%)  avg=%.3fus", prof_submit_count,
            pa_orchestration_cycles_to_us(prof_submit_task),
            prof_submit_task * 100.0 / total,
            prof_submit_count > 0 ? pa_orchestration_cycles_to_us(prof_submit_task) / prof_submit_count : 0.0
        );
        LOG_INFO_V9(
            "  scope_and_loop   : %7.3fus (%5.1f%%)", pa_orchestration_cycles_to_us(prof_scope_and_loop),
            prof_scope_and_loop * 100.0 / total
        );
    }
#endif

#undef CYCLE_COUNT_START
#undef CYCLE_COUNT_LAP
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
extern "C" __attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void
aicpu_orchestration_entry_aic(const L2TaskArgs &orch_args) {
    pa_orchestration_entry_impl<CoreType::AIC>(orch_args);
}

extern "C" __attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void
aicpu_orchestration_entry_aiv(const L2TaskArgs &orch_args) {
    pa_orchestration_entry_impl<CoreType::AIV>(orch_args);
}
#endif

#undef PTO_FDWIC_SHARED_PA_TASK_IDENTITY
#undef PTO_FDWIC_SHARED_PA_ALLOC_IDENTITY
#undef PTO_FDWIC_SHARED_PA_TASK_CALL
#undef PTO_FDWIC_SHARED_PA_ALLOC_CALL

#endif  // PTO_FDWIC_SHARED_PA_EMIT_ORCHESTRATION

#undef PTO_FDWIC_SHARED_PA_EMIT_ORCHESTRATION
#undef PTO_FDWIC_SAME_CORE_SHARED_PA_PATH
