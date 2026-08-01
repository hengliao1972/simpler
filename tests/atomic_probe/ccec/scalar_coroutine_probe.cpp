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

// 独立验证“保存 continuation → 发射 engine → scalar 推进 loser Submit →
// 恢复 continuation → 最终 wait → 发布完成”的基础合同。这里不 include
// pa_scheduler，也不修改正式调度路径；AIC/AIV 由同一源文件分别编译。

#include <pto/common/constants.hpp>
#include <pto/common/kernel_meta.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "scalar_coroutine_probe_shared.h"
#include "scalar_pmu_device.h"
#include "ccec_utils.h"

#if defined(SCALAR_COROUTINE_BUILD_AIC)
PTO_SYNCALL_AIC_KERNEL_META(scalar_coroutine_probe_0_mix_aic);
#define SCALAR_COROUTINE_ENTRY scalar_coroutine_probe_0_mix_aic
#elif defined(SCALAR_COROUTINE_BUILD_AIV)
PTO_SYNCALL_AIV_KERNEL_META(scalar_coroutine_probe_0_mix_aiv);
#define SCALAR_COROUTINE_ENTRY scalar_coroutine_probe_0_mix_aiv
#else
#error "Compile with SCALAR_COROUTINE_BUILD_AIC or SCALAR_COROUTINE_BUILD_AIV"
#endif

namespace {

using namespace pto;
using scalar_coroutine_probe::ContinuationFrame;

// A5 的 BUFFER_ID 模式查询的是 get_buf/rls_buf 协议的未完成数量，
// 不是 set_flag/wait_flag 的 EVENT_ID。选用有效范围末端的独立 ID，
// 避免与本探针原有的 EVENT_ID0/EVENT_ID7 混为同一状态源。
constexpr uint8_t kTryWaitBufferId = 31U;

struct RuntimeContext {
    uint64_t words[scalar_coroutine_probe::kContextWords];
};

__aicore__ __attribute__((always_inline)) inline uint64_t ContextSignature(
    const RuntimeContext &context
) {
    uint64_t signature = 0x6a09e667f3bcc909ULL;
    for (uint32_t index = 0;
         index < scalar_coroutine_probe::kContextWords; ++index) {
        signature = scalar_coroutine_probe::Mix(
            signature ^ context.words[index] ^
            (static_cast<uint64_t>(index) << 56U)
        );
    }
    return signature;
}

__aicore__ __attribute__((always_inline)) inline RuntimeContext MakeContext(
    uint32_t seed, uint64_t task_id
) {
    RuntimeContext context{};
    for (uint32_t index = 0;
         index < scalar_coroutine_probe::kContextWords; ++index) {
        context.words[index] = scalar_coroutine_probe::Mix(
            (static_cast<uint64_t>(seed) << 32U) ^ task_id ^
            (static_cast<uint64_t>(index + 1U) *
             0x9e3779b97f4a7c15ULL)
        );
    }
    context.words[0] = task_id;
    return context;
}

__aicore__ __attribute__((noinline, used)) void SaveContinuation(
    volatile ContinuationFrame *frame, const RuntimeContext &context,
    uint32_t generation
) {
    for (uint32_t index = 0;
         index < scalar_coroutine_probe::kContextWords; ++index) {
        frame->words[index] = context.words[index];
    }
    frame->signature = ContextSignature(context);
    frame->generation = generation;
    frame->state = static_cast<uint32_t>(
        scalar_coroutine_probe::ContinuationState::Suspended
    );
    asm volatile("" ::: "memory");
}

__aicore__ __attribute__((noinline, used)) RuntimeContext RestoreContinuation(
    volatile ContinuationFrame *frame, uint32_t expected_generation,
    bool &state_ok
) {
    asm volatile("" ::: "memory");
    RuntimeContext context{};
    state_ok =
        frame->state == static_cast<uint32_t>(
            scalar_coroutine_probe::ContinuationState::Suspended
        ) && frame->generation == expected_generation;
    for (uint32_t index = 0;
         index < scalar_coroutine_probe::kContextWords; ++index) {
        context.words[index] = frame->words[index];
    }
    state_ok = state_ok &&
        ContextSignature(context) == frame->signature;
    frame->state = static_cast<uint32_t>(
        scalar_coroutine_probe::ContinuationState::Resumed
    );
    asm volatile("" ::: "memory");
    return context;
}

__aicore__ __attribute__((noinline, used)) uint64_t ReplayLoserSubmitsUntilSecondWinner(
    volatile __gm__ const scalar_coroutine_probe::TaskRecord *records,
    uint32_t iterations, uint32_t seed,
    volatile ContinuationFrame *second_winner
) {
    uint32_t cursor = seed & (scalar_coroutine_probe::kTaskRecords - 1U);
    uint64_t checksum = scalar_coroutine_probe::Mix(
        static_cast<uint64_t>(seed) ^ 0xbb67ae8584caa73bULL
    );
    // 每轮读取一个独占 64B synthetic task cell，并执行与 loser Submit
    // 类似的 cursor、ready 分支和参数摘要更新。串行/重叠模式都只调用
    // 这一份 noinline 机器码，避免再把后端展开差异误判成重叠收益。
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const volatile __gm__ scalar_coroutine_probe::TaskRecord &record =
            records[cursor];
        const uint64_t task_id = record.task_id;
        const uint64_t ready = record.ready_token;
        const uint64_t payload = record.payload[iteration % 5U];
        checksum = scalar_coroutine_probe::Mix(
            checksum ^ task_id ^ payload ^
            (ready + static_cast<uint64_t>(iteration))
        );
        const uint32_t delta = static_cast<uint32_t>(record.next_delta);
        cursor = (cursor + 1U + delta +
                  static_cast<uint32_t>((ready ^ checksum) & 1ULL)) &
                 (scalar_coroutine_probe::kTaskRecords - 1U);
    }

    // 前面的 iterations 次都代表 loser；到这里模拟第一次再次 win，
    // 保存第二份 continuation 后立即停止 replay，不产生无界上下文。
    RuntimeContext second = MakeContext(
        seed ^ static_cast<uint32_t>(checksum),
        0x200000000ULL | cursor
    );
    second.words[1] ^= checksum;
    SaveContinuation(second_winner, second, /*generation=*/2U);
    return checksum;
}

#if defined(SCALAR_COROUTINE_BUILD_AIV)
template <typename GlobalData, typename TileData>
__aicore__ __attribute__((always_inline)) inline void IssueEngine(
    GlobalData &input_a_global, GlobalData &input_b_global,
    GlobalData &output_global, TileData &input_a_tile,
    TileData &input_b_tile, TileData &output_tile, bool tag_final_store,
    uint32_t dynamic_buffer_count, uint32_t runtime_buffer_id
) {
    TLOAD(input_a_tile, input_a_global);
    TLOAD(input_b_tile, input_b_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(output_tile, input_a_tile, input_b_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if (tag_final_store) {
        if (dynamic_buffer_count != 0U) {
            for (uint32_t slot = 0U;
                 slot < dynamic_buffer_count; ++slot) {
                get_buf(PIPE_MTE3, runtime_buffer_id + slot, false);
            }
        } else {
            get_buf(PIPE_MTE3, kTryWaitBufferId, false);
        }
    }
    TSTORE(output_global, output_tile);
    if (tag_final_store) {
        if (dynamic_buffer_count != 0U) {
            for (uint32_t slot = 0U;
                 slot < dynamic_buffer_count; ++slot) {
                rls_buf(PIPE_MTE3, runtime_buffer_id + slot, false);
            }
        } else {
            rls_buf(PIPE_MTE3, kTryWaitBufferId, false);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

__aicore__ __attribute__((always_inline)) inline void WaitEngine() {
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}
#else
template <typename GlobalData, typename TileMatA, typename TileMatB,
          typename LeftTile, typename RightTile, typename AccTile>
__aicore__ __attribute__((always_inline)) inline void IssueEngine(
    GlobalData &input_a_global, GlobalData &input_b_global,
    GlobalData &output_global, TileMatA &input_a_mat,
    TileMatB &input_b_mat, LeftTile &input_a_l0,
    RightTile &input_b_l0, AccTile &output_l0, bool tag_final_store,
    uint32_t dynamic_buffer_count, uint32_t runtime_buffer_id
) {
    TLOAD(input_a_mat, input_a_global);
    TLOAD(input_b_mat, input_b_global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(input_a_l0, input_a_mat);
    TMOV(input_b_l0, input_b_mat);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(output_l0, input_a_l0, input_b_l0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    if (tag_final_store) {
        if (dynamic_buffer_count != 0U) {
            for (uint32_t slot = 0U;
                 slot < dynamic_buffer_count; ++slot) {
                get_buf(PIPE_FIX, runtime_buffer_id + slot, false);
            }
        } else {
            get_buf(PIPE_FIX, kTryWaitBufferId, false);
        }
    }
    TSTORE(output_global, output_l0);
    if (tag_final_store) {
        if (dynamic_buffer_count != 0U) {
            for (uint32_t slot = 0U;
                 slot < dynamic_buffer_count; ++slot) {
                rls_buf(PIPE_FIX, runtime_buffer_id + slot, false);
            }
        } else {
            rls_buf(PIPE_FIX, kTryWaitBufferId, false);
        }
    }
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

__aicore__ __attribute__((always_inline)) inline void WaitEngine() {
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}
#endif

__aicore__ __attribute__((always_inline)) inline int64_t SampleTryWait(
    scalar_coroutine_probe::Mode mode, uint32_t runtime_buffer_id
) {
    // builtin 的第二个参数必须是编译期常量，因此按 mode 保留明确分支，
    // 不能把 sync_mode_t 当作运行时参数传入。前三项是 EVENT_ID7 负对照；
    // 其余项查询显式 get_buf/rls_buf 所绑定的 buffer id。
    switch (mode) {
    case scalar_coroutine_probe::Mode::TryWaitCrossCore:
        return try_wait(EVENT_ID7, CROSS_CORE);
    case scalar_coroutine_probe::Mode::TryWaitIntraBlock:
        return try_wait(EVENT_ID7, INTRA_BLOCK);
    case scalar_coroutine_probe::Mode::TryWaitBufferId:
        return try_wait(EVENT_ID7, BUFFER_ID);
    case scalar_coroutine_probe::Mode::TryWaitTaggedBufferId:
        return try_wait(kTryWaitBufferId, BUFFER_ID);
    case scalar_coroutine_probe::Mode::TryWaitTaggedDynamicBufferId:
        return try_wait(runtime_buffer_id, BUFFER_ID);
    case scalar_coroutine_probe::Mode::TryWaitPollUntilDone:
        return try_wait(kTryWaitBufferId, BUFFER_ID);
    case scalar_coroutine_probe::Mode::TryWaitFourSlots: {
        int64_t pending_mask = 0;
        for (uint32_t slot = 0U; slot < 4U; ++slot) {
            if (try_wait(runtime_buffer_id + slot, BUFFER_ID) != 0) {
                pending_mask |= 1LL << slot;
            }
        }
        return pending_mask;
    }
    case scalar_coroutine_probe::Mode::TryWaitIdleCost:
        return try_wait(runtime_buffer_id, BUFFER_ID);
    default:
        return 0;
    }
}

__aicore__ __attribute__((noinline, used)) int64_t MeasureIdleTryWait(
    uint32_t runtime_buffer_id, uint32_t iterations
) {
    int64_t accumulator = 0;
    for (uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        accumulator ^= try_wait(runtime_buffer_id, BUFFER_ID);
    }
    return accumulator;
}

}  // namespace

extern "C" __global__ __aicore__ void SCALAR_COROUTINE_ENTRY(
    __gm__ scalar_coroutine_probe::ProbeState *state
) {
    using scalar_coroutine_probe::Mode;
    using atomic_probe::scalar_pmu::Publish64;

    bisheng::cce::metrics_prof_stop();
    dcci(&state->control, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);
    // CCEC 不允许从 GM 地址空间直接拷贝构造本地结构体。逐字段读取也能
    // 明确约束：控制块只在 PMU 窗口外读取一次，后续测量不再访问它。
    scalar_coroutine_probe::ProbeControl control{};
    control.pmu_register_bases = state->control.pmu_register_bases;
    control.input_a = state->control.input_a;
    control.input_b = state->control.input_b;
    control.output = state->control.output;
    control.task_records = state->control.task_records;
    control.mode = state->control.mode;
    control.schedule_iterations = state->control.schedule_iterations;
    control.seed = state->control.seed;
    control.magic = state->control.magic;
    control.try_wait_buffer_id = state->control.try_wait_buffer_id;
    if (control.magic != scalar_coroutine_probe::kControlMagic ||
        control.mode >= static_cast<uint32_t>(Mode::Count) ||
        control.task_records == 0U) {
        return;
    }

    const uint32_t physical_core_id =
        static_cast<uint32_t>(get_coreid()) & 0x0fffU;
    uint64_t register_base = 0U;
    if (control.pmu_register_bases != 0U &&
        physical_core_id < atomic_probe::scalar_pmu::kPhysicalSubcores) {
        __gm__ uint64_t *bases = reinterpret_cast<__gm__ uint64_t *>(
            control.pmu_register_bases
        );
        register_base = bases[physical_core_id];
    }

    constexpr int kRows =
        static_cast<int>(scalar_coroutine_probe::kTileRows);
    constexpr int kCols =
        static_cast<int>(scalar_coroutine_probe::kTileCols);
    using GlobalData = GlobalTensor<
        float, Shape<1, 1, 1, kRows, kCols>,
        pto::Stride<kRows * kCols, kRows * kCols, kRows * kCols,
                    kCols, 1>>;
    GlobalData input_a_global(
        reinterpret_cast<__gm__ float *>(control.input_a)
    );
    GlobalData input_b_global(
        reinterpret_cast<__gm__ float *>(control.input_b)
    );
    GlobalData output_global(
        reinterpret_cast<__gm__ float *>(control.output)
    );

#if defined(SCALAR_COROUTINE_BUILD_AIV)
    using TileData = Tile<
        TileType::Vec, float, kRows, kCols,
        BLayout::RowMajor, -1, -1>;
    TileData input_a_tile(kRows, kCols);
    TileData input_b_tile(kRows, kCols);
    TileData output_tile(kRows, kCols);
    TASSIGN(input_a_tile, 0x0);
    TASSIGN(input_b_tile, 0x10000);
    TASSIGN(output_tile, 0x20000);
#else
    constexpr int kBlockAlign = C0_SIZE_BYTE / sizeof(float);
    static_assert(kRows % 16 == 0, "cube M must be 16-aligned");
    static_assert(kCols % kBlockAlign == 0, "cube K/N alignment changed");
    using TileMatA = Tile<
        TileType::Mat, float, kRows, kCols, BLayout::ColMajor,
        kRows, kCols, SLayout::RowMajor, 512>;
    using TileMatB = Tile<
        TileType::Mat, float, kRows, kCols, BLayout::ColMajor,
        kRows, kCols, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, kRows, kCols, kRows, kCols>;
    using RightTile = TileRight<float, kRows, kCols, kRows, kCols>;
    using AccTile = TileAcc<float, kRows, kCols, kRows, kCols>;
    TileMatA input_a_mat;
    TileMatB input_b_mat;
    LeftTile input_a_l0;
    RightTile input_b_l0;
    AccTile output_l0;
    TASSIGN(input_a_mat, 0x0);
    TASSIGN(input_b_mat, 0x20000);
    TASSIGN(input_a_l0, 0x0);
    TASSIGN(input_b_l0, 0x0);
    TASSIGN(output_l0, 0x0);
#endif

    const Mode mode = static_cast<Mode>(control.mode);
    const bool idle_cost_mode = mode == Mode::TryWaitIdleCost;
    const bool use_engine =
        mode != Mode::ContextFifo && !idle_cost_mode;
    const bool serial = mode == Mode::EngineThenSchedule;
    const bool sample_try_wait =
        mode == Mode::TryWaitCrossCore ||
        mode == Mode::TryWaitIntraBlock ||
        mode == Mode::TryWaitBufferId ||
        mode == Mode::TryWaitTaggedBufferId ||
        mode == Mode::TryWaitTaggedDynamicBufferId ||
        mode == Mode::TryWaitPollUntilDone ||
        mode == Mode::TryWaitFourSlots;
    const bool tag_final_store =
        mode == Mode::TryWaitTaggedBufferId ||
        mode == Mode::TryWaitTaggedDynamicBufferId ||
        mode == Mode::TryWaitPollUntilDone ||
        mode == Mode::TryWaitFourSlots;
    const uint32_t dynamic_buffer_count =
        mode == Mode::TryWaitFourSlots ? 4U :
        (mode == Mode::TryWaitTaggedDynamicBufferId ? 1U : 0U);
    uint64_t completion = 0U;
    uint64_t completion_before_wait = 0U;
    int64_t try_wait_before_issue = 0;
    int64_t try_wait_after_issue = 0;
    int64_t try_wait_after_schedule = 0;
    int64_t try_wait_after_final_wait = 0;
    uint64_t try_wait_positive_poll_count = 0U;
    int64_t try_wait_poll_final_value = 0;

    uint64_t selector_status = 0U;
    if (register_base != 0U) {
        selector_status =
            atomic_probe::scalar_pmu::ReadSelectorStatus(register_base);
        atomic_probe::scalar_pmu::ClearCounters(register_base);
    }
    const uint64_t sys_begin = static_cast<uint64_t>(get_sys_cnt());
    bisheng::cce::metrics_prof_start();

    // 把第一份 save 也纳入窗口，ContextFifo(iterations=0) 才能给出完整的
    // “两次保存 + 两次恢复 + 协议校验”基础成本，而不是只测后半段。
    volatile ContinuationFrame frames[2]{};
    RuntimeContext first = MakeContext(control.seed, 0x100000001ULL);
    const uint64_t context0_signature = ContextSignature(first);
    SaveContinuation(&frames[0], first, /*generation=*/1U);
    // 主动覆盖原活跃变量，确保后续校验依赖 frame 恢复，而不是编译器
    // 恰好保留的旧 SSA/寄存器值。
    first = MakeContext(control.seed ^ 0xffffffffU, 0xdeadbeefULL);
    asm volatile("" : "+l"(first.words[0]) : : "memory");

    if (idle_cost_mode) {
        try_wait_poll_final_value = MeasureIdleTryWait(
            control.try_wait_buffer_id, control.schedule_iterations
        );
    }

    if (use_engine) {
        if (sample_try_wait) {
            try_wait_before_issue = SampleTryWait(
                mode, control.try_wait_buffer_id
            );
        }
#if defined(SCALAR_COROUTINE_BUILD_AIV)
        IssueEngine(
            input_a_global, input_b_global, output_global,
            input_a_tile, input_b_tile, output_tile, tag_final_store,
            dynamic_buffer_count, control.try_wait_buffer_id
        );
#else
        IssueEngine(
            input_a_global, input_b_global, output_global,
            input_a_mat, input_b_mat, input_a_l0, input_b_l0,
            output_l0, tag_final_store, dynamic_buffer_count,
            control.try_wait_buffer_id
        );
#endif
        if (sample_try_wait) {
            try_wait_after_issue = SampleTryWait(
                mode, control.try_wait_buffer_id
            );
        }
        if (mode == Mode::TryWaitPollUntilDone) {
            // try_wait 本身不等待。这里按返回值循环，验证返回值可直接作为
            // continuation 是否可恢复的分支条件，并限制上界避免协议错误时
            // 无限占用设备。
            constexpr uint64_t kMaxPolls = 1000000U;
            while (try_wait_positive_poll_count < kMaxPolls) {
                try_wait_poll_final_value = SampleTryWait(
                    mode, control.try_wait_buffer_id
                );
                if (try_wait_poll_final_value == 0) {
                    break;
                }
                ++try_wait_positive_poll_count;
            }
        }
        if (serial) {
            completion_before_wait = completion;
            WaitEngine();
            completion = 1U;
        }
    }

    const uint32_t replay_iterations =
        idle_cost_mode ? 0U : control.schedule_iterations;
    const uint64_t schedule_checksum = ReplayLoserSubmitsUntilSecondWinner(
        reinterpret_cast<volatile __gm__ const scalar_coroutine_probe::TaskRecord *>(
            control.task_records
        ),
        replay_iterations, control.seed, &frames[1]
    );
    if (sample_try_wait) {
        try_wait_after_schedule = SampleTryWait(
            mode, control.try_wait_buffer_id
        );
    }

    // replay 遇到第二个 winner 后，先恢复较早的 engine continuation。
    // overlap 模式由这个 continuation 执行最终 wait/commit；完成后才恢复
    // 第二个 winner。顺序是 task-age FIFO，而不是继续堆叠活跃局部变量。
    uint64_t resume_sequence = 0U;
    bool first_state_ok = false;
    RuntimeContext restored_first = RestoreContinuation(
        &frames[0], /*expected_generation=*/1U, first_state_ok
    );
    resume_sequence = (resume_sequence << 2U) | 1U;
    if (use_engine && !serial) {
        completion_before_wait = completion;
        WaitEngine();
        completion = 1U;
        if (sample_try_wait) {
            try_wait_after_final_wait = SampleTryWait(
                mode, control.try_wait_buffer_id
            );
        }
    }
    bool second_state_ok = false;
    RuntimeContext restored_second = RestoreContinuation(
        &frames[1], /*expected_generation=*/2U, second_state_ok
    );
    resume_sequence = (resume_sequence << 2U) | 2U;
    const uint64_t restored_context0_signature =
        ContextSignature(restored_first);
    const uint64_t restored_context1_signature =
        ContextSignature(restored_second);
    const uint64_t context1_signature = frames[1].signature;

    uint64_t protocol_status = 0U;
    protocol_status |= first_state_ok &&
        restored_context0_signature == context0_signature
        ? scalar_coroutine_probe::kProtocolContext0Restored : 0U;
    protocol_status |= second_state_ok &&
        restored_context1_signature == context1_signature
        ? scalar_coroutine_probe::kProtocolContext1Restored : 0U;
    protocol_status |= resume_sequence == 6U &&
        frames[0].generation == 1U &&
        frames[1].generation == 2U && frames[0].state ==
            static_cast<uint32_t>(
                scalar_coroutine_probe::ContinuationState::Resumed
            ) && frames[1].state == static_cast<uint32_t>(
                scalar_coroutine_probe::ContinuationState::Resumed
            )
        ? scalar_coroutine_probe::kProtocolFifoOrder : 0U;
    protocol_status |= (!use_engine ||
        (completion_before_wait == 0U && completion == 1U))
        ? scalar_coroutine_probe::kProtocolCompletionHeldUntilWait : 0U;

    bisheng::cce::metrics_prof_stop();
    const uint64_t sys_end = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t ctrl_after_stop = static_cast<uint64_t>(get_ctrl());
    atomic_probe::scalar_pmu::Snapshot snapshot{};
    if (register_base != 0U) {
        snapshot = atomic_probe::scalar_pmu::ReadSnapshot(register_base);
    }

    __gm__ scalar_coroutine_probe::ProbeResult *result = &state->result;
    Publish64(&result->sys_ticks, sys_end - sys_begin);
    Publish64(&result->pmu_total_cycles, snapshot.total);
    Publish64(&result->pmu_vector_busy, snapshot.vector_busy);
    Publish64(&result->pmu_cube_busy, snapshot.cube_busy);
    Publish64(&result->pmu_scalar_busy, snapshot.scalar_busy);
    Publish64(&result->pmu_mte1_busy, snapshot.mte1_busy);
    Publish64(&result->pmu_mte2_busy, snapshot.mte2_busy);
    Publish64(&result->pmu_mte3_busy, snapshot.mte3_busy);
    Publish64(&result->pmu_fix_busy, snapshot.fix_busy);
    Publish64(&result->pmu_icache_request, snapshot.icache_request);
    Publish64(&result->pmu_icache_miss, snapshot.icache_miss);
    Publish64(&result->physical_core_id, physical_core_id);
    Publish64(&result->selector_status, selector_status);
    Publish64(&result->context0_signature, context0_signature);
    Publish64(&result->context1_signature, context1_signature);
    Publish64(&result->schedule_checksum, schedule_checksum);
    Publish64(
        &result->restored_context0_signature,
        restored_context0_signature
    );
    Publish64(
        &result->restored_context1_signature,
        restored_context1_signature
    );
    Publish64(&result->completion_before_wait, completion_before_wait);
    Publish64(&result->completion_after_wait, completion);
    Publish64(&result->observed_mode, control.mode);
    Publish64(
        &result->observed_iterations, control.schedule_iterations
    );
#if defined(SCALAR_COROUTINE_BUILD_AIC)
    Publish64(
        &result->observed_role,
        static_cast<uint64_t>(scalar_coroutine_probe::Role::Aic)
    );
#else
    Publish64(
        &result->observed_role,
        static_cast<uint64_t>(scalar_coroutine_probe::Role::Aiv)
    );
#endif
    Publish64(&result->protocol_status, protocol_status);
    Publish64(&result->pmu_ctrl_after_stop, ctrl_after_stop);
    Publish64(&result->resume_sequence, resume_sequence);
    Publish64(
        &result->try_wait_before_issue,
        static_cast<uint64_t>(try_wait_before_issue)
    );
    Publish64(
        &result->try_wait_after_issue,
        static_cast<uint64_t>(try_wait_after_issue)
    );
    Publish64(
        &result->try_wait_after_schedule,
        static_cast<uint64_t>(try_wait_after_schedule)
    );
    Publish64(
        &result->try_wait_after_final_wait,
        static_cast<uint64_t>(try_wait_after_final_wait)
    );
    Publish64(
        &result->try_wait_positive_poll_count,
        try_wait_positive_poll_count
    );
    Publish64(
        &result->try_wait_poll_final_value,
        static_cast<uint64_t>(try_wait_poll_final_value)
    );
    dsb(DSB_ALL);
}
