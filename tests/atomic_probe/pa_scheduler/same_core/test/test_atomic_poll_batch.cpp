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

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// 该自测在普通 CPU 编译器上直接实例化 device 公共模板：只消去地址空间
// 修饰符，不复制 PollBatch 实现，避免测试与被测代码各维护一套逻辑。
#define PA_DEVICE inline
#define PA_GM
#include "pa_trace.h"

namespace {

using pa_scheduler::AtomicOp;
using pa_scheduler::AtomicPollBatchEnabled;
using pa_scheduler::AtomicPollBoundaryAt;
using pa_scheduler::AtomicPollRegionBegin;
using pa_scheduler::AtomicPollRegionEnd;
using pa_scheduler::AtomicSite;
using pa_scheduler::AccumulateAtomicPollCall;
using pa_scheduler::CaptureAtomicCompareExchange;
using pa_scheduler::TraceAtomicLoad;
using pa_scheduler::TraceAtomicPollBatchMask;
using pa_scheduler::TraceAtomicPollBatchIndex;
using pa_scheduler::TraceContext;
using pa_scheduler::TraceCoreState;
using pa_scheduler::TracePhase;
using pa_scheduler::TraceRecord;
using pa_scheduler::WorkerResult;
using pa_scheduler::WriteAggregateAtomicPollBatch;
using pa_scheduler::WriteAtomicTrace;
using pa_scheduler::kAtomicOpMask;
using pa_scheduler::kAtomicPollBatch;
using pa_scheduler::kAtomicPollCountMax;
using pa_scheduler::kAtomicPollCountShift;
using pa_scheduler::kAtomicResultUsed;
using pa_scheduler::kAtomicReturnReady;

int g_failures = 0;

void Expect(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "[FAIL] atomic PollBatch: %s\n", message);
    ++g_failures;
}

// 可控时钟让边界断言不依赖 host 调度；Load 只为验证 trace 包装器的分流，
// 不尝试在这个单线程单元测试中模拟 A5 atomicAdd(0) 的硬件时延。
struct TestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static uint64_t now;

    static uint64_t Now() { return now++; }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value) {
        (void)value;
        return Now();
    }

    template <typename T>
    static T Load(volatile T *address) {
        return *address;
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        const int64_t observed = *address;
        if (observed == expected) {
            *address = desired;
        }
        return observed;
    }
};

uint64_t TestOps::now = 0;

struct ReturnReadyTestOps : TestOps {
    static constexpr bool kAtomicReturnReadyObserved = true;
};

struct Fixture {
    TraceCoreState core{};
    TraceRecord records[8]{};
    WorkerResult result{};
    TraceContext trace{};

    Fixture() {
        trace.core = &core;
        trace.records = records;
        trace.capacity =
            static_cast<uint32_t>(sizeof(records) / sizeof(records[0]));
        trace.atomics_enabled = true;
    }
};

void TestSplitAtMaximumCount() {
    Fixture fixture;
    constexpr AtomicSite kSite = AtomicSite::StartupPoll;
    constexpr uint32_t kBatchIndex = 0;
    constexpr uint32_t kBatchBit = 1U << kBatchIndex;
    constexpr uint64_t kFirstStart = 111;
    constexpr uint64_t kSecondStart = 222;
    constexpr uint64_t kSecondEnd = 333;

    Expect(
        TraceAtomicPollBatchIndex(kSite) == static_cast<int32_t>(kBatchIndex),
        "StartupPoll 的 batch index 与稳定映射不一致"
    );
    fixture.trace.poll_burst.active_mask = kBatchBit;
    fixture.trace.poll_burst.start_cycle[kBatchIndex] = kFirstStart;
    fixture.trace.poll_burst.call_count[kBatchIndex] = kAtomicPollCountMax - 1;
    TestOps::now = 200;

    // 第 0xFFFFFF 次调用属于第一条记录，并在达到 24-bit 上限时立即落盘。
    AccumulateAtomicPollCall<TestOps>(fixture.trace, fixture.result, kSite, kFirstStart);

    Expect(fixture.trace.record_count == 1U, "达到最大计数时应立即写出第一条记录");
    Expect(fixture.trace.poll_batch_records == 1U, "第一条 PollBatch 物理记录计数错误");
    Expect(fixture.trace.poll_burst.active_mask == 0U, "达到上限后 active mask 应清零");
    Expect(
        fixture.trace.poll_burst.call_count[kBatchIndex] == 0U,
        "达到上限后站点调用计数应清零"
    );
    Expect(
        fixture.records[0].phase == static_cast<int32_t>(TracePhase::Atomic),
        "第一条记录 phase 不是 Atomic"
    );
    Expect(
        fixture.records[0].auxiliary == static_cast<uint32_t>(kSite),
        "第一条记录 site 不正确"
    );
    Expect(fixture.records[0].start_cycle == kFirstStart, "第一条记录起始时钟不正确");
    Expect(
        fixture.records[0].end_cycle >= fixture.records[0].start_cycle,
        "第一条记录的结束时钟早于起始时钟"
    );
    Expect(
        (fixture.records[0].flags & kAtomicPollBatch) != 0U,
        "第一条记录缺少 PollBatch 标志"
    );
    Expect(
        fixture.records[0].flags >> kAtomicPollCountShift == kAtomicPollCountMax,
        "第一条记录没有编码最大 24-bit 调用数"
    );

    // 第 0x1000000 次调用必须重新开启 count=1 的新 batch，不能饱和或丢失。
    AccumulateAtomicPollCall<TestOps>(fixture.trace, fixture.result, kSite, kSecondStart);
    Expect(fixture.trace.poll_burst.active_mask == kBatchBit, "上限后的下一次调用没有重开 batch");
    Expect(
        fixture.trace.poll_burst.call_count[kBatchIndex] == 1U,
        "重开 batch 的初始调用数不是 1"
    );
    AtomicPollBoundaryAt<TestOps>(fixture.trace, kSecondEnd);

    Expect(fixture.trace.record_count == 2U, "max+1 次调用应写出两条记录");
    Expect(fixture.trace.poll_batch_records == 2U, "max+1 次调用的物理 batch 数错误");
    Expect(fixture.trace.poll_burst.active_mask == 0U, "第二条记录关闭后 active mask 未清零");
    Expect(
        fixture.trace.poll_burst.call_count[kBatchIndex] == 0U,
        "第二条记录关闭后站点调用计数未清零"
    );
    Expect(fixture.records[1].start_cycle == kSecondStart, "第二条记录起始时钟不正确");
    Expect(fixture.records[1].end_cycle == kSecondEnd, "第二条记录结束时钟不正确");
    Expect(
        (fixture.records[1].flags & kAtomicPollBatch) != 0U,
        "第二条记录缺少 PollBatch 标志"
    );
    Expect(
        fixture.records[1].flags >> kAtomicPollCountShift == 1U,
        "第二条记录没有精确编码一次调用"
    );
    const uint64_t represented_calls =
        static_cast<uint64_t>(fixture.records[0].flags >> kAtomicPollCountShift) +
        static_cast<uint64_t>(fixture.records[1].flags >> kAtomicPollCountShift);
    Expect(
        represented_calls == static_cast<uint64_t>(kAtomicPollCountMax) + 1U,
        "两条记录的加权调用数没有闭合到 max+1"
    );
    Expect(fixture.trace.dropped_records == 0U, "边界拆批不应丢记录");
    Expect(!fixture.trace.atomic_counter_overflow, "边界拆批不应报告计数溢出");
}

void TestNestedRegionRestoresMask() {
    Fixture fixture;
    volatile int64_t startup_value = 96;
    volatile int64_t fanin_value = 1;
    const uint32_t startup_mask = TraceAtomicPollBatchMask(AtomicSite::StartupPoll);
    const uint32_t fanin_mask = TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad);
    TestOps::now = 1000;

    const uint32_t outer_previous =
        AtomicPollRegionBegin<TestOps>(fixture.trace, fixture.result, startup_mask);
    Expect(outer_previous == 0U, "最外层 region 的 previous mask 应为零");
    Expect(fixture.trace.poll_burst.enabled_mask == startup_mask, "最外层 region 未启用 startup site");
    Expect(
        AtomicPollBatchEnabled(fixture.trace, AtomicSite::StartupPoll, AtomicOp::Load),
        "最外层 startup site 应允许聚合"
    );
    (void)TraceAtomicLoad<TestOps>(
        fixture.trace, fixture.result, -1, AtomicSite::StartupPoll, &startup_value
    );

    // 嵌套 begin 会先关闭外层已有 batch，再把 inner mask 与外层 mask 合并。
    const uint32_t inner_previous =
        AtomicPollRegionBegin<TestOps>(fixture.trace, fixture.result, fanin_mask);
    Expect(inner_previous == startup_mask, "内层 region 没有保存外层 mask");
    Expect(
        fixture.trace.poll_burst.enabled_mask == (startup_mask | fanin_mask),
        "内层 region 没有合并两层 mask"
    );
    Expect(fixture.trace.record_count == 1U, "内层 begin 没有关闭外层 active batch");
    (void)TraceAtomicLoad<TestOps>(
        fixture.trace, fixture.result, -1, AtomicSite::FaninFlagLoad, &fanin_value
    );

    AtomicPollRegionEnd<TestOps>(fixture.trace, fixture.result, inner_previous);
    Expect(fixture.trace.poll_burst.enabled_mask == startup_mask, "内层 end 没有还原外层 mask");
    Expect(fixture.trace.record_count == 2U, "内层 end 没有关闭内层 active batch");
    (void)TraceAtomicLoad<TestOps>(
        fixture.trace, fixture.result, -1, AtomicSite::StartupPoll, &startup_value
    );

    AtomicPollRegionEnd<TestOps>(fixture.trace, fixture.result, outer_previous);
    Expect(fixture.trace.poll_burst.enabled_mask == 0U, "最外层 end 没有还原初始 mask");
    Expect(fixture.trace.poll_burst.active_mask == 0U, "嵌套 region 结束后仍有 active batch");
    Expect(fixture.trace.record_count == 3U, "嵌套 region 应按三个边界写出三条 batch");
    Expect(fixture.trace.poll_batch_records == 3U, "嵌套 region 的物理 batch 计数错误");
    Expect(fixture.trace.poll_calls == 3U, "嵌套 region 的逻辑 poll 调用数错误");
    Expect(fixture.result.atomic_trace_calls == 3U, "嵌套 region 的逻辑 atomic 调用数错误");
    Expect(
        fixture.records[0].auxiliary == static_cast<uint32_t>(AtomicSite::StartupPoll) &&
            fixture.records[1].auxiliary == static_cast<uint32_t>(AtomicSite::FaninFlagLoad) &&
            fixture.records[2].auxiliary == static_cast<uint32_t>(AtomicSite::StartupPoll),
        "嵌套 region 的 batch site 顺序错误"
    );
    Expect(fixture.trace.dropped_records == 0U, "嵌套 region 不应丢记录");
    Expect(!fixture.trace.atomic_counter_overflow, "嵌套 region 不应报告计数溢出");
}

void TestNonAllowlistedSiteStaysDirect() {
    Fixture fixture;
    volatile int64_t frontier_flag = 7;
    constexpr AtomicSite kSite = AtomicSite::FrontierFlagLoad;
    const uint32_t site_mask = TraceAtomicPollBatchMask(kSite);
    TestOps::now = 2000;

    Expect(site_mask == 0U, "非 PollBatch allowlist 的 site 必须得到空 compact mask");
    const uint32_t previous =
        AtomicPollRegionBegin<TestOps>(fixture.trace, fixture.result, site_mask);
    Expect(
        !AtomicPollBatchEnabled(fixture.trace, kSite, AtomicOp::Load),
        "非 allowlist 的 frontier scan 不得因 region mask 被聚合"
    );
    const int64_t observed = TraceAtomicLoad<TestOps>(
        fixture.trace, fixture.result, 37, kSite, &frontier_flag
    );
    AtomicPollRegionEnd<TestOps>(fixture.trace, fixture.result, previous);

    Expect(observed == frontier_flag, "非 allowlist direct load 返回值错误");
    Expect(fixture.trace.record_count == 1U, "非 allowlist load 应写一条 direct 记录");
    Expect(fixture.trace.poll_calls == 0U, "非 allowlist load 不得增加 batched poll 调用数");
    Expect(fixture.trace.poll_batch_records == 0U, "非 allowlist load 不得写 PollBatch 记录");
    Expect(fixture.result.atomic_trace_calls == 1U, "非 allowlist direct load 的逻辑调用计数错误");
    Expect(
        (fixture.records[0].flags & kAtomicPollBatch) == 0U,
        "非 allowlist load 被错误标成 PollBatch"
    );
    Expect(fixture.records[0].task_id == 37, "非 allowlist direct load 丢失 task 归因");
    Expect(
        fixture.records[0].auxiliary == static_cast<uint32_t>(kSite),
        "非 allowlist direct load 的 site 错误"
    );
    Expect(fixture.trace.poll_burst.active_mask == 0U, "非 allowlist load 不应留下 active batch");
    Expect(fixture.trace.dropped_records == 0U, "非 allowlist direct load 不应丢记录");
    Expect(!fixture.trace.atomic_counter_overflow, "非 allowlist direct load 不应报告溢出");
}

void TestRawSiteIdNeverBecomesTheEnableMaskBit() {
    constexpr AtomicSite kHighRawSite = static_cast<AtomicSite>(40);
    Expect(
        TraceAtomicPollBatchIndex(kHighRawSite) == -1,
        "未登记的高编号 site 不应获得 compact PollBatch index"
    );
    Expect(
        TraceAtomicPollBatchMask(kHighRawSite) == 0U,
        "高编号 raw site 不得参与 32-bit 移位构造 enable mask"
    );
    Expect(
        TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad) == (1U << 2),
        "Fanin 的 enable bit 必须来自 compact index 2，而不是 raw site 5"
    );
    Expect(
        TraceAtomicPollBatchIndex(
            AtomicSite::SharedInsertTurnPoll
        ) == -1 &&
            TraceAtomicPollBatchMask(
                AtomicSite::SharedInsertTurnPoll
            ) == 0,
        "aggregate-only insert-turn poll 不得扩张热循环 compact state"
    );
}

void TestAggregateInsertTurnPollBatch() {
    Fixture fixture;
    constexpr uint64_t kBegin = 4000;
    constexpr uint64_t kEnd = 4500;
    constexpr uint64_t kCalls = 37;
    const bool written = WriteAggregateAtomicPollBatch(
        fixture.trace, fixture.result,
        AtomicSite::SharedInsertTurnPoll,
        kBegin, kEnd, kCalls, true
    );
    Expect(written, "insert-turn aggregate PollBatch 应写入一条记录");
    Expect(fixture.trace.record_count == 1, "aggregate PollBatch 物理记录数不是 1");
    Expect(
        fixture.result.atomic_trace_calls == kCalls &&
            fixture.trace.poll_calls == kCalls,
        "aggregate PollBatch 没有一次性累计精确 logical calls"
    );
    Expect(
        fixture.trace.poll_batch_records == 1,
        "aggregate PollBatch 物理 batch 计数不是 1"
    );
    const TraceRecord &record = fixture.records[0];
    Expect(
        record.start_cycle == kBegin &&
            record.end_cycle == kEnd,
        "aggregate PollBatch 没有复用传入的 Register/Ready 边界"
    );
    Expect(
        record.auxiliary ==
            static_cast<uint32_t>(
                AtomicSite::SharedInsertTurnPoll
            ),
        "aggregate PollBatch site 不正确"
    );
    Expect(
        (record.flags & kAtomicOpMask) ==
                static_cast<uint32_t>(AtomicOp::Load) &&
            (record.flags & kAtomicResultUsed) != 0 &&
            (record.flags & kAtomicPollBatch) != 0 &&
            (record.flags & kAtomicReturnReady) != 0,
        "aggregate PollBatch 的 Load/result/poll/return-ready 标志不闭合"
    );
    Expect(
        record.flags >> kAtomicPollCountShift == kCalls,
        "aggregate PollBatch 未编码精确 logical call_count"
    );

    Fixture overflow;
    const bool overflow_written = WriteAggregateAtomicPollBatch(
        overflow.trace, overflow.result,
        AtomicSite::SharedInsertTurnPoll,
        kBegin, kEnd,
        static_cast<uint64_t>(kAtomicPollCountMax) + 1,
        true
    );
    Expect(
        !overflow_written && overflow.trace.record_count == 0 &&
            overflow.result.atomic_trace_calls == 0 &&
            overflow.trace.poll_calls == 0 &&
            overflow.trace.atomic_counter_overflow,
        "超过 24-bit 的聚合调用数必须 fail-closed，不能饱和或拆批"
    );
}

void TestInsertTurnHandoffCompareExchange() {
    Fixture fixture;
    volatile int64_t token = 7;
    uint64_t trace_begin = 0;
    uint64_t trace_end = 0;
    TestOps::now = 5000;
    const int64_t observed =
        CaptureAtomicCompareExchange<ReturnReadyTestOps>(
            fixture.trace, &token, 7, 8,
            trace_begin, trace_end
        );
    Expect(
        observed == 7 && token == 8,
        "handoff CompareExchange 返回值或目标 token 不正确"
    );
    Expect(
        fixture.trace.record_count == 0 &&
            fixture.result.atomic_trace_calls == 0,
        "CAS 捕获阶段不得提前写 raw 或更新 logical counter"
    );
    WriteAtomicTrace<ReturnReadyTestOps>(
        fixture.trace, fixture.result, 7,
        AtomicSite::SharedInsertTurnHandoff,
        AtomicOp::CompareExchange,
        trace_begin, trace_end, true, true
    );
    Expect(
        fixture.trace.record_count == 1 &&
            fixture.result.atomic_trace_calls == 1,
        "父/detail 端点固定后，handoff CAS 必须恰好写一条 direct atomic"
    );
    const TraceRecord &record = fixture.records[0];
    Expect(
        record.task_id == 7 &&
            record.auxiliary ==
                static_cast<uint32_t>(
                    AtomicSite::SharedInsertTurnHandoff
                ),
        "handoff CAS 没有保留 task/site 身份"
    );
    Expect(
        (record.flags & kAtomicOpMask) ==
                static_cast<uint32_t>(
                    AtomicOp::CompareExchange
                ) &&
            (record.flags & kAtomicResultUsed) != 0 &&
            (record.flags & kAtomicReturnReady) != 0 &&
            (record.flags & kAtomicPollBatch) == 0,
        "handoff CAS 的 op/result/return-ready/direct 标志不正确"
    );
    Expect(
        trace_end == record.end_cycle &&
            record.end_cycle >= record.start_cycle,
        "handoff CAS 记录没有使用捕获的返回依赖边界"
    );
}

}  // namespace

int main() {
    TestSplitAtMaximumCount();
    TestNestedRegionRestoresMask();
    TestNonAllowlistedSiteStaysDirect();
    TestRawSiteIdNeverBecomesTheEnableMaskBit();
    TestAggregateInsertTurnPollBatch();
    TestInsertTurnHandoffCompareExchange();
    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] atomic PollBatch self-test failures=%d\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("[PASS] atomic PollBatch split/region/allowlist self-test\n");
    return EXIT_SUCCESS;
}
