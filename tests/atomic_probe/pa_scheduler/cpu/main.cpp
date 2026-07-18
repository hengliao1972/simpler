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

#include "../common/host_support.h"

#define PA_DEVICE inline
#define PA_GM
// CPU 后端直接实例化与设备端相同的公共调度器；这里只消去 AICore 地址空间
// 修饰符，不另写一套简化状态机，因此它可以承担协议和边界回归。
#include "../common/pa_scheduler_core.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

template <uint32_t Count>
inline void EmitNops() {
    // 编译期 Count 配合强制展开，避免编译器把空循环折叠掉。
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 256
#endif
    for (uint32_t index = 0; index < Count; ++index) {
        asm volatile("nop");
    }
}

inline void RuntimeNop(uint32_t count) {
    // 与 AscendC 后端采用相同的 256 + 二进制尾块分解，保证入参含义一致。
    // x86 nop 的吞吐和 CPU 线程调度都不同于 A5，所以这里只复现指令数量，
    // 不能把 CPU 测得时间解释为 A5 kernel 时间。
    while (count >= 256) {
        EmitNops<256>();
        count -= 256;
    }
    if ((count & 128U) != 0) EmitNops<128>();
    if ((count & 64U) != 0) EmitNops<64>();
    if ((count & 32U) != 0) EmitNops<32>();
    if ((count & 16U) != 0) EmitNops<16>();
    if ((count & 8U) != 0) EmitNops<8>();
    if ((count & 4U) != 0) EmitNops<4>();
    if ((count & 2U) != 0) EmitNops<2>();
    if ((count & 1U) != 0) EmitNops<1>();
}

struct CpuOps {
    // CPU 原子 built-in 在函数返回前已产生旧值；该后端只做协议回归，不把
    // x86 时间分布外推到 A5。
    static constexpr bool kAtomicReturnReadyObserved = true;

    // 用 fetch_add(0) 模拟 A5 atomicAdd(addr, 0) 原子读，而不是退化为普通
    // CPU load。Acquire/AcqRel 只建立本 CPU 协议回归需要的发布/观察关系，
    // 不模拟 A5 cache 或设备内存模型细节。
    static inline int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(address, static_cast<int32_t>(0), __ATOMIC_ACQUIRE);
    }

    static inline int64_t Load(volatile int64_t *address) {
        // 保留原子 add-zero 路径，让 96 个 pthread 仍在同一批热点地址上竞争。
        return __atomic_fetch_add(address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE);
    }

    static inline uint64_t Load(volatile uint64_t *address) {
        return __atomic_fetch_add(address, static_cast<uint64_t>(0), __ATOMIC_ACQUIRE);
    }

    static inline int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static inline int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static inline uint64_t Exchange(volatile uint64_t *address, uint64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static inline int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static inline int64_t FetchMax(volatile int64_t *address, int64_t value, uint64_t &retries) {
        // CPU 没有直接对应本测试签名的 fetch-max，用 CAS loop 实现同一返回值
        // 语义；retries 仅用于诊断软件竞争，不能与 A5 硬件 AtomicMax 对比。
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (value > current) {
            if (__atomic_compare_exchange_n(address, &current, value, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                break;
            }
            ++retries;
        }
        return current;
    }

    // 前后的 Exchange 已使用 AcqRel，Publish 使用 Release，因此这里不重复
    // 插入 fence，保持与设备适配层相同的调用边界。
    static inline void StoreBarrier() {}

    // 将 steady_clock 统一换算成纳秒，数值上适配公共模型的 1 GHz tick 标度；
    // 这不表示 CPU 物理时钟为 1 GHz，也不保证实际分辨率达到 1 ns。
    static inline uint64_t Now() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count()
        );
    }

    template <typename T>
    static inline uint64_t NowAfterAtomicResult(T value) {
        // 空 asm 让编译器保留返回值到计时点的数据依赖，不额外插入 CPU fence。
        asm volatile("" : "+r"(value));
        return Now();
    }

    static inline void Nop(uint32_t count) { RuntimeNop(count); }

    static inline void SpinHint() {}

    static inline void InvalidateRegion(const void *, uint64_t) {
        // CPU 没有 A5 DCache line 失效指令；这里仅提供保守的本线程顺序边界，
        // 接口占位但不模拟设备 cache line 行为。共享状态本身仍使用 atomic。
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static inline void FlushRegion(void *, uint64_t) { std::atomic_thread_fence(std::memory_order_seq_cst); }

    static inline void Publish(uint64_t *address, uint64_t value) {
        // Release store 对应设备端 bypass-DCache 结果发布的可见性边界。
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }
};

}  // namespace

// CPU runner 不承担 A5 性能对比，只负责用同一份 SchedulerState 和公共调度器
// 做协议回归。生命周期为参数解析、host 内存准备、逐轮 96 线程执行、严格校验、
// 可选泳道后处理，最后统一释放 trace buffer。
int main(int argc, char **argv) {
    pa_scheduler::host::Options options;
    const pa_scheduler::host::ParseStatus parse_status = pa_scheduler::host::ParseOptions(argc, argv, false, &options);
    if (parse_status != pa_scheduler::host::ParseStatus::Ok) {
        return parse_status == pa_scheduler::host::ParseStatus::Help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    pa_scheduler::host::PrintBanner("CPU", options);
    std::printf("[NOTE] CPU NOP counts preserve instruction count, not A5 microseconds.\n");

    // SchedulerState 很大，放到 heap 而不是主线程栈；trace 继续保持独立的
    // 64 字节对齐区域，以复用设备端完全相同的二进制布局。
    std::unique_ptr<pa_scheduler::SchedulerState> state(new pa_scheduler::SchedulerState);
    void *trace_memory = nullptr;
    if (options.trace_enabled) {
        trace_memory = std::aligned_alloc(64, pa_scheduler::kTraceBytes);
        if (trace_memory == nullptr) {
            std::fprintf(stderr, "Cannot allocate %zu-byte swimlane trace buffer.\n", pa_scheduler::kTraceBytes);
            return EXIT_FAILURE;
        }
    }
    auto *trace_header = static_cast<pa_scheduler::TraceHeader *>(trace_memory);
    std::vector<double> spans;
    bool all_passed = true;
    bool postprocess_ok = true;
    // 每轮复用大块 host 分配，只重置公共状态和 trace header。与设备后端一样，
    // runs>1 表示同进程热运行，不等价于多个独立首轮。
    for (uint32_t run = 1; run <= options.runs; ++run) {
        pa_scheduler::host::InitializeState(state.get(), options);
        pa_scheduler::host::ConfigureTrace(state.get(), options, trace_memory);
        if (options.trace_enabled) {
            pa_scheduler::host::InitializeTraceHeader(trace_header);
        }
        const auto wall_begin = std::chrono::steady_clock::now();
        // 固定创建 96 个参与者：worker 0..31 扮演 AIC，32..95 扮演 AIV。
        // 每个线程仍会进入公共 started_count 屏障，再共同回放完整 task 流。
        std::vector<std::thread> workers;
        workers.reserve(pa_scheduler::kWorkers);
        for (uint32_t worker_id = 0; worker_id < pa_scheduler::kWorkers; ++worker_id) {
            const pa_scheduler::CoreRole role =
                worker_id < pa_scheduler::kAicWorkers ? pa_scheduler::CoreRole::Aic : pa_scheduler::CoreRole::Aiv;
            workers.emplace_back([state_pointer = state.get(), worker_id, role]() {
                pa_scheduler::RunScheduler<CpuOps>(state_pointer, worker_id, role);
            });
        }
        // join 是本后端的 kernel 完成屏障；所有 worker 退出后才能读取最终状态，
        // 对应设备 runner 的 aclrtSynchronizeStream。
        for (std::thread &worker : workers)
            worker.join();
        const auto wall_end = std::chrono::steady_clock::now();
        const double host_us = std::chrono::duration<double, std::micro>(wall_end - wall_begin).count();
        // host 内存沿用与 A5 相同的 TraceHeader + 每 worker 固定跨度 ABI；
        // 分析器和 raw JSON writer 因而可以与设备后端共用同一回调接口。
        const auto read_trace_records =
            [trace_memory](uint32_t worker, uint32_t count, pa_scheduler::TraceRecord *records) {
                const uint64_t offset = sizeof(pa_scheduler::TraceHeader) +
                                        static_cast<uint64_t>(worker) * pa_scheduler::kTraceRecordsPerCore *
                                            sizeof(pa_scheduler::TraceRecord);
                std::memcpy(
                    records, static_cast<uint8_t *>(trace_memory) + offset,
                    static_cast<size_t>(count) * sizeof(pa_scheduler::TraceRecord)
                );
                return true;
            };
        // 先完成严格语义校验，再允许写出；失败运行不会生成可误认成有效
        // 基线的泳道 JSON。
        const pa_scheduler::host::Metrics metrics = pa_scheduler::host::Validate(
            *state, run, host_us, options.trace_enabled ? trace_header : nullptr
        );
        all_passed &= metrics.passed;
        spans.push_back(metrics.submit_span_us);
        // 分析只打印统计，导出则写 raw JSON；两者失败都标记 postprocess，
        // 与调度语义失败分开报告，便于区分协议问题和产物问题。
        if (options.analyze_swimlane &&
            !pa_scheduler::host::AnalyzeSwimlaneRecords(*trace_header, *state, read_trace_records)) {
            postprocess_ok = false;
            break;
        }
        if (!options.swimlane_json.empty()) {
            if (!metrics.passed) {
                std::fprintf(stderr, "Skipping swimlane export because semantic validation failed.\n");
                postprocess_ok = false;
                break;
            }
            if (!pa_scheduler::host::ExportSwimlaneRecords(
                    *trace_header, options.swimlane_json, read_trace_records
                )) {
                postprocess_ok = false;
                break;
            }
        }
    }

    std::printf(
        "[SUMMARY] runs=%u median_submit_span_us=%.3f semantic_status=%s postprocess_status=%s\n", options.runs,
        pa_scheduler::host::Median(spans), all_passed ? "PASS" : "FAIL", postprocess_ok ? "PASS" : "FAIL"
    );
    // std::free(nullptr) 合法，因此关闭泳道时也走同一条收尾路径。
    std::free(trace_memory);
    return all_passed && postprocess_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
