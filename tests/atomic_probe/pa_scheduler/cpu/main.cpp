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
#include "../common/winner_workload_host.h"

#define PA_DEVICE inline
#define PA_DEVICE_NOINLINE static __attribute__((noinline))
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

#if PA_BUILD_PERF_CLOCK
thread_local uint32_t g_perf_clock_read_count = 0;
#endif

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

// CPU 后端用相同的 128x128 float 输入、输出布局执行真实算术，以便在不依赖
// CANN 的环境中回归任务分派和输出闭环。这里的普通 CPU 浮点循环只与数学
// 结果对等，不冒充 A5 Cube/Vector 指令、流水线或 PMU 数据。
__attribute__((noinline)) void RunRealMatrixWorkload(
    const float *input_a, const float *input_b, float *output, uint32_t repeats
) {
    using namespace pa_scheduler::winner_workload;
    for (uint32_t iteration = 0; iteration < repeats; ++iteration) {
        for (uint32_t row = 0; row < kTileRows; ++row) {
            for (uint32_t column = 0; column < kTileCols; ++column) {
                float accumulator = 0.0F;
                for (uint32_t inner = 0; inner < kTileCols; ++inner) {
                    accumulator += input_a[static_cast<size_t>(row) * kTileCols + inner] *
                                   input_b[static_cast<size_t>(inner) * kTileCols + column];
                }
                output[static_cast<size_t>(row) * kTileCols + column] = accumulator;
            }
        }
        // 每轮都必须把完整结果物化到输出 tile，不能让 O3 因后续轮次覆盖同一
        // tile 而删除前一轮算术；这是编译器边界，不增加硬件 PMU 或 A5 屏障语义。
        asm volatile("" : : "r"(output) : "memory");
    }
}

template <bool Multiply>
__attribute__((noinline)) void RunRealVectorWorkload(
    const float *input_a, const float *input_b, float *output, uint32_t repeats
) {
    using namespace pa_scheduler::winner_workload;
    for (uint32_t iteration = 0; iteration < repeats; ++iteration) {
        for (size_t element = 0; element < kTileElements; ++element) {
            output[element] = Multiply ? input_a[element] * input_b[element]
                                       : input_a[element] + input_b[element];
        }
        // 与矩阵路径相同，明确保留每一次完整 elementwise 迭代。
        asm volatile("" : : "r"(output) : "memory");
    }
}

__attribute__((noinline)) void ExecuteRealWinnerWorkload(
    pa_scheduler::SchedulerState *state, pa_scheduler::WorkerState &worker,
    pa_scheduler::TaskKind kind
) {
    using namespace pa_scheduler;
    using namespace pa_scheduler::winner_workload;
    const WinnerWorkloadConfig &config = state->winner_workload;
    const uint32_t repeats = WorkloadCountForKind(config.repeats, kind);
    const bool role_matches =
        (worker.role == CoreRole::Aic && (kind == TaskKind::Qk || kind == TaskKind::Pv)) ||
        (worker.role == CoreRole::Aiv && (kind == TaskKind::Sf || kind == TaskKind::Up));
    // 与设备实现采用同一组版本、范围和角色门禁；配置错误不解引用 workspace，
    // host 的 active tile 数值/sentinel 校验会把本轮判为失败。
    if (config.version != kWinnerWorkloadConfigVersion || config.workspace_base == 0 ||
        config.workspace_bytes < kWorkspaceBytes || worker.core_idx < 0 ||
        static_cast<uint32_t>(worker.core_idx) >= kWorkers || repeats == 0 ||
        repeats > kMaxRealComputeCount || !role_matches) {
        return;
    }

    float *workspace = reinterpret_cast<float *>(static_cast<uintptr_t>(config.workspace_base));
    const uint32_t kind_slot = (kind == TaskKind::Pv || kind == TaskKind::Up) ? 1U : 0U;
    const size_t output_tile =
        kSharedInputTiles + static_cast<size_t>(worker.core_idx) * kOutputTilesPerWorker + kind_slot;
    float *output = workspace + output_tile * kTileElements;
    const float *input_a = workspace;
    const float *input_b = workspace + kTileElements;
    if (worker.role == CoreRole::Aic) {
        RunRealMatrixWorkload(input_a, input_b, output, repeats);
    } else if (kind == TaskKind::Sf) {
        RunRealVectorWorkload<false>(input_a, input_b, output, repeats);
    } else {
        RunRealVectorWorkload<true>(input_a, input_b, output, repeats);
    }
}

struct CpuOps {
    // CPU 后端只验证调度协议与 raw schema，没有建立与 A5 CCEC
    // 同构的“atomic 返回值依赖 + SYS_CNT”硬件边界；因此必须标记为
    // source_issue，不能让 x86 built-in 的函数返回冒充 A5 return_ready。
    static constexpr bool kAtomicReturnReadyObserved = false;

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

    static inline int64_t CompareExchange(
        volatile int64_t *address, int64_t expected, int64_t desired
    ) {
        // 与 production atomic wrapper 保持一致：返回线性化点观察到的
        // 旧值，而不是 bool。失败时目标字保持原样，调用方据此保留现场。
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
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

#if PA_BUILD_PERF_CLOCK
    static inline void ResetPerfClockReadCount() { g_perf_clock_read_count = 0; }

    static inline uint32_t PerfClockReadCount() { return g_perf_clock_read_count; }

    static inline uint64_t PerfClockNow() {
        ++g_perf_clock_read_count;
        return Now();
    }
#endif

    template <typename T>
    static inline uint64_t NowAfterAtomicResult(T value) {
        // 空 asm 让编译器保留返回值到计时点的数据依赖，不额外插入 CPU fence。
        asm volatile("" : "+r"(value));
        return Now();
    }

    static inline void ExecuteKernel(
        pa_scheduler::SchedulerState *state, pa_scheduler::WorkerState &worker, pa_scheduler::TaskKind kind,
        uint32_t nop_count
    ) {
        if (state->winner_workload.mode ==
            static_cast<uint32_t>(pa_scheduler::WinnerWorkloadMode::RealCompute)) {
            ExecuteRealWinnerWorkload(state, worker, kind);
            return;
        }
        RuntimeNop(nop_count);
    }

    static inline bool PmuWindowStart(pa_scheduler::SchedulerState *, uint32_t) { return false; }

    static inline void PmuWindowStop(pa_scheduler::SchedulerState *, uint32_t, bool) {}

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
    pa_scheduler::host::WinnerWorkloadOptions workload_options;
    std::vector<char *> common_argv;
    if (!pa_scheduler::host::ParseWinnerWorkloadOptions(
            argc, argv, &workload_options, &common_argv
        )) {
        return EXIT_FAILURE;
    }
    const pa_scheduler::host::ParseStatus parse_status = pa_scheduler::host::ParseOptions(
        static_cast<int>(common_argv.size()), common_argv.data(), false, &options
    );
    if (parse_status != pa_scheduler::host::ParseStatus::Ok) {
        if (parse_status == pa_scheduler::host::ParseStatus::Help) {
            std::fprintf(
                stderr,
                "CPU winner workload options: [--winner-workload scalar-nop|real-compute] "
                "[--real-compute-count N | --real-compute-counts QK,SF,PV,UP] "
                "[--real-compute-pattern constant|layout-diagnostic]\n"
            );
            std::fprintf(
                stderr,
                "Default: real-compute, constant, counts=6,28,4,1; "
                "scalar-nop is the calibration compatibility mode.\n"
            );
        }
        return parse_status == pa_scheduler::host::ParseStatus::Help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (!pa_scheduler::host::ValidateWinnerWorkloadOptions(workload_options)) {
        return EXIT_FAILURE;
    }
#if PA_BUILD_PERF_CLOCK
    if (options.runs != 1 || options.trace_enabled || options.trace_atomics ||
        options.profile_phases || options.analyze_swimlane ||
        !options.swimlane_json.empty()) {
        std::fprintf(
            stderr,
            "CPU perf-clock requires one trace-free run: "
            "--runs 1 --no-swimlane and no trace/profile options.\n"
        );
        return EXIT_FAILURE;
    }
#endif
    const bool real_compute =
        workload_options.mode == pa_scheduler::WinnerWorkloadMode::RealCompute;
    std::vector<float> workload_image;
    std::vector<float> workload_outputs;
    pa_scheduler::host::PrintBanner("CPU", options);
    pa_scheduler::host::PrintWinnerWorkloadConfig(workload_options, options.nops);
    std::printf(
        "[NOTE] CPU scalar NOP preserves instruction count; real-compute preserves arithmetic "
        "and workspace semantics. Neither represents A5 engine timing or PMU.\n"
    );

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
    std::vector<double> startup_barrier_spans;
    std::vector<double> final_barrier_spans;
    std::vector<double> final_drain_spans;
    std::vector<double> lifecycle_spans;
    bool all_passed = true;
    bool postprocess_ok = true;
#if PA_BUILD_PERF_CLOCK
    std::atomic<uint32_t> perf_clock_reads{0};
    std::atomic<bool> perf_clock_read_shape_ok{true};
#endif
    // 每轮复用大块 host 分配，只重置公共状态和 trace header。与设备后端一样，
    // runs>1 表示同进程热运行，不等价于多个独立首轮。
    for (uint32_t run = 1; run <= options.runs; ++run) {
        pa_scheduler::host::InitializeState(state.get(), options);
#if PTO_FDWIC_SHARED_MAP
        pa_scheduler::host::SharedHostTaskPlan launch_plan;
        pa_scheduler::host::SharedHostHeapAdmission heap_admission;
        std::string admission_error;
        if (!pa_scheduler::host::BuildSharedHostTaskPlan(
                *state, &launch_plan, &admission_error
            ) ||
            !pa_scheduler::host::ValidateSharedHostHeapAdmission(
                launch_plan, state->heap_size,
                &heap_admission, &admission_error
            )) {
            std::fprintf(
                stderr,
                "Shared launch rejected before CPU worker start: %s\n",
                admission_error.c_str()
            );
            all_passed = false;
            break;
        }
        pa_scheduler::host::PrintSharedHostHeapAdmission(
            launch_plan, heap_admission
        );
#endif
        pa_scheduler::host::ConfigureTrace(state.get(), options, trace_memory);
        if (real_compute) {
            // runs>1 必须恢复所有输出 sentinel，避免上一轮 winner 的 tile 被误认
            // 为本轮结果；输入也由公共 helper 恢复成与设备后端相同的选定 pattern。
            pa_scheduler::host::InitializeWinnerWorkloadBuffers(
                workload_options, &workload_image, &workload_outputs
            );
        }
        pa_scheduler::host::ConfigureWinnerWorkload(
            state.get(), workload_options, real_compute ? workload_image.data() : nullptr
        );
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
#if PA_BUILD_PERF_CLOCK
            workers.emplace_back([
                state_pointer = state.get(), worker_id, role,
                &perf_clock_reads, &perf_clock_read_shape_ok
            ]() {
                CpuOps::ResetPerfClockReadCount();
                pa_scheduler::RunScheduler<CpuOps>(state_pointer, worker_id, role);
                const uint32_t reads = CpuOps::PerfClockReadCount();
                perf_clock_reads.fetch_add(reads, std::memory_order_relaxed);
                if (reads != 2) {
                    perf_clock_read_shape_ok.store(false, std::memory_order_relaxed);
                }
            });
#else
            workers.emplace_back([state_pointer = state.get(), worker_id, role]() {
                pa_scheduler::RunScheduler<CpuOps>(state_pointer, worker_id, role);
            });
#endif
        }
        // join 是本后端的 kernel 完成屏障；所有 worker 退出后才能读取最终状态，
        // 对应设备 runner 的 aclrtSynchronizeStream。
        for (std::thread &worker : workers)
            worker.join();
        const auto wall_end = std::chrono::steady_clock::now();
        const double host_us = std::chrono::duration<double, std::micro>(wall_end - wall_begin).count();
        if (real_compute) {
            const size_t output_begin =
                static_cast<size_t>(pa_scheduler::winner_workload::kSharedInputTiles) *
                pa_scheduler::winner_workload::kTileElements;
            std::copy_n(
                workload_image.begin() + output_begin, workload_outputs.size(),
                workload_outputs.begin()
            );
        }
        // host 内存沿用与 A5 相同的 TraceHeader + 每 worker 固定跨度 ABI；
        // 分析器和 raw JSON writer 因而可以与设备后端共用同一回调接口。
        const auto read_trace_records =
            [trace_memory](uint32_t worker, uint32_t count, pa_scheduler::TraceRecord *records) {
                const size_t offset =
                    pa_scheduler::TraceRecordsOffset(worker);
                std::memcpy(
                    records, static_cast<uint8_t *>(trace_memory) + offset,
                    static_cast<size_t>(count) * sizeof(pa_scheduler::TraceRecord)
                );
                return true;
            };
#if PTO_FDWIC_SHARED_MAP
        const auto read_submit_claim_records =
            [trace_memory](
                uint32_t worker, uint32_t count,
                pa_scheduler::SharedSubmitClaimTraceRecord *records
            ) {
                const size_t offset =
                    pa_scheduler::TraceSubmitClaimOffset(worker);
                std::memcpy(
                    records,
                    static_cast<uint8_t *>(trace_memory) + offset,
                    static_cast<size_t>(count) *
                        sizeof(pa_scheduler::SharedSubmitClaimTraceRecord)
                );
                return true;
            };
#endif
        // 先完成严格语义校验，再允许写出；失败运行不会生成可误认成有效
        // 基线的泳道 JSON。
        const pa_scheduler::host::Metrics metrics = pa_scheduler::host::Validate(
            *state, run, host_us, options.trace_enabled ? trace_header : nullptr
        );
#if PA_BUILD_PERF_CLOCK
        const bool perf_clock_reads_ok =
            perf_clock_read_shape_ok.load(std::memory_order_relaxed) &&
            perf_clock_reads.load(std::memory_order_relaxed) == 2U * pa_scheduler::kWorkers;
        std::printf(
            "[PERF-CLOCK] perf_boundary_reads_per_worker=2 "
            "perf_boundary_total_reads=%u expected=%u status=%s\n",
            perf_clock_reads.load(std::memory_order_relaxed),
            2U * pa_scheduler::kWorkers,
            perf_clock_reads_ok ? "PASS" : "FAIL"
        );
#endif
        const bool workload_passed =
            !real_compute || pa_scheduler::host::ValidateRealComputeOutputs(
                *state, workload_options, workload_outputs, run
            );
        all_passed &= metrics.passed && workload_passed;
#if PA_BUILD_PERF_CLOCK
        all_passed &= perf_clock_reads_ok;
#endif
        spans.push_back(metrics.submit_span_us);
        startup_barrier_spans.push_back(metrics.startup_barrier_span_us);
        final_barrier_spans.push_back(metrics.final_barrier_span_us);
        final_drain_spans.push_back(metrics.final_drain_span_us);
        lifecycle_spans.push_back(metrics.lifecycle_span_us);
        // 分析只打印统计，导出则写 raw JSON；两者失败都标记 postprocess，
        // 与调度语义失败分开报告，便于区分协议问题和产物问题。
        if (options.analyze_swimlane &&
            !pa_scheduler::host::AnalyzeSwimlaneRecords(
                *trace_header, *state, read_trace_records
#if PTO_FDWIC_SHARED_MAP
                , read_submit_claim_records
#endif
            )) {
            postprocess_ok = false;
            break;
        }
        if (!options.swimlane_json.empty()) {
            if (!metrics.passed || !workload_passed) {
                std::fprintf(
                    stderr,
                    "Skipping swimlane export because semantic or winner-workload validation failed.\n"
                );
                postprocess_ok = false;
                break;
            }
            if (!pa_scheduler::host::ExportSwimlaneRecords(
#if PTO_FDWIC_SHARED_MAP
                    *trace_header, *state, options.swimlane_json,
#else
                    *trace_header, options.swimlane_json,
#endif
                    workload_options.mode,
                    real_compute
                        ? workload_options.repeats
                        : pa_scheduler::WorkloadCounts{
                              options.nops.qk, options.nops.sf, options.nops.pv, options.nops.up
                          },
                    real_compute
                        ? pa_scheduler::host::RealComputePatternName(workload_options.pattern)
                        : "none",
                    options.final_barrier_shape, options.trace_atomics,
                    read_trace_records
#if PTO_FDWIC_SHARED_MAP
                    , read_submit_claim_records
#endif
                )) {
                postprocess_ok = false;
                break;
            }
        }
    }

    // host 准入可以在第一个 worker 启动前拒绝本轮，此时没有可统计样本。
    // 显式输出 0 和 completed_runs=0，避免把空 vector 交给 Median()。
    const double median_submit_span_us =
        spans.empty() ? 0.0 : pa_scheduler::host::Median(spans);
#if PA_BUILD_PERF_CLOCK
    std::printf(
        "[SUMMARY] runs=%u completed_runs=%zu final_shape=%s "
        "median_submit_span_us=%.3f "
        "lifecycle_timing=disabled semantic_status=%s postprocess_status=%s\n",
        options.runs, spans.size(),
        pa_scheduler::host::FinalBarrierShapeName(options.final_barrier_shape),
        median_submit_span_us,
        all_passed ? "PASS" : "FAIL",
        postprocess_ok ? "PASS" : "FAIL"
    );
#else
    const double median_startup_barrier_us =
        startup_barrier_spans.empty()
            ? 0.0
            : pa_scheduler::host::Median(startup_barrier_spans);
    const double median_final_barrier_us =
        final_barrier_spans.empty()
            ? 0.0
            : pa_scheduler::host::Median(final_barrier_spans);
    const double median_final_drain_us =
        final_drain_spans.empty()
            ? 0.0
            : pa_scheduler::host::Median(final_drain_spans);
    const double median_lifecycle_us =
        lifecycle_spans.empty()
            ? 0.0
            : pa_scheduler::host::Median(lifecycle_spans);
    std::printf(
        "[SUMMARY] runs=%u completed_runs=%zu final_shape=%s "
        "median_submit_span_us=%.3f median_startup_barrier_us=%.3f "
        "median_final_barrier_us=%.3f median_final_drain_us=%.3f median_lifecycle_us=%.3f "
        "semantic_status=%s postprocess_status=%s\n",
        options.runs, spans.size(),
        pa_scheduler::host::FinalBarrierShapeName(
            options.final_barrier_shape
        ),
        median_submit_span_us, median_startup_barrier_us,
        median_final_barrier_us, median_final_drain_us,
        median_lifecycle_us, all_passed ? "PASS" : "FAIL",
        postprocess_ok ? "PASS" : "FAIL"
    );
#endif
    // std::free(nullptr) 合法，因此关闭泳道时也走同一条收尾路径。
    std::free(trace_memory);
    return all_passed && postprocess_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
