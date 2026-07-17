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

#include "acl/acl.h"
#include "runtime/rt.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

bool CheckAcl(aclError error, const char *label) {
    if (error == ACL_SUCCESS) return true;
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), label);
    return false;
}

bool CheckRt(rtError_t error, const char *label) {
    if (error == RT_ERROR_NONE) return true;
    std::fprintf(stderr, "RT error %d: %s\n", static_cast<int>(error), label);
    return false;
}

std::vector<char> ReadBinary(const std::string &path) {
    // ELF 整体保存在 vector 中直到 runtime 卸载完成，保证 rtDevBinary_t.data 在整个注册生命周期内有效。
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) return {};
    return data;
}

}  // namespace

int main(int argc, char **argv) {
    // 参数和 ELF 在创建 ACL 资源前完成校验，早期错误不会留下 device、stream 或 kernel handle。
    pa_scheduler::host::Options options;
    const pa_scheduler::host::ParseStatus parse_status = pa_scheduler::host::ParseOptions(argc, argv, true, &options);
    if (parse_status != pa_scheduler::host::ParseStatus::Ok) {
        return parse_status == pa_scheduler::host::ParseStatus::Help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const std::vector<char> binary_data = ReadBinary(options.kernel_path);
    if (binary_data.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", options.kernel_path.c_str());
        return EXIT_FAILURE;
    }
    pa_scheduler::host::PrintBanner("CCEC", options);

    // 正常及后处理路径依次完成 ACL 初始化、选卡、stream/ELF/设备区创建、launch/D2H
    // 和尾部清理；初始化、传输或 launch 的早期错误仍按当前实现就地返回。
    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(options.device), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) return EXIT_FAILURE;

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, binary_data.data(), binary_data.size()};
    void *kernel_handle = nullptr;
    bool registered_all = true;
    // 先尝试注册带 mixed metadata 的 ELF；若 rtRegisterAllKernel 报错或未返回 handle，
    // 再尝试无 tiling-key 装载。这里仅描述实际回退条件，不假设具体运行时原因。
    rtError_t register_error = rtRegisterAllKernel(&binary, &kernel_handle);
    if (register_error != RT_ERROR_NONE || kernel_handle == nullptr) {
        registered_all = false;
        register_error = rtBinaryLoadWithoutTilingKey(binary_data.data(), binary_data.size(), &kernel_handle);
    }
    if (!CheckRt(register_error, "register mixed AICore ELF") || kernel_handle == nullptr) return EXIT_FAILURE;

    // SchedulerState 保留被测关键 offset、DistCore ABI 和约 1 GiB 生产总跨度；
    // 使用 HUGE_FIRST 降低大块设备内存碎片风险。
    void *state_device = nullptr;
    if (!CheckAcl(
            aclrtMalloc(&state_device, sizeof(pa_scheduler::SchedulerState), ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(state)"
        )) {
        return EXIT_FAILURE;
    }
    if ((reinterpret_cast<uintptr_t>(state_device) & 63U) != 0) {
        std::fprintf(stderr, "Device state is not 64-byte aligned: %p\n", state_device);
        return EXIT_FAILURE;
    }

    // 泳道区按 96 worker 各 65536 条记录预留，约 384 MiB；关闭泳道时不申请，也不会传递有效 base。
    void *trace_device = nullptr;
    if (options.trace_enabled &&
        !CheckAcl(
            aclrtMalloc(&trace_device, pa_scheduler::kTraceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(swimlane trace)"
        )) {
        return EXIT_FAILURE;
    }
    if (options.trace_enabled && (reinterpret_cast<uintptr_t>(trace_device) & 63U) != 0) {
        std::fprintf(stderr, "Device swimlane trace is not 64-byte aligned: %p\n", trace_device);
        return EXIT_FAILURE;
    }

    // host shadow 保留约 1 GiB 总跨度以便按关键 offset 寻址，但每轮传输只选择
    // 共享前缀、控制区和结果区。
    std::unique_ptr<pa_scheduler::SchedulerState> state(new pa_scheduler::SchedulerState);
    pa_scheduler::TraceHeader trace_header{};
    std::vector<double> spans;
    bool all_passed = true;
    bool postprocess_ok = true;
    for (uint32_t run = 1; run <= options.runs; ++run) {
        pa_scheduler::host::InitializeState(state.get(), options);
        pa_scheduler::host::ConfigureTrace(state.get(), options, trace_device);
        if (options.trace_enabled) {
            // 每轮只需重置约 7 KiB header；各 worker 会从 count=0 覆盖自己的记录区，无需清零整块 384 MiB。
            pa_scheduler::host::InitializeTraceHeader(&trace_header);
            if (!CheckAcl(
                    aclrtMemcpy(
                        trace_device, sizeof(trace_header), &trace_header, sizeof(trace_header),
                        ACL_MEMCPY_HOST_TO_DEVICE
                    ),
                    "aclrtMemcpy(H2D swimlane header)"
                )) {
                return EXIT_FAILURE;
            }
        }
        // 为避免每轮搬运约 1 GiB，只 H2D 被测共享前缀和位于生产总跨度之后的
        // standalone 控制区；
        // 每个 worker 的大块私有状态由 device kernel 自行初始化。
        if (!CheckAcl(
                aclrtMemcpy(
                    state_device, pa_scheduler::host::StatePrefixBytes(), state.get(),
                    pa_scheduler::host::StatePrefixBytes(), ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D state prefix)"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &static_cast<pa_scheduler::SchedulerState *>(state_device)->config,
                    pa_scheduler::host::ControlBytes(), &state->config, pa_scheduler::host::ControlBytes(),
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D standalone controls)"
            )) {
            return EXIT_FAILURE;
        }

        void *kernel_args[] = {state_device};
        rtArgsEx_t args_info{};
        args_info.args = kernel_args;
        args_info.argsSize = sizeof(kernel_args);
        rtTaskCfgInfo_t task_config{};
        // launch 维度是 32 个物理 mixed block；ELF metadata 让每个 block 同时产生 1 AIC + 2 AIV，共 96 worker。
        // wall time 在同步完成处截止，包含 launch、完整调度、最终 drain 和 stream 同步，但不包含后续 D2H/JSON。
        const auto wall_begin = std::chrono::steady_clock::now();
        if (!CheckRt(
                rtKernelLaunchWithHandleV2(
                    kernel_handle, 0, pa_scheduler::kAicWorkers, &args_info, nullptr, stream, &task_config
                ),
                "rtKernelLaunchWithHandleV2"
            ) ||
            !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
            return EXIT_FAILURE;
        }
        const auto wall_end = std::chrono::steady_clock::now();
        const double host_us = std::chrono::duration<double, std::micro>(wall_end - wall_begin).count();
        // D2H 同样避开约 1 GiB 的 worker arena：共享前缀用于 flag/vend/frontier 校验，末尾 results 单独回传。
        if (!CheckAcl(
                aclrtMemcpy(
                    state.get(), pa_scheduler::host::StatePrefixBytes(), state_device,
                    pa_scheduler::host::StatePrefixBytes(), ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H state prefix)"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    state->results, pa_scheduler::host::ResultBytes(),
                    &static_cast<pa_scheduler::SchedulerState *>(state_device)->results[0],
                    pa_scheduler::host::ResultBytes(), ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H worker results)"
            )) {
            return EXIT_FAILURE;
        }
        if (options.trace_enabled &&
            !CheckAcl(
                aclrtMemcpy(
                    &trace_header, sizeof(trace_header), trace_device, sizeof(trace_header),
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H swimlane header)"
            )) {
            return EXIT_FAILURE;
        }
        // 常规校验只需 header 中的 per-worker count；真实 records 在分析或导出时才按核、按实际 count 懒加载。
        const auto read_trace_records =
            [trace_device](uint32_t worker, uint32_t count, pa_scheduler::TraceRecord *records) {
                // 每核记录区采用固定容量 stride；只复制 header 声明的实际 count，避免 D2H 未使用的尾部空间。
                const uint64_t offset = sizeof(pa_scheduler::TraceHeader) +
                                        static_cast<uint64_t>(worker) * pa_scheduler::kTraceRecordsPerCore *
                                            sizeof(pa_scheduler::TraceRecord);
                return CheckAcl(
                    aclrtMemcpy(
                        records, static_cast<size_t>(count) * sizeof(pa_scheduler::TraceRecord),
                        static_cast<uint8_t *>(trace_device) + offset,
                        static_cast<size_t>(count) * sizeof(pa_scheduler::TraceRecord),
                        ACL_MEMCPY_DEVICE_TO_HOST
                    ),
                    "aclrtMemcpy(D2H swimlane records)"
                );
            };
        // 先完成共享状态、拓扑、计数和 trace header 的语义校验，再允许 raw JSON 成为性能证据。
        const pa_scheduler::host::Metrics metrics = pa_scheduler::host::Validate(
            *state, run, host_us, options.trace_enabled ? &trace_header : nullptr
        );
        all_passed &= metrics.passed;
        spans.push_back(metrics.submit_span_us);
        if (options.analyze_swimlane &&
            !pa_scheduler::host::AnalyzeSwimlaneRecords(trace_header, *state, read_trace_records)) {
            // 后处理错误使用 break 汇入统一 cleanup；与初始化/launch 失败的进程级立即返回语义区分开。
            postprocess_ok = false;
            break;
        }
        if (!options.swimlane_json.empty()) {
            // 只有语义校验通过才把 raw JSON 经“临时文件写完后 rename”发布，
            // 避免把截断或错误调度结果误当成可用性能证据。
            if (!metrics.passed) {
                std::fprintf(stderr, "Skipping swimlane export because semantic validation failed.\n");
                postprocess_ok = false;
                break;
            }
            if (!pa_scheduler::host::ExportSwimlaneRecords(
                    trace_header, options.swimlane_json, read_trace_records
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

    // 后处理失败也统一走设备资源释放、ELF 卸载和 ACL 收尾，避免文件系统错误遗留运行时上下文。
    bool cleanup_ok = true;
    // 先释放依赖当前 device/context 的大块内存，再卸载 ELF、销毁 stream，最后 reset device 与 finalize ACL。
    if (trace_device != nullptr) {
        cleanup_ok &= CheckAcl(aclrtFree(trace_device), "aclrtFree(swimlane trace)");
    }
    cleanup_ok &= CheckAcl(aclrtFree(state_device), "aclrtFree(state)");
    const rtError_t unload_error =
        registered_all ? rtDevBinaryUnRegister(kernel_handle) : rtBinaryUnLoad(kernel_handle);
    cleanup_ok &= CheckRt(unload_error, "unload mixed AICore ELF");
    cleanup_ok &= CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup_ok &= CheckAcl(aclrtResetDevice(options.device), "aclrtResetDevice");
    cleanup_ok &= CheckAcl(aclFinalize(), "aclFinalize");
    // 运行语义、后处理和资源清理三者全部成功，进程才返回成功，脚本据此决定是否继续生成 merged 泳道。
    return all_passed && postprocess_ok && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
