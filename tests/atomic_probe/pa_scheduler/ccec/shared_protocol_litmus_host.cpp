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

#include "../common/pa_model.h"
#include "shared_protocol_litmus_shared.h"

#include "acl/acl.h"
#include "runtime/rt.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace {

using pa_scheduler::SchedulerState;
using pa_scheduler::SharedWriterHistoryCell;
using pa_scheduler::SharedWriterHistoryRecord;
using pa_scheduler::TensorDesc;
using pa_scheduler::WorkerResult;
using pa_scheduler::shared_protocol_litmus::Control;
using pa_scheduler::shared_protocol_litmus::Direction;
using pa_scheduler::shared_protocol_litmus::HistoryChain;
using pa_scheduler::shared_protocol_litmus::Scenario;
using pa_scheduler::shared_protocol_litmus::kAicToAiv;
using pa_scheduler::shared_protocol_litmus::kAivToAic;
using pa_scheduler::shared_protocol_litmus::kControlMagic;
using pa_scheduler::shared_protocol_litmus::kControlVersion;
using pa_scheduler::shared_protocol_litmus::kFutureWritersStatus;
using pa_scheduler::shared_protocol_litmus::kReaderStatus;
using pa_scheduler::shared_protocol_litmus::kResultMagic;
using pa_scheduler::shared_protocol_litmus::kSymbolCount;
using pa_scheduler::shared_protocol_litmus::kWriterBStatus;

constexpr size_t kStatePrefixBytes =
    offsetof(SchedulerState, workers);
constexpr size_t kResultBytes =
    sizeof(WorkerResult) * pa_scheduler::kWorkers;
constexpr size_t kSharedSidecarBytes =
    sizeof(pa_scheduler::SharedTensorMapSidecar);

bool CheckAcl(aclError error, const char *label) {
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(
        stderr, "ACL error %d: %s\n",
        static_cast<int>(error), label
    );
    return false;
}

bool CheckRt(rtError_t error, const char *label) {
    if (error == RT_ERROR_NONE) {
        return true;
    }
    std::fprintf(
        stderr, "RT error %d: %s\n",
        static_cast<int>(error), label
    );
    return false;
}

std::vector<char> ReadBinary(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return {};
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) {
        return {};
    }
    return data;
}

SchedulerState *MapSparseState() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(SchedulerState),
        PROT_READ | PROT_WRITE, flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    return ::new (memory) SchedulerState;
}

bool ParseDevice(const char *text, int32_t *device) {
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < 0 || parsed > INT32_MAX) {
        return false;
    }
    *device = static_cast<int32_t>(parsed);
    return true;
}

bool ParseDirection(
    const char *text, Direction *direction,
    const HistoryChain **chain
) {
    const std::string name = text;
    if (name == "aic-to-aiv") {
        *direction = Direction::AicToAiv;
        *chain = &kAicToAiv;
        return true;
    }
    if (name == "aiv-to-aic") {
        *direction = Direction::AivToAic;
        *chain = &kAivToAic;
        return true;
    }
    return false;
}

void ResetTaskGate(SchedulerState *state, int32_t task_id) {
    pa_scheduler::TaskCell &task =
        state->tasks[static_cast<uint32_t>(task_id)];
    task.flag = 0;
    task.deps_prepared = -1;
}

void InitializeDescriptor(
    TensorDesc *tensor, const HistoryChain &chain,
    uint32_t slot
) {
    std::memset(tensor, 0, sizeof(*tensor));
    tensor->buffer_addr =
        0x500000000ULL +
        static_cast<uint64_t>(chain.producer) * 0x100000ULL +
        static_cast<uint64_t>(slot) * 0x10000ULL;
    tensor->buffer_size = 4096;
    tensor->owner_task_id =
        static_cast<uint64_t>(chain.producer);
    tensor->ndims = 1;
    tensor->dtype = pa_scheduler::DataType::Float32;
    tensor->is_contiguous = true;
    tensor->shapes[0] = 1024;
    tensor->strides[0] = 1;
    tensor->extent_elem_cache = 1024;
}

void InitializeHistoryState(
    SchedulerState *state, const HistoryChain &chain
) {
    // 约 1 GiB shadow 由匿名稀疏映射承载；只触碰实际 H2D/D2H 的前缀、
    // results 和 shared sidecar，不为门槛制造无意义的整块主机写流量。
    std::memset(state, 0, kStatePrefixBytes);
    std::memset(state->results, 0, kResultBytes);
    std::memset(
        &state->shared_map, 0, kSharedSidecarBytes
    );
    state->fatal.value = 0;
    state->heap_window = pa_scheduler::kHeapWindow;
    for (uint32_t worker = 0;
         worker < pa_scheduler::kWorkers; ++worker) {
        state->shared_map.reader_done[worker].value = -1;
    }

    ResetTaskGate(state, chain.writer_b);
    ResetTaskGate(state, chain.writer_d);
    ResetTaskGate(state, chain.writer_e);
    ResetTaskGate(state, chain.reader_past_b_signal);
    ResetTaskGate(state, chain.future_done_signal);

    pa_scheduler::SharedOutputCell &cell =
        state->shared_map.shared_outputs[
            static_cast<uint32_t>(chain.producer)
        ];
    for (uint32_t slot = 0; slot < kSymbolCount; ++slot) {
        InitializeDescriptor(
            &cell.tensors[slot], chain, slot
        );
        cell.published[slot].value = chain.producer;
        cell.last_writer[slot].value = chain.producer;
    }
}

bool Expect(bool condition, const char *label) {
    std::printf(
        "[ASSERT] %-62s %s\n",
        label, condition ? "PASS" : "FAIL"
    );
    return condition;
}

bool HistoryMatches(
    const SharedWriterHistoryCell &history,
    const HistoryChain &chain, int32_t writer,
    int32_t predecessor
) {
    if (history.magic !=
            pa_scheduler::kSharedWriterHistoryMagic ||
        history.writer_task != writer ||
        history.count != kSymbolCount ||
        history.reserved != 0) {
        return false;
    }
    for (uint32_t slot = 0; slot < kSymbolCount; ++slot) {
        const SharedWriterHistoryRecord &record =
            history.entries[slot];
        const uint32_t expected_key =
            static_cast<uint32_t>(chain.producer) *
                pa_scheduler::kSharedOutputMaxPerTask +
            slot + 1U;
        if (record.symbol_key != expected_key ||
            record.previous_writer != predecessor) {
            return false;
        }
    }
    return true;
}

bool ResultMatches(
    const WorkerResult &result, uint64_t tag,
    uint64_t status, uint64_t quantity, int32_t fanin
) {
    return result.submit_begin == (kResultMagic | tag) &&
           result.submit_end == status &&
           result.finish_cycle == quantity &&
           static_cast<int64_t>(result.checksum) == fanin;
}

bool ValidateHistory(
    const SchedulerState &state, const HistoryChain &chain,
    const HistoryChain &inactive
) {
    bool passed = true;
    passed &= Expect(
        state.fatal.value == 0,
        "device protocol leaves fatal clear"
    );
    passed &= Expect(
        state.tasks[chain.writer_b].deps_prepared ==
                chain.writer_b &&
            state.tasks[chain.writer_d].deps_prepared ==
                chain.writer_d &&
            state.tasks[chain.writer_e].deps_prepared ==
                chain.writer_e,
        "B/D/E each publish their own writer-ready gate"
    );
    passed &= Expect(
        state.tasks[chain.reader_past_b_signal]
                .deps_prepared ==
            chain.reader_past_b_signal &&
            state.tasks[chain.future_done_signal]
                    .deps_prepared ==
                chain.future_done_signal,
        "reader-past-B and future-done ordering gates both close"
    );
    passed &= Expect(
        state.tasks[chain.writer_b].flag == 0 &&
            state.tasks[chain.writer_d].flag == 0 &&
            state.tasks[chain.writer_e].flag == 0,
        "writer-ready remains independent from kernel completion"
    );

    const WorkerResult &writer_b =
        state.results[chain.writer_b_worker];
    const WorkerResult &future =
        state.results[chain.future_worker];
    const WorkerResult &reader =
        state.results[chain.reader_worker];
    passed &= Expect(
        ResultMatches(
            writer_b, chain.result_tag | 1U,
            kWriterBStatus, kSymbolCount, chain.producer
        ),
        "B publishes seven symbol CAS operations after A"
    );
    passed &= Expect(
        ResultMatches(
            future, chain.result_tag | 2U,
            kFutureWritersStatus, 2 * kSymbolCount,
            chain.writer_d
        ),
        "D then E publish fourteen ordered symbol CAS operations"
    );
    passed &= Expect(
        ResultMatches(
            reader, chain.result_tag | 3U,
            kReaderStatus, 1, chain.writer_b
        ),
        "slow C resolves one fanin and returns B after E->D->B"
    );
    passed &= Expect(
        reader.submits == 0 &&
            reader.claim_attempts == 0,
        "C really prewarms both future history cache lines as zero"
    );

    const pa_scheduler::SharedOutputCell &cell =
        state.shared_map.shared_outputs[
            static_cast<uint32_t>(chain.producer)
        ];
    bool latest_ok = true;
    for (uint32_t slot = 0; slot < kSymbolCount; ++slot) {
        latest_ok &=
            cell.published[slot].value == chain.producer &&
            cell.last_writer[slot].value == chain.writer_e;
    }
    passed &= Expect(
        latest_ok,
        "all seven symbol latest cells advance from A to E"
    );
    passed &= Expect(
        HistoryMatches(
            state.shared_map.writer_history[chain.writer_b],
            chain, chain.writer_b, chain.producer
        ) &&
            HistoryMatches(
                state.shared_map.writer_history[chain.writer_d],
                chain, chain.writer_d, chain.writer_b
            ) &&
            HistoryMatches(
                state.shared_map.writer_history[chain.writer_e],
                chain, chain.writer_e, chain.writer_d
            ),
        "B/D/E immutable histories preserve all seven predecessors"
    );

    const SharedWriterHistoryCell &inactive_b =
        state.shared_map.writer_history[inactive.writer_b];
    const SharedWriterHistoryCell &inactive_d =
        state.shared_map.writer_history[inactive.writer_d];
    const SharedWriterHistoryCell &inactive_e =
        state.shared_map.writer_history[inactive.writer_e];
    passed &= Expect(
        inactive_b.magic == 0 && inactive_b.count == 0 &&
            inactive_d.magic == 0 && inactive_d.count == 0 &&
            inactive_e.magic == 0 && inactive_e.count == 0,
        "the opposite direction remains untouched in this launch"
    );

    bool ordinary_ring_untouched = true;
    for (uint32_t bucket = 0;
         bucket < pa_scheduler::kMapBuckets; ++bucket) {
        ordinary_ring_untouched &=
            state.shared_map.buckets[bucket].head.value == 0 &&
            state.shared_map.buckets[bucket].tail.value == 0;
    }
    for (uint32_t worker = 0;
         worker < pa_scheduler::kWorkers; ++worker) {
        ordinary_ring_untouched &=
            state.shared_map.reader_done[worker].value == -1;
    }
    passed &= Expect(
        ordinary_ring_untouched,
        "symbol history litmus leaves ordinary ring/progress untouched"
    );
    return passed;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(
            stderr,
            "Usage: %s <shared_protocol_litmus_kernel.o> "
            "history aic-to-aiv|aiv-to-aic [device]\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }

    if (std::strcmp(argv[2], "history") != 0) {
        std::fprintf(
            stderr, "Invalid shared protocol scenario: %s\n",
            argv[2]
        );
        return EXIT_FAILURE;
    }
    Direction direction{};
    const HistoryChain *chain = nullptr;
    if (!ParseDirection(argv[3], &direction, &chain)) {
        std::fprintf(
            stderr, "Invalid history direction: %s\n", argv[3]
        );
        return EXIT_FAILURE;
    }
    const HistoryChain &inactive =
        direction == Direction::AicToAiv
            ? kAivToAic
            : kAicToAiv;
    int32_t device = 0;
    if (argc == 5 && !ParseDevice(argv[4], &device)) {
        std::fprintf(stderr, "Invalid device id: %s\n", argv[4]);
        return EXIT_FAILURE;
    }
    const std::vector<char> binary = ReadBinary(argv[1]);
    if (binary.empty()) {
        std::fprintf(
            stderr, "Cannot read mixed AICore ELF: %s\n", argv[1]
        );
        return EXIT_FAILURE;
    }

    SchedulerState *host_state = MapSparseState();
    if (host_state == nullptr) {
        return EXIT_FAILURE;
    }
    InitializeHistoryState(host_state, *chain);
    Control host_control{};
    host_control.magic = kControlMagic;
    host_control.version = kControlVersion;
    host_control.scenario =
        static_cast<uint32_t>(Scenario::SymbolHistory);
    host_control.direction = static_cast<uint32_t>(direction);
    host_control.launch_nonce =
        static_cast<uint64_t>(chain->producer) << 32 |
        static_cast<uint32_t>(chain->reader_c);

    bool acl_initialized = false;
    bool device_set = false;
    aclrtStream stream = nullptr;
    void *kernel_handle = nullptr;
    bool registered_all = false;
    void *device_state = nullptr;
    void *device_control = nullptr;
    bool execution_ok = false;

    do {
        if (!CheckAcl(aclInit(nullptr), "aclInit")) {
            break;
        }
        acl_initialized = true;
        if (!CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) {
            break;
        }
        device_set = true;
        if (!CheckAcl(
                aclrtCreateStream(&stream), "aclrtCreateStream"
            )) {
            break;
        }

        rtDevBinary_t device_binary{
            RT_DEV_BINARY_MAGIC_ELF, 0,
            binary.data(), binary.size()
        };
        rtError_t load_error =
            rtRegisterAllKernel(&device_binary, &kernel_handle);
        if (load_error == RT_ERROR_NONE &&
            kernel_handle != nullptr) {
            registered_all = true;
        } else {
            registered_all = false;
            kernel_handle = nullptr;
            load_error = rtBinaryLoadWithoutTilingKey(
                binary.data(), binary.size(), &kernel_handle
            );
        }
        if (!CheckRt(
                load_error, "load history mixed AICore ELF"
            ) ||
            kernel_handle == nullptr) {
            break;
        }

        if (!CheckAcl(
                aclrtMalloc(
                    &device_state, sizeof(SchedulerState),
                    ACL_MEM_MALLOC_HUGE_FIRST
                ),
                "aclrtMalloc(history SchedulerState)"
            ) ||
            !CheckAcl(
                aclrtMalloc(
                    &device_control, sizeof(Control),
                    ACL_MEM_MALLOC_NORMAL_ONLY
                ),
                "aclrtMalloc(history control)"
            )) {
            break;
        }
        if ((reinterpret_cast<uintptr_t>(device_state) & 63U) !=
                0 ||
            (reinterpret_cast<uintptr_t>(device_control) & 63U) !=
                0) {
            std::fprintf(
                stderr,
                "History allocations must be 64-byte aligned: "
                "state=%p control=%p\n",
                device_state, device_control
            );
            break;
        }

        if (!CheckAcl(
                aclrtMemcpy(
                    device_state, kStatePrefixBytes,
                    host_state, kStatePrefixBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D history state prefix"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &static_cast<SchedulerState *>(
                         device_state
                     )->results[0],
                    kResultBytes, host_state->results,
                    kResultBytes, ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D zero history results"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &static_cast<SchedulerState *>(
                         device_state
                     )->shared_map,
                    kSharedSidecarBytes,
                    &host_state->shared_map,
                    kSharedSidecarBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D history shared sidecar"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    device_control, sizeof(Control),
                    &host_control, sizeof(Control),
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "H2D history control"
            )) {
            break;
        }

        void *kernel_args[] = {
            device_state, device_control
        };
        rtArgsEx_t args_info{};
        args_info.args = kernel_args;
        args_info.argsSize = sizeof(kernel_args);
        rtTaskCfgInfo_t task_config{};
        if (!CheckRt(
                rtKernelLaunchWithHandleV2(
                    kernel_handle, 0,
                    pa_scheduler::kAicWorkers,
                    &args_info, nullptr, stream,
                    &task_config
                ),
                "launch history mixed AICore kernel"
            ) ||
            !CheckAcl(
                aclrtSynchronizeStream(stream),
                "synchronize history mixed AICore kernel"
            )) {
            break;
        }

        if (!CheckAcl(
                aclrtMemcpy(
                    host_state, kStatePrefixBytes,
                    device_state, kStatePrefixBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "D2H history state prefix"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    host_state->results, kResultBytes,
                    &static_cast<SchedulerState *>(
                         device_state
                     )->results[0],
                    kResultBytes, ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "D2H history results"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &host_state->shared_map,
                    kSharedSidecarBytes,
                    &static_cast<SchedulerState *>(
                         device_state
                     )->shared_map,
                    kSharedSidecarBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "D2H history shared sidecar"
            )) {
            break;
        }
        execution_ok = ValidateHistory(
            *host_state, *chain, inactive
        );
    } while (false);

    bool cleanup_ok = true;
    if (device_control != nullptr) {
        cleanup_ok &=
            CheckAcl(
                aclrtFree(device_control),
                "aclrtFree(history control)"
            );
    }
    if (device_state != nullptr) {
        cleanup_ok &=
            CheckAcl(
                aclrtFree(device_state),
                "aclrtFree(history SchedulerState)"
            );
    }
    if (kernel_handle != nullptr) {
        const rtError_t unload_error = registered_all
            ? rtDevBinaryUnRegister(kernel_handle)
            : rtBinaryUnLoad(kernel_handle);
        cleanup_ok &=
            CheckRt(
                unload_error, "unload history mixed AICore ELF"
            );
    }
    if (stream != nullptr) {
        cleanup_ok &=
            CheckAcl(
                aclrtDestroyStream(stream),
                "aclrtDestroyStream"
            );
    }
    if (device_set) {
        cleanup_ok &=
            CheckAcl(
                aclrtResetDevice(device), "aclrtResetDevice"
            );
    }
    if (acl_initialized) {
        cleanup_ok &=
            CheckAcl(aclFinalize(), "aclFinalize");
    }
    (void)munmap(host_state, sizeof(SchedulerState));

    std::printf(
        "[SHARED-PROTOCOL-LITMUS] scenario=%s direction=%s "
        "device=%d semantic=%s cleanup=%s\n",
        argv[2], argv[3], device,
        execution_ok ? "PASS" : "FAIL",
        cleanup_ok ? "PASS" : "FAIL"
    );
    return execution_ok && cleanup_ok
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
