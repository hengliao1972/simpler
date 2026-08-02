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

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define PA_DEVICE inline
#define PA_GM
#include "../../common/shared_exec_protocol.h"

namespace {

using namespace pa_scheduler::cross_core;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] cross-core protocol: %s\n", message);
    ++g_failures;
}

// 用显式暂停点构造协议交错，不依赖线程调度碰巧命中窗口。
class PausePoint {
public:
    void Enable() {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = true;
        reached_ = false;
        released_ = false;
    }

    void Disable() {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = false;
        reached_ = false;
        released_ = true;
        condition_.notify_all();
    }

    void StopHereIfEnabled() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!enabled_) {
            return;
        }
        reached_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this] { return released_; });
    }

    void WaitUntilReached() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return reached_; });
    }

    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool enabled_ = false;
    bool reached_ = false;
    bool released_ = false;
};

PausePoint g_mid_pack_pause;
PausePoint g_before_built_pause;
PausePoint g_invalid_fanin_pause;

struct ProtocolTestOps {
    static constexpr uint32_t kTraceActorCount = 32;
    static constexpr uint32_t kNoTraceActor = UINT32_MAX;
    static inline thread_local uint32_t trace_actor = kNoTraceActor;
    static inline std::atomic<uint64_t> load_calls{0};
    static inline std::atomic<uint64_t> cas_calls{0};
    static inline std::atomic<uint64_t> payload_stores{0};
    static inline std::atomic<uint64_t> payload_loads{0};
    static inline std::atomic<uint64_t> token_stores{0};
    static inline std::atomic<uint64_t> flush_calls{0};
    static inline std::atomic<uint64_t> invalidate_calls{0};
    static inline std::atomic<uintptr_t> flush_address{0};
    static inline std::atomic<uint64_t> flush_bytes{0};
    static inline std::atomic<uintptr_t> invalidate_address{0};
    static inline std::atomic<uint64_t> invalidate_bytes{0};
    static inline std::atomic<uint64_t> preload_build_calls{0};
    static inline std::atomic<uint64_t> preload_source_calls{0};
    static inline std::atomic<uint64_t> preload_token_calls{0};
    static inline std::atomic<uintptr_t> preload_build_address{0};
    static inline std::atomic<uintptr_t> preload_source_address{0};
    static inline std::atomic<uintptr_t> preload_token_address{0};
    static inline std::atomic<uint64_t> preload_build_bytes{0};
    static inline std::atomic<uint64_t> preload_source_bytes{0};
    static inline std::atomic<uint64_t> preload_token_bytes{0};
    static inline std::atomic<uint64_t> event_clock{0};
    static inline std::atomic<uint64_t> last_payload_store_event{0};
    static inline std::atomic<uint64_t> flush_event{0};
    static inline std::atomic<uint64_t> built_cas_event{0};
    static inline std::atomic<uint64_t> claim_cas_event{0};
    static inline std::atomic<uint64_t> invalidate_event{0};
    static inline std::atomic<uint64_t> first_payload_load_event{0};
    static inline std::atomic<uint64_t> first_token_store_event{0};
    static inline std::atomic<uintptr_t> fatal_address{0};
    static inline std::atomic<uint64_t> fatal_cas_calls{0};
    static inline std::array<std::atomic<uint64_t>, kTraceActorCount>
        actor_cas_calls{};
    static inline std::array<std::atomic<uint64_t>, kTraceActorCount>
        actor_invalidate_calls{};
    static inline std::array<std::atomic<uint64_t>, kTraceActorCount>
        actor_payload_loads{};

    static uint64_t NextEvent() {
        return event_clock.fetch_add(
                   1, std::memory_order_relaxed
               ) + 1U;
    }

    static void SetTraceActor(uint32_t actor) {
        trace_actor = actor;
    }

    static void ClearTraceActor() {
        trace_actor = kNoTraceActor;
    }

    static void SetFatalAddress(const void *address) {
        fatal_address.store(
            reinterpret_cast<uintptr_t>(address),
            std::memory_order_relaxed
        );
    }

    static void ResetTrace() {
        load_calls.store(0, std::memory_order_relaxed);
        cas_calls.store(0, std::memory_order_relaxed);
        payload_stores.store(0, std::memory_order_relaxed);
        payload_loads.store(0, std::memory_order_relaxed);
        token_stores.store(0, std::memory_order_relaxed);
        flush_calls.store(0, std::memory_order_relaxed);
        invalidate_calls.store(0, std::memory_order_relaxed);
        flush_address.store(0, std::memory_order_relaxed);
        flush_bytes.store(0, std::memory_order_relaxed);
        invalidate_address.store(0, std::memory_order_relaxed);
        invalidate_bytes.store(0, std::memory_order_relaxed);
        preload_build_calls.store(0, std::memory_order_relaxed);
        preload_source_calls.store(0, std::memory_order_relaxed);
        preload_token_calls.store(0, std::memory_order_relaxed);
        preload_build_address.store(0, std::memory_order_relaxed);
        preload_source_address.store(0, std::memory_order_relaxed);
        preload_token_address.store(0, std::memory_order_relaxed);
        preload_build_bytes.store(0, std::memory_order_relaxed);
        preload_source_bytes.store(0, std::memory_order_relaxed);
        preload_token_bytes.store(0, std::memory_order_relaxed);
        event_clock.store(0, std::memory_order_relaxed);
        last_payload_store_event.store(0, std::memory_order_relaxed);
        flush_event.store(0, std::memory_order_relaxed);
        built_cas_event.store(0, std::memory_order_relaxed);
        claim_cas_event.store(0, std::memory_order_relaxed);
        invalidate_event.store(0, std::memory_order_relaxed);
        first_payload_load_event.store(0, std::memory_order_relaxed);
        first_token_store_event.store(0, std::memory_order_relaxed);
        fatal_cas_calls.store(0, std::memory_order_relaxed);
        for (uint32_t actor = 0; actor < kTraceActorCount; ++actor) {
            actor_cas_calls[actor].store(0, std::memory_order_relaxed);
            actor_invalidate_calls[actor].store(
                0, std::memory_order_relaxed
            );
            actor_payload_loads[actor].store(
                0, std::memory_order_relaxed
            );
        }
    }

    static int64_t Load(volatile int64_t *address) {
        load_calls.fetch_add(1, std::memory_order_relaxed);
        return __atomic_fetch_add(
            address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE
        );
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        cas_calls.fetch_add(1, std::memory_order_relaxed);
        if (trace_actor < kTraceActorCount) {
            actor_cas_calls[trace_actor].fetch_add(
                1, std::memory_order_relaxed
            );
        }
        const uint64_t event = NextEvent();
        if (reinterpret_cast<uintptr_t>(address) ==
            fatal_address.load(std::memory_order_relaxed)) {
            fatal_cas_calls.fetch_add(1, std::memory_order_relaxed);
        } else {
            const DecodedExecState desired_state =
                DecodeExecState(desired);
            if (desired_state.valid &&
                desired_state.phase == ExecPhase::Built) {
                built_cas_event.store(event, std::memory_order_relaxed);
            } else if (desired_state.valid &&
                       desired_state.phase == ExecPhase::Claimed) {
                claim_cas_event.store(event, std::memory_order_relaxed);
            }
        }
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }

    static void BeforeBuiltPublish(uint32_t) {
        g_before_built_pause.StopHereIfEnabled();
    }

    static void BeforePayloadAcquire(uint32_t) {}

    static void PreloadBuildDestination(
        const void *address, uint64_t bytes
    ) {
        preload_build_calls.fetch_add(1, std::memory_order_relaxed);
        preload_build_address.store(
            reinterpret_cast<uintptr_t>(address),
            std::memory_order_relaxed
        );
        preload_build_bytes.store(bytes, std::memory_order_relaxed);
    }

    static void PreloadPayloadSource(
        const void *address, uint64_t bytes
    ) {
        preload_source_calls.fetch_add(1, std::memory_order_relaxed);
        preload_source_address.store(
            reinterpret_cast<uintptr_t>(address),
            std::memory_order_relaxed
        );
        preload_source_bytes.store(bytes, std::memory_order_relaxed);
    }

    static void PreloadTokenDestination(
        const void *address, uint64_t bytes
    ) {
        preload_token_calls.fetch_add(1, std::memory_order_relaxed);
        preload_token_address.store(
            reinterpret_cast<uintptr_t>(address),
            std::memory_order_relaxed
        );
        preload_token_bytes.store(bytes, std::memory_order_relaxed);
    }

    static void StorePayloadWord(
        volatile uint64_t *address, uint64_t value
    ) {
        payload_stores.fetch_add(1, std::memory_order_relaxed);
        *address = value;
        last_payload_store_event.store(
            NextEvent(), std::memory_order_relaxed
        );
    }

    static uint64_t LoadPayloadWord(
        const volatile uint64_t *address
    ) {
        payload_loads.fetch_add(1, std::memory_order_relaxed);
        if (trace_actor < kTraceActorCount) {
            actor_payload_loads[trace_actor].fetch_add(
                1, std::memory_order_relaxed
            );
        }
        const uint64_t event = NextEvent();
        uint64_t unset = 0;
        (void)first_payload_load_event.compare_exchange_strong(
            unset, event, std::memory_order_relaxed
        );
        return *address;
    }

    static void StoreTokenPayloadWord(
        volatile uint64_t *address, uint64_t value
    ) {
        token_stores.fetch_add(1, std::memory_order_relaxed);
        const uint64_t event = NextEvent();
        uint64_t unset = 0;
        (void)first_token_store_event.compare_exchange_strong(
            unset, event, std::memory_order_relaxed
        );
        *address = value;
    }

    static void FlushRegion(void *address, uint64_t bytes) {
        flush_calls.fetch_add(1, std::memory_order_relaxed);
        flush_address.store(
            reinterpret_cast<uintptr_t>(address),
            std::memory_order_relaxed
        );
        flush_bytes.store(bytes, std::memory_order_relaxed);
        flush_event.store(NextEvent(), std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void InvalidateRegion(
        const void *address, uint64_t bytes
    ) {
        invalidate_calls.fetch_add(1, std::memory_order_relaxed);
        if (trace_actor < kTraceActorCount) {
            actor_invalidate_calls[trace_actor].fetch_add(
                1, std::memory_order_relaxed
            );
        }
        invalidate_address.store(
            reinterpret_cast<uintptr_t>(address),
            std::memory_order_relaxed
        );
        invalidate_bytes.store(bytes, std::memory_order_relaxed);
        invalidate_event.store(NextEvent(), std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
};

struct SyntheticPayloadSource {
    std::array<std::array<uint64_t, kExecTensorDescWords>, 3>
        tensors{};
    std::array<uint64_t, 4> scalars{};
    std::array<int32_t, 4> fanin{};
    bool pause_mid_pack = false;
    bool pause_invalid_fanin = false;

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const {
        if (pause_mid_pack && tensor == 1 && word == 0) {
            g_mid_pack_pause.StopHereIfEnabled();
        }
        return tensors[tensor][word];
    }

    uint64_t Scalar(uint32_t index) const {
        return scalars[index];
    }

    int32_t Fanin(uint32_t index) const {
        if (pause_invalid_fanin && index == 1) {
            g_invalid_fanin_pause.StopHereIfEnabled();
        }
        return fanin[index];
    }
};

struct ReadyTable {
    std::array<std::atomic<bool>, 64> ready{};
    mutable std::array<std::atomic<uint32_t>, 64> reads{};

    bool IsReady(int32_t task_id) const {
        reads[static_cast<uint32_t>(task_id)].fetch_add(
            1, std::memory_order_relaxed
        );
        return ready[static_cast<uint32_t>(task_id)].load(
            std::memory_order_acquire
        );
    }

    uint32_t ReadCount(int32_t task_id) const {
        return reads[static_cast<uint32_t>(task_id)].load(
            std::memory_order_relaxed
        );
    }
};

struct CompletionRecorder {
    std::array<uint32_t, 4> events{};
    uint32_t event_count = 0;
    uint32_t vend_calls = 0;
    uint32_t flag_calls = 0;
    uint32_t vend_task = UINT32_MAX;
    uint32_t flag_task = UINT32_MAX;
    uint64_t vend = 0;
    bool allow_vend = true;
    bool allow_flag = true;

    bool PublishVend(uint32_t task_id, uint64_t value) {
        ++vend_calls;
        if (!allow_vend) return false;
        vend_task = task_id;
        vend = value;
        events[event_count++] = 1;
        return true;
    }

    bool PublishFlag(uint32_t task_id) {
        ++flag_calls;
        if (!allow_flag) return false;
        flag_task = task_id;
        events[event_count++] = 2;
        return true;
    }
};

struct EngineCompletion {
    bool complete = false;

    bool IsComplete(const ExecutionToken &) const {
        return complete;
    }
};

ExecPayloadSpec MakeSpec(
    uint32_t task_id = 42,
    ExecEngineClass engine_class = ExecEngineClass::Aiv
) {
    return ExecPayloadSpec{
        task_id,
        0xABCDEF0012345678ULL,
        0x1020304050607080ULL,
        7,
        3,
        4,
        4,
        engine_class,
        0,
        0,
        0,
        1,
    };
}

SyntheticPayloadSource MakeSource(bool pause_mid_pack = false) {
    SyntheticPayloadSource source{};
    for (uint32_t tensor = 0; tensor < source.tensors.size();
         ++tensor) {
        for (uint32_t word = 0;
             word < source.tensors[tensor].size(); ++word) {
            source.tensors[tensor][word] =
                0xA000000000000000ULL |
                (static_cast<uint64_t>(tensor) << 16U) | word;
        }
    }
    source.scalars = {
        0xB000000000000001ULL,
        0xB000000000000002ULL,
        0xB000000000000003ULL,
        0xB000000000000004ULL,
    };
    source.fanin = {1, 7, 17, 41};
    source.pause_mid_pack = pause_mid_pack;
    return source;
}

void InitializeCell(SharedExecCell &cell) {
    std::memset(&cell, 0, sizeof(cell));
}

void InitializeToken(ExecutionToken &token) {
    std::memset(&token, 0xCD, sizeof(token));
    ResetExecutionToken(token);
}

void InitializeFatal(SharedExecFatalControl &fatal) {
    std::memset(&fatal, 0, sizeof(fatal));
    ProtocolTestOps::SetFatalAddress(&fatal);
}

DecodedExecState CurrentState(const SharedExecCell &cell) {
    return DecodeExecState(__atomic_load_n(
        &cell.control.state, __ATOMIC_ACQUIRE
    ));
}

void CheckPayloadMatches(
    const ExecutionToken &token,
    const ExecPayloadSpec &spec,
    const SyntheticPayloadSource &source
) {
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    Check(header.task_id == spec.task_id, "task id survives binding");
    Check(
        header.function_address == spec.function_address,
        "function address survives binding"
    );
    Check(
        header.completion_vend == spec.completion_vend,
        "completion vend is frozen in the portable payload"
    );
    Check(
        header.tensor_count == spec.tensor_count &&
            header.scalar_count == spec.scalar_count &&
            header.fanin_count == spec.fanin_count,
        "active counts survive binding"
    );
    for (uint32_t tensor = 0; tensor < spec.tensor_count; ++tensor) {
        for (uint32_t word = 0; word < kExecTensorDescWords; ++word) {
            uint64_t value = 0;
            Check(
                ExecutionTokenTensorWord(
                    token, tensor, word, value
                ) && value == source.tensors[tensor][word],
                "TensorDesc word survives binding"
            );
        }
    }
    for (uint32_t scalar = 0; scalar < spec.scalar_count; ++scalar) {
        uint64_t value = 0;
        Check(
            ExecutionTokenScalar(token, scalar, value) &&
                value == source.scalars[scalar],
            "scalar survives binding"
        );
    }
    for (uint32_t edge = 0; edge < spec.fanin_count; ++edge) {
        int32_t producer = -1;
        Check(
            ExecutionTokenFanin(token, edge, producer) &&
                producer == source.fanin[edge],
            "fanin survives binding"
        );
    }
}

void TestLayoutAndStateAbi() {
    Check(sizeof(SharedExecControl) == 64, "control owns 64 bytes");
    Check(
        sizeof(SharedExecFatalControl) == 64,
        "global fatal owns one atomic-only cache line"
    );
    Check(
        offsetof(SharedExecCell, payload) == 64,
        "payload starts after the atomic-only line"
    );
    Check(
        sizeof(SharedExecCell) == 4416,
        "task-indexed cell ABI is stable"
    );
    Check(
        sizeof(ExecutionDispatchBinding) == 512 &&
            alignof(ExecutionDispatchBinding) == 64 &&
            offsetof(ExecutionDispatchBinding, args) == 0 &&
            offsetof(ExecutionDispatchBinding, local_context) == 400 &&
            offsetof(ExecutionDispatchBinding, global_context) == 448,
        "dispatch binding keeps 50 args and both context regions"
    );
    Check(
        offsetof(ExecutionToken, dispatch) == 4416 &&
            sizeof(ExecutionToken) == 4928,
        "token control, payload and dispatch binding use disjoint lines"
    );

    ExecPayloadLayout empty{};
    Check(
        ComputeExecPayloadLayout(0, 0, 0, empty) &&
            empty.payload_bytes == 64 &&
            empty.payload_lines == 1 &&
            empty.written_words == 8,
        "empty active payload is one header line"
    );
    ExecPayloadLayout maximum{};
    Check(
        ComputeExecPayloadLayout(32, 16, 16, maximum) &&
            maximum.payload_bytes == 4352 &&
            maximum.payload_lines == 68 &&
            maximum.written_words == 544,
        "maximum payload fills exactly 68 lines"
    );
    ExecPayloadLayout overflow{};
    Check(
        !ComputeExecPayloadLayout(33, 0, 0, overflow),
        "tensor overflow is rejected"
    );

    const int64_t raw = static_cast<int64_t>(EncodeExecState(
        ExecPhase::Claimed, 37, 19,
        ExecEngineClass::Aic, 68, 901
    ));
    const DecodedExecState decoded = DecodeExecState(raw);
    Check(
        decoded.valid && decoded.phase == ExecPhase::Claimed &&
            decoded.build_owner == 37 &&
            decoded.execute_owner == 19 &&
            decoded.engine_class == ExecEngineClass::Aic &&
            decoded.payload_lines == 68 &&
            decoded.task_id == 901,
        "packed state round trip is exact"
    );
    Check(
        !DecodeExecState(raw | (1LL << 61)).valid,
        "reserved state bits fail closed"
    );
    const DecodedExecState built = DecodeExecState(
        static_cast<int64_t>(EncodeExecState(
            ExecPhase::Built, 37, kExecUnboundOwner,
            ExecEngineClass::Aiv, 1, 902
        ))
    );
    Check(
        built.valid && built.build_owner == 37 &&
            built.execute_owner == kExecUnboundOwner,
        "BUILT carries an explicit unbound execute owner"
    );
    Check(
        !DecodeExecState(static_cast<int64_t>(EncodeExecState(
            ExecPhase::Built, 37, 19,
            ExecEngineClass::Aiv, 1, 902
        ))).valid,
        "BUILT rejects a prematurely bound execute owner"
    );
    const int64_t fatal_raw = static_cast<int64_t>(EncodeExecFatal(
        ExecFatalReason::ClaimedPayloadInvalid, 19, 901
    ));
    const DecodedExecFatal fatal = DecodeExecFatal(fatal_raw);
    Check(
        fatal.valid &&
            fatal.reason ==
                ExecFatalReason::ClaimedPayloadInvalid &&
            fatal.reporter_owner == 19 && fatal.task_id == 901,
        "packed fatal record round trip is exact"
    );
    Check(
        !DecodeExecFatal(fatal_raw | (1LL << 60)).valid,
        "fatal reserved bits fail closed"
    );
}

void TestPublishBindAndComplete() {
    SharedExecCell cell{};
    ExecutionToken token{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeToken(token);
    InitializeFatal(fatal);
    ProtocolTestOps::ResetTrace();
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    ExecPayloadLayout layout{};
    Check(
        ValidateExecPayloadSpec(spec, layout),
        "reference payload spec is valid"
    );

    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 3, spec, source, fatal
        ) == ExecBuildResult::Published,
        "builder publishes one payload"
    );
    const DecodedExecState built = CurrentState(cell);
    Check(
        built.valid && built.phase == ExecPhase::Built &&
            built.build_owner == 3 &&
            built.execute_owner == kExecUnboundOwner &&
            built.payload_lines == layout.payload_lines,
        "BUILT publishes build owner, unbound executor and line count"
    );
    Check(
        ProtocolTestOps::payload_stores.load() ==
            layout.written_words,
        "builder performs one forward write of every valid word"
    );
    Check(
        ProtocolTestOps::flush_calls.load() == 1 &&
            ProtocolTestOps::flush_address.load() ==
                reinterpret_cast<uintptr_t>(&cell.payload) &&
            ProtocolTestOps::flush_bytes.load() ==
                static_cast<uint64_t>(layout.payload_lines) * 64,
        "builder flushes the complete payload exactly once"
    );
    Check(
        ProtocolTestOps::preload_build_calls.load() == 1 &&
            ProtocolTestOps::preload_build_address.load() ==
                reinterpret_cast<uintptr_t>(&cell.payload) &&
            ProtocolTestOps::preload_build_bytes.load() ==
                static_cast<uint64_t>(layout.payload_lines) * 64,
        "builder preload hook is bounded to the active payload"
    );
    Check(
        ProtocolTestOps::last_payload_store_event.load() <
                ProtocolTestOps::flush_event.load() &&
            ProtocolTestOps::flush_event.load() <
                ProtocolTestOps::built_cas_event.load(),
        "payload stores precede one flush and explicit BUILT publish"
    );

    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            cell, spec.task_id, 9, ExecEngineClass::Aiv,
            token, fatal
        ) == ExecClaimResult::Claimed,
        "compatible executor claims and binds"
    );
    Check(
        token.control.phase == ExecTokenPhase::WaitingFanin &&
            token.control.task_id == spec.task_id &&
            token.control.build_owner == 3 &&
            token.control.execute_owner == 9,
        "binding preserves both owners in the executor's sole token"
    );
    const DecodedExecState claimed = CurrentState(cell);
    Check(
        claimed.valid && claimed.phase == ExecPhase::Claimed &&
            claimed.build_owner == 3 &&
            claimed.execute_owner == 9,
        "CLAIMED binds execute owner without discarding build owner"
    );
    Check(
        ProtocolTestOps::invalidate_calls.load() == 1 &&
            ProtocolTestOps::invalidate_address.load() ==
                reinterpret_cast<uintptr_t>(&cell.payload) &&
            ProtocolTestOps::invalidate_bytes.load() ==
                static_cast<uint64_t>(layout.payload_lines) * 64,
        "winner invalidates exactly the published payload once"
    );
    Check(
        ProtocolTestOps::payload_loads.load() ==
            layout.payload_lines * kExecHeaderWords &&
            ProtocolTestOps::token_stores.load() ==
                layout.payload_lines * kExecHeaderWords,
        "binding makes one forward copy of the published lines"
    );
    Check(
        ProtocolTestOps::preload_source_calls.load() == 1 &&
            ProtocolTestOps::preload_token_calls.load() == 1 &&
            ProtocolTestOps::preload_source_address.load() ==
                reinterpret_cast<uintptr_t>(&cell.payload) &&
            ProtocolTestOps::preload_token_address.load() ==
                reinterpret_cast<uintptr_t>(&token.payload) &&
            ProtocolTestOps::preload_source_bytes.load() ==
                static_cast<uint64_t>(layout.payload_lines) * 64 &&
            ProtocolTestOps::preload_token_bytes.load() ==
                static_cast<uint64_t>(layout.payload_lines) * 64,
        "executor preload hooks share the exact acquire boundary"
    );
    Check(
        ProtocolTestOps::claim_cas_event.load() <
                ProtocolTestOps::invalidate_event.load() &&
            ProtocolTestOps::invalidate_event.load() <
                ProtocolTestOps::first_payload_load_event.load() &&
            ProtocolTestOps::first_payload_load_event.load() <
                ProtocolTestOps::first_token_store_event.load(),
        "claim CAS precedes invalidate, shared load and token store"
    );

    const uint64_t shared_reads_after_binding =
        ProtocolTestOps::payload_loads.load();
    CheckPayloadMatches(token, spec, source);
    PA_GM const uint64_t *dispatch_args =
        ExecutionTokenDispatchArgs(token);
    bool dispatch_tensors_match = true;
    for (uint32_t tensor = 0;
         tensor < spec.tensor_count; ++tensor) {
        dispatch_tensors_match = dispatch_tensors_match &&
            dispatch_args[tensor] == static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(
                    &token.payload.words[
                        layout.tensor_word_offset +
                        tensor * kExecTensorDescWords
                    ]
                )
            );
    }
    bool dispatch_scalars_match = true;
    for (uint32_t scalar = 0;
         scalar < spec.scalar_count; ++scalar) {
        dispatch_scalars_match = dispatch_scalars_match &&
            dispatch_args[spec.tensor_count + scalar] ==
                source.scalars[scalar];
    }
    Check(
        dispatch_tensors_match && dispatch_scalars_match &&
            dispatch_args[kExecDispatchLocalContextIndex] ==
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                    &token.dispatch.local_context[0]
                )) &&
            dispatch_args[kExecDispatchGlobalContextIndex] ==
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                    &token.dispatch.global_context[0]
                )),
        "executor rebuilds tensor, scalar and context dispatch args"
    );
    ReadyTable ready{};
    for (int32_t producer : source.fanin) {
        ready.ready[static_cast<uint32_t>(producer)].store(true);
    }
    Check(
        ExecutionTokenFaninReady(token, ready),
        "local fanin is ready after all producer flags"
    );
    Check(
        ProtocolTestOps::payload_loads.load() ==
            shared_reads_after_binding,
        "token access never rereads the shared payload"
    );
    Check(
        TryMarkExecutionTokenEngineInflight<ProtocolTestOps>(
            token, ready, fatal
        ),
        "ready helper enters engine-inflight"
    );
    EngineCompletion engine_completion{};
    Check(
        !TryMarkExecutionTokenCompleting<ProtocolTestOps>(
            token, engine_completion, fatal
        ),
        "incomplete engine cannot enter completion"
    );
    engine_completion.complete = true;
    Check(
        TryMarkExecutionTokenCompleting<ProtocolTestOps>(
            token, engine_completion, fatal
        ),
        "engine completion enters completing"
    );
    CompletionRecorder completion{};
    const uint64_t unrelated_executor_heap_cursor =
        spec.completion_vend ^ 0x00FF00FF00FF00FFULL;
    Check(
        PublishExecDoneAfterCompletion<ProtocolTestOps>(
            cell, token, completion, fatal
        ) == ExecDoneResult::Done,
        "completion publishes DONE before releasing the token"
    );
    Check(
        completion.event_count == 2 &&
            completion.events[0] == 1 &&
            completion.events[1] == 2 &&
            completion.vend_task == spec.task_id &&
            completion.flag_task == spec.task_id &&
            completion.vend == spec.completion_vend &&
            completion.vend != unrelated_executor_heap_cursor,
        "completion consumes payload vend before publishing the flag"
    );
    const DecodedExecState done = CurrentState(cell);
    Check(
        done.valid && done.phase == ExecPhase::Done &&
            done.build_owner == 3 && done.execute_owner == 9,
        "DONE retains both build and execute owners"
    );
    Check(
        token.control.phase == ExecTokenPhase::Idle &&
            token.control.build_owner == UINT32_MAX &&
            token.control.execute_owner == UINT32_MAX &&
            token.control.fanin_ready_prefix == 0,
        "token returns to a fully reset IDLE state only after DONE"
    );
}

void TestHalfBuiltPayloadIsInvisible() {
    SharedExecCell cell{};
    ExecutionToken token{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeToken(token);
    InitializeFatal(fatal);
    ProtocolTestOps::ResetTrace();
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource(true);
    g_mid_pack_pause.Enable();
    ExecBuildResult build_result = ExecBuildResult::InvalidInput;
    std::thread builder([&] {
        build_result = BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 2, spec, source, fatal
        );
    });
    g_mid_pack_pause.WaitUntilReached();
    Check(
        CurrentState(cell).phase == ExecPhase::Building,
        "half-packed payload remains BUILDING"
    );
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            cell, spec.task_id, 4, ExecEngineClass::Aiv,
            token, fatal
        ) == ExecClaimResult::NotBuilt,
        "executor rejects a half-packed payload"
    );
    Check(
        ProtocolTestOps::invalidate_calls.load() == 0 &&
            ProtocolTestOps::payload_loads.load() == 0,
        "half-packed rejection performs no payload access"
    );
    g_mid_pack_pause.Release();
    builder.join();
    g_mid_pack_pause.Disable();
    Check(
        build_result == ExecBuildResult::Published,
        "paused builder resumes and publishes"
    );
}

void TestFlushedPayloadIsInvisibleBeforeBuilt() {
    SharedExecCell cell{};
    ExecutionToken token{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeToken(token);
    InitializeFatal(fatal);
    ProtocolTestOps::ResetTrace();
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    g_before_built_pause.Enable();
    ExecBuildResult build_result = ExecBuildResult::InvalidInput;
    std::thread builder([&] {
        build_result = BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 5, spec, source, fatal
        );
    });
    g_before_built_pause.WaitUntilReached();
    Check(
        ProtocolTestOps::flush_calls.load() == 1 &&
            CurrentState(cell).phase == ExecPhase::Building,
        "flush alone does not publish BUILT"
    );
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            cell, spec.task_id, 6, ExecEngineClass::Aiv,
            token, fatal
        ) == ExecClaimResult::NotBuilt,
        "executor waits for the explicit BUILT boundary"
    );
    Check(
        ProtocolTestOps::invalidate_calls.load() == 0 &&
            ProtocolTestOps::payload_loads.load() == 0,
        "pre-BUILT rejection does not touch payload"
    );
    g_before_built_pause.Release();
    builder.join();
    g_before_built_pause.Disable();
    Check(
        build_result == ExecBuildResult::Published,
        "flushed builder eventually publishes BUILT"
    );
}

void TestConcurrentExecutorsHaveOneWinner() {
    constexpr uint32_t kExecutorCount = 16;
    SharedExecCell cell{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeFatal(fatal);
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 1, spec, source, fatal
        ) == ExecBuildResult::Published,
        "multi-executor cell is published"
    );
    ExecPayloadLayout layout{};
    Check(
        ValidateExecPayloadSpec(spec, layout),
        "multi-executor layout is valid"
    );
    ProtocolTestOps::ResetTrace();

    std::array<ExecutionToken, kExecutorCount> tokens{};
    std::array<ExecClaimResult, kExecutorCount> results{};
    for (ExecutionToken &token : tokens) {
        InitializeToken(token);
    }
    std::atomic<bool> start{false};
    std::vector<std::thread> executors;
    executors.reserve(kExecutorCount);
    for (uint32_t index = 0; index < kExecutorCount; ++index) {
        executors.emplace_back([&, index] {
            ProtocolTestOps::SetTraceActor(index);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[index] =
                ClaimAndBindExecPayload<ProtocolTestOps>(
                    cell, spec.task_id, index,
                    ExecEngineClass::Aiv, tokens[index], fatal
                );
            ProtocolTestOps::ClearTraceActor();
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread &executor : executors) {
        executor.join();
    }

    uint32_t winners = 0;
    uint32_t winner_index = UINT32_MAX;
    for (uint32_t index = 0; index < kExecutorCount; ++index) {
        if (results[index] == ExecClaimResult::Claimed) {
            ++winners;
            winner_index = index;
        }
    }
    Check(winners == 1, "concurrent executors elect exactly one owner");
    Check(
        ProtocolTestOps::invalidate_calls.load() == 1 &&
            ProtocolTestOps::payload_loads.load() ==
                layout.payload_lines * kExecHeaderWords,
        "only the CAS winner invalidates and copies payload"
    );
    for (uint32_t index = 0; index < kExecutorCount; ++index) {
        if (index == winner_index) {
            Check(
                tokens[index].control.phase ==
                    ExecTokenPhase::WaitingFanin,
                "winner owns one active token"
            );
            Check(
                ProtocolTestOps::actor_invalidate_calls[index].load() ==
                        1 &&
                    ProtocolTestOps::actor_payload_loads[index].load() ==
                        layout.payload_lines * kExecHeaderWords,
                "winner alone performs acquire and payload reads"
            );
        } else {
            Check(
                tokens[index].control.phase == ExecTokenPhase::Idle,
                "loser token remains IDLE"
            );
            Check(
                ProtocolTestOps::actor_invalidate_calls[index].load() ==
                        0 &&
                    ProtocolTestOps::actor_payload_loads[index].load() ==
                        0,
                "every losing actor performs zero DCCI and payload read"
            );
        }
    }
}

void TestConcurrentBuildersPublishExactlyOnce() {
    constexpr uint32_t kBuilderCount = 8;
    SharedExecCell cell{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeFatal(fatal);
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    ExecPayloadLayout layout{};
    Check(
        ValidateExecPayloadSpec(spec, layout),
        "multi-builder layout is valid"
    );
    ProtocolTestOps::ResetTrace();
    std::array<ExecBuildResult, kBuilderCount> results{};
    std::atomic<bool> start{false};
    std::vector<std::thread> builders;
    builders.reserve(kBuilderCount);
    for (uint32_t index = 0; index < kBuilderCount; ++index) {
        builders.emplace_back([&, index] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            results[index] =
                BuildAndPublishExecPayload<ProtocolTestOps>(
                    cell, index, spec, source, fatal
                );
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread &builder : builders) {
        builder.join();
    }
    uint32_t published = 0;
    uint32_t unavailable = 0;
    for (ExecBuildResult result : results) {
        published += result == ExecBuildResult::Published ? 1U : 0U;
        unavailable +=
            result == ExecBuildResult::CellUnavailable ? 1U : 0U;
    }
    Check(
        published == 1 && unavailable == kBuilderCount - 1,
        "concurrent builders publish exactly one payload"
    );
    Check(
        ProtocolTestOps::payload_stores.load() ==
                layout.written_words &&
            ProtocolTestOps::flush_calls.load() == 1 &&
            CurrentState(cell).phase == ExecPhase::Built,
        "losing builders perform no payload store or DCCI"
    );
}

void TestBusyExecutorDoesNotTouchCell() {
    SharedExecCell first{};
    SharedExecCell second{};
    ExecutionToken busy{};
    ExecutionToken idle{};
    SharedExecFatalControl fatal{};
    InitializeCell(first);
    InitializeCell(second);
    InitializeToken(busy);
    InitializeToken(idle);
    InitializeFatal(fatal);
    const SyntheticPayloadSource source = MakeSource();
    const ExecPayloadSpec first_spec = MakeSpec(42);
    const ExecPayloadSpec second_spec = MakeSpec(43);
    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            first, 1, first_spec, source, fatal
        ) == ExecBuildResult::Published &&
        BuildAndPublishExecPayload<ProtocolTestOps>(
            second, 2, second_spec, source, fatal
        ) == ExecBuildResult::Published,
        "two cells are available to executors"
    );
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            first, first_spec.task_id, 3,
            ExecEngineClass::Aiv, busy, fatal
        ) == ExecClaimResult::Claimed,
        "first task occupies the executor token"
    );

    ProtocolTestOps::ResetTrace();
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            second, second_spec.task_id, 3,
            ExecEngineClass::Aiv, busy, fatal
        ) == ExecClaimResult::TokenBusy,
        "busy executor refuses a second task"
    );
    Check(
        ProtocolTestOps::load_calls.load() == 0 &&
            ProtocolTestOps::cas_calls.load() == 0 &&
            ProtocolTestOps::invalidate_calls.load() == 0,
        "busy executor emits no shared operation"
    );
    Check(
        CurrentState(second).phase == ExecPhase::Built,
        "unclaimed task remains in the global backlog"
    );
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            second, second_spec.task_id, 4,
            ExecEngineClass::Aiv, idle, fatal
        ) == ExecClaimResult::Claimed,
        "another idle executor can claim the backlog task"
    );
}

void TestTaskIdentityGuardsClaimAndCompletion() {
    SharedExecCell first{};
    SharedExecCell second{};
    ExecutionToken first_token{};
    ExecutionToken second_token{};
    SharedExecFatalControl fatal{};
    InitializeCell(first);
    InitializeCell(second);
    InitializeToken(first_token);
    InitializeToken(second_token);
    InitializeFatal(fatal);
    const SyntheticPayloadSource source = MakeSource();
    const ExecPayloadSpec first_spec = MakeSpec(42);
    const ExecPayloadSpec second_spec = MakeSpec(43);
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            first, 1, first_spec, source, fatal
        ) == ExecBuildResult::Published &&
        BuildAndPublishExecPayload<ProtocolTestOps>(
            second, 2, second_spec, source, fatal
        ) == ExecBuildResult::Published,
        "task-identity cells are published"
    );

    ProtocolTestOps::ResetTrace();
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            first, second_spec.task_id, 7,
            ExecEngineClass::Aiv, first_token, fatal
        ) == ExecClaimResult::InvalidControl,
        "claim rejects a cell whose packed task id differs"
    );
    Check(
        ProtocolTestOps::claim_cas_event.load() == 0 &&
            ProtocolTestOps::fatal_cas_calls.load() == 1 &&
            ProtocolTestOps::invalidate_calls.load() == 0 &&
            ProtocolTestOps::payload_loads.load() == 0 &&
            CurrentState(first).phase == ExecPhase::Built,
        "wrong-task claim fails before CAS, DCCI and payload copy"
    );

    // fatal 在一次设备运行中永不清零；下面是独立的防御性场景，
    // 因此显式初始化一轮新的 synthetic runtime。
    InitializeFatal(fatal);

    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            first, first_spec.task_id, 7,
            ExecEngineClass::Aiv, first_token, fatal
        ) == ExecClaimResult::Claimed &&
        ClaimAndBindExecPayload<ProtocolTestOps>(
            second, second_spec.task_id, 7,
            ExecEngineClass::Aiv, second_token, fatal
        ) == ExecClaimResult::Claimed,
        "defensive cross-cell test binds two same-owner tokens"
    );
    ReadyTable ready{};
    for (int32_t producer : source.fanin) {
        ready.ready[static_cast<uint32_t>(producer)].store(true);
    }
    EngineCompletion engine_completion{true};
    Check(
        TryMarkExecutionTokenEngineInflight<ProtocolTestOps>(
            first_token, ready, fatal
        ) &&
            TryMarkExecutionTokenCompleting<ProtocolTestOps>(
                first_token, engine_completion, fatal
            ),
        "first token reaches completion"
    );
    CompletionRecorder completion{};
    Check(
        PublishExecDoneAfterCompletion<ProtocolTestOps>(
            second, first_token, completion, fatal
        ) == ExecDoneResult::StateConflict,
        "a token cannot mark another task cell DONE"
    );
    Check(
        CurrentState(first).phase == ExecPhase::Claimed &&
            CurrentState(second).phase == ExecPhase::Claimed &&
            first_token.control.phase ==
                ExecTokenPhase::CompletionPublished,
        "cross-cell conflict preserves both claimed cells"
    );
    Check(
        completion.vend_calls == 1 &&
            completion.flag_calls == 1 &&
            completion.event_count == 2 &&
            ExecFatalPublished<ProtocolTestOps>(fatal),
        "cross-cell mismatch publishes terminal fatal without duplicate side effect"
    );
}

void TestClaimedPayloadCannotBeRebuilt() {
    SharedExecCell cell{};
    ExecutionToken token{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeToken(token);
    InitializeFatal(fatal);
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 1, spec, source, fatal
        ) == ExecBuildResult::Published &&
        ClaimAndBindExecPayload<ProtocolTestOps>(
            cell, spec.task_id, 2, ExecEngineClass::Aiv,
            token, fatal
        ) == ExecClaimResult::Claimed,
        "cell reaches CLAIMED before duplicate build"
    );
    std::array<uint64_t, kExecMaxPayloadWords> snapshot{};
    for (uint32_t word = 0; word < snapshot.size(); ++word) {
        snapshot[word] = cell.payload.words[word];
    }

    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 7, spec, source, fatal
        ) == ExecBuildResult::CellUnavailable,
        "duplicate builder cannot reopen a CLAIMED cell"
    );
    Check(
        ProtocolTestOps::payload_stores.load() == 0 &&
            ProtocolTestOps::flush_calls.load() == 0,
        "duplicate builder performs no ordinary write or DCCI"
    );
    bool unchanged = true;
    for (uint32_t word = 0; word < snapshot.size(); ++word) {
        unchanged = unchanged &&
            snapshot[word] == cell.payload.words[word];
    }
    Check(unchanged, "CLAIMED payload remains immutable");
}

void TestInvalidControlAndPayloadFailClosed() {
    SharedExecCell invalid_control{};
    ExecutionToken token{};
    SharedExecFatalControl control_fatal{};
    InitializeCell(invalid_control);
    InitializeToken(token);
    InitializeFatal(control_fatal);
    invalid_control.control.state = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Built, 1, kExecUnboundOwner,
            ExecEngineClass::Aiv,
            kExecMaxPayloadLines + 1, 42
        )
    );
    ProtocolTestOps::ResetTrace();
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            invalid_control, 42, 2, ExecEngineClass::Aiv,
            token, control_fatal
        ) == ExecClaimResult::InvalidControl,
        "invalid atomic payload range fails before DCCI"
    );
    Check(
        ProtocolTestOps::invalidate_calls.load() == 0 &&
            ProtocolTestOps::payload_loads.load() == 0 &&
            DecodeExecFatal(control_fatal.state).valid &&
            DecodeExecFatal(control_fatal.state).reason ==
                ExecFatalReason::InvalidBuiltControl &&
            token.control.phase == ExecTokenPhase::Idle,
        "invalid control cannot trigger DCCI or copy"
    );

    SharedExecCell corrupt_payload{};
    SharedExecFatalControl payload_fatal{};
    InitializeCell(corrupt_payload);
    InitializeToken(token);
    InitializeFatal(payload_fatal);
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            corrupt_payload, 1, spec, source, payload_fatal
        ) == ExecBuildResult::Published,
        "payload is published before corruption injection"
    );
    const uint64_t word3 = corrupt_payload.payload.words[3];
    corrupt_payload.payload.words[3] =
        static_cast<uint32_t>(word3) |
        (static_cast<uint64_t>(UINT32_MAX) << 32U);
    ProtocolTestOps::ResetTrace();
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            corrupt_payload, spec.task_id, 2,
            ExecEngineClass::Aiv, token, payload_fatal
        ) == ExecClaimResult::InvalidPayload,
        "corrupt header fails after bounded binding"
    );
    Check(
        ProtocolTestOps::invalidate_calls.load() == 1 &&
            token.control.phase == ExecTokenPhase::Faulted &&
            CurrentState(corrupt_payload).phase ==
                ExecPhase::Claimed &&
            DecodeExecFatal(payload_fatal.state).valid &&
            DecodeExecFatal(payload_fatal.state).reason ==
                ExecFatalReason::ClaimedPayloadInvalid,
        "invalid payload remains claimed and fail-closed"
    );
    CompletionRecorder completion{};
    Check(
        PublishExecDoneAfterCompletion<ProtocolTestOps>(
            corrupt_payload, token, completion, payload_fatal
        ) == ExecDoneResult::TokenNotCompleting &&
            completion.event_count == 0,
        "invalid payload cannot enter completion or publish vend"
    );
}

void TestCorruptCountsRemainBounded() {
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    ExecPayloadLayout published_layout{};
    Check(
        ValidateExecPayloadSpec(spec, published_layout),
        "corrupt-count reference layout is valid"
    );
    for (uint32_t field = 0; field < 3; ++field) {
        SharedExecCell cell{};
        ExecutionToken token{};
        SharedExecFatalControl fatal{};
        InitializeCell(cell);
        InitializeToken(token);
        InitializeFatal(fatal);
        Check(
            BuildAndPublishExecPayload<ProtocolTestOps>(
                cell, 1, spec, source, fatal
            ) == ExecBuildResult::Published,
            "corrupt-count payload is initially valid"
        );
        const uint32_t shift = field * 16U;
        const uint32_t invalid_count =
            field == 0 ? kExecMaxTensors + 1U
                       : field == 1 ? kExecMaxScalars + 1U
                                    : kExecMaxFanin + 1U;
        const uint64_t field_mask = 0xFFFFULL << shift;
        cell.payload.words[4] =
            (cell.payload.words[4] & ~field_mask) |
            (static_cast<uint64_t>(invalid_count) << shift);
        ProtocolTestOps::ResetTrace();
        Check(
            ClaimAndBindExecPayload<ProtocolTestOps>(
                cell, spec.task_id, 2,
                ExecEngineClass::Aiv, token, fatal
            ) == ExecClaimResult::InvalidPayload,
            "corrupt tensor/scalar/fanin count fails closed"
        );
        Check(
            ProtocolTestOps::invalidate_calls.load() == 1 &&
                ProtocolTestOps::invalidate_bytes.load() ==
                    static_cast<uint64_t>(
                        published_layout.payload_lines
                    ) * kExecCacheLineBytes &&
                ProtocolTestOps::payload_loads.load() ==
                    published_layout.payload_lines *
                        kExecHeaderWords &&
                token.control.phase == ExecTokenPhase::Faulted,
            "corrupt count cannot expand bounded DCCI or copy"
        );
        Check(
            ExecFatalPublished<ProtocolTestOps>(fatal),
            "corrupt count publishes terminal fatal"
        );
    }

    SharedExecCell reserved_cell{};
    ExecutionToken reserved_token{};
    SharedExecFatalControl reserved_fatal{};
    InitializeCell(reserved_cell);
    InitializeToken(reserved_token);
    InitializeFatal(reserved_fatal);
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            reserved_cell, 1, spec, source, reserved_fatal
        ) == ExecBuildResult::Published,
        "reserved-word payload is initially valid"
    );
    reserved_cell.payload.words[0] |= 1ULL << 32U;
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            reserved_cell, spec.task_id, 2,
            ExecEngineClass::Aiv, reserved_token,
            reserved_fatal
        ) == ExecClaimResult::InvalidPayload &&
            reserved_token.control.phase ==
                ExecTokenPhase::Faulted &&
            DecodeExecFatal(reserved_fatal.state).reason ==
                ExecFatalReason::ClaimedPayloadInvalid,
        "word0 reserved identity bits are forced to zero"
    );
}

void TestFaninGatesExecutionAndRejectsFutureProducer() {
    SharedExecCell cell{};
    ExecutionToken token{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeToken(token);
    InitializeFatal(fatal);
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 1, spec, source, fatal
        ) == ExecBuildResult::Published &&
        ClaimAndBindExecPayload<ProtocolTestOps>(
            cell, spec.task_id, 2, ExecEngineClass::Aiv,
            token, fatal
        ) == ExecClaimResult::Claimed,
        "fanin test binds one task"
    );
    const uint64_t shared_reads_after_binding =
        ProtocolTestOps::payload_loads.load();
    ReadyTable ready{};
    for (int32_t producer : source.fanin) {
        ready.ready[static_cast<uint32_t>(producer)].store(true);
    }
    ready.ready[17].store(false);
    Check(
        !ExecutionTokenFaninReady(token, ready),
        "one unfinished producer blocks execution"
    );
    Check(
        token.control.fanin_ready_prefix == 2 &&
            ready.ReadCount(source.fanin[0]) == 1 &&
            ready.ReadCount(source.fanin[1]) == 1 &&
            ready.ReadCount(source.fanin[2]) == 1 &&
            ready.ReadCount(source.fanin[3]) == 0,
        "fanin polling advances each successful ready prefix item"
    );
    Check(
        !TryMarkExecutionTokenEngineInflight<ProtocolTestOps>(
            token, ready, fatal
        ),
        "unfinished producer cannot be bypassed by state transition"
    );
    Check(
        token.control.phase == ExecTokenPhase::WaitingFanin &&
            token.control.fanin_ready_prefix == 2 &&
            ready.ReadCount(source.fanin[0]) == 1 &&
            ready.ReadCount(source.fanin[1]) == 1 &&
            ready.ReadCount(source.fanin[2]) == 2 &&
            ready.ReadCount(source.fanin[3]) == 0,
        "blocked poll never rereads the already-ready prefix"
    );
    ready.ready[17].store(true, std::memory_order_release);
    Check(
        ExecutionTokenFaninReady(token, ready),
        "task becomes ready after its final producer"
    );
    const uint32_t first_reads = ready.ReadCount(source.fanin[0]);
    const uint32_t second_reads = ready.ReadCount(source.fanin[1]);
    const uint32_t third_reads = ready.ReadCount(source.fanin[2]);
    const uint32_t fourth_reads = ready.ReadCount(source.fanin[3]);
    Check(
        TryMarkExecutionTokenEngineInflight<ProtocolTestOps>(
            token, ready, fatal
        ),
        "ready task advances through the combined gate"
    );
    Check(
        token.control.fanin_ready_prefix == spec.fanin_count &&
            ready.ReadCount(source.fanin[0]) == first_reads &&
            ready.ReadCount(source.fanin[1]) == second_reads &&
            ready.ReadCount(source.fanin[2]) == third_reads &&
            ready.ReadCount(source.fanin[3]) == fourth_reads,
        "fully-ready prefix is not reread at engine transition"
    );
    Check(
        ProtocolTestOps::payload_loads.load() ==
            shared_reads_after_binding,
        "fanin polling only uses the private token"
    );

    // 把最后一个 producer 改成当前 task，验证 Build N 的过滤边界。
    ExecPayloadHeader header = ExecutionTokenHeader(token);
    ExecPayloadLayout layout{};
    Check(
        ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count, layout
        ),
        "fanin layout can be reconstructed"
    );
    const uint32_t final_word =
        layout.fanin_word_offset +
        (header.fanin_count - 1U) / 2U;
    const uint64_t original = token.payload.words[final_word];
    token.control.phase = ExecTokenPhase::WaitingFanin;
    token.control.fanin_ready_prefix = 0;
    token.payload.words[final_word] =
        (original & 0xFFFFFFFFULL) |
        (static_cast<uint64_t>(spec.task_id) << 32U);
    ready.ready[spec.task_id].store(true);
    Check(
        !ExecutionTokenFaninReady(token, ready),
        "producer equal to the current task is rejected"
    );
}

void TestBuildRejectsInvalidFaninBeforePublication() {
    const ExecPayloadSpec spec = MakeSpec();
    for (uint32_t invalid_case = 0; invalid_case < 3;
        ++invalid_case) {
        SharedExecCell cell{};
        SharedExecFatalControl fatal{};
        InitializeCell(cell);
        InitializeFatal(fatal);
        SyntheticPayloadSource source = MakeSource();
        if (invalid_case == 0) {
            source.fanin[1] = -1;
        } else if (invalid_case == 1) {
            source.fanin[1] = static_cast<int32_t>(spec.task_id);
        } else {
            source.fanin[1] =
                static_cast<int32_t>(spec.task_id + 1U);
        }
        ProtocolTestOps::ResetTrace();
        Check(
            BuildAndPublishExecPayload<ProtocolTestOps>(
                cell, 1, spec, source, fatal
            ) == ExecBuildResult::InvalidInput,
            "builder rejects negative, self and future fanin"
        );
        Check(
            CurrentState(cell).phase != ExecPhase::Built &&
                ProtocolTestOps::flush_calls.load() == 0 &&
                DecodeExecFatal(fatal.state).valid &&
                DecodeExecFatal(fatal.state).reason ==
                    ExecFatalReason::BuildPackFailed,
            "invalid fanin never becomes externally consumable"
        );
    }
}

void TestBuildingFailurePublishesFatalAndCutsOffWork() {
    SharedExecCell broken{};
    SharedExecCell untouched{};
    ExecutionToken token{};
    SharedExecFatalControl fatal{};
    InitializeCell(broken);
    InitializeCell(untouched);
    InitializeToken(token);
    InitializeFatal(fatal);
    const ExecPayloadSpec spec = MakeSpec();
    SyntheticPayloadSource source = MakeSource();
    source.fanin[1] = -1;
    source.pause_invalid_fanin = true;
    g_invalid_fanin_pause.Enable();
    ProtocolTestOps::ResetTrace();
    ExecBuildResult result = ExecBuildResult::Published;
    std::thread builder([&] {
        result = BuildAndPublishExecPayload<ProtocolTestOps>(
            broken, 5, spec, source, fatal
        );
    });
    g_invalid_fanin_pause.WaitUntilReached();
    Check(
        CurrentState(broken).phase == ExecPhase::Building &&
            fatal.state == 0 &&
            ProtocolTestOps::payload_stores.load() > 0 &&
            ProtocolTestOps::flush_calls.load() == 0,
        "pack failure window preserves BUILDING before fatal"
    );
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            broken, spec.task_id, 6,
            ExecEngineClass::Aiv, token, fatal
        ) == ExecClaimResult::NotBuilt &&
            ProtocolTestOps::invalidate_calls.load() == 0,
        "executor cannot acquire a half-built payload"
    );
    g_invalid_fanin_pause.Release();
    builder.join();
    g_invalid_fanin_pause.Disable();
    const DecodedExecFatal decoded = DecodeExecFatal(fatal.state);
    Check(
        result == ExecBuildResult::InvalidInput && decoded.valid &&
            decoded.reason == ExecFatalReason::BuildPackFailed &&
            decoded.task_id == spec.task_id &&
            CurrentState(broken).phase == ExecPhase::Building &&
            ProtocolTestOps::flush_calls.load() == 0,
        "failed pack publishes first fatal without BUILT or DCCI"
    );

    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            untouched, 7, spec, source, fatal
        ) == ExecBuildResult::FatalObserved &&
        ClaimAndBindExecPayload<ProtocolTestOps>(
            broken, spec.task_id, 8,
            ExecEngineClass::Aiv, token, fatal
        ) == ExecClaimResult::FatalObserved,
        "fatal stops later builders and claimants"
    );
    Check(
        ProtocolTestOps::cas_calls.load() == 0 &&
            ProtocolTestOps::payload_stores.load() == 0 &&
            ProtocolTestOps::flush_calls.load() == 0 &&
            ProtocolTestOps::invalidate_calls.load() == 0 &&
            CurrentState(untouched).phase == ExecPhase::Empty,
        "post-fatal roles perform no irreversible shared operation"
    );
}

void TestFatalFirstFailureWins() {
    constexpr uint32_t kReporterCount = 8;
    SharedExecFatalControl fatal{};
    InitializeFatal(fatal);
    ProtocolTestOps::ResetTrace();
    std::array<bool, kReporterCount> won{};
    std::atomic<bool> start{false};
    std::vector<std::thread> reporters;
    reporters.reserve(kReporterCount);
    for (uint32_t reporter = 0;
         reporter < kReporterCount; ++reporter) {
        reporters.emplace_back([&, reporter] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            const ExecFatalReason reason = static_cast<ExecFatalReason>(
                1U + reporter
            );
            won[reporter] = PublishExecFatal<ProtocolTestOps>(
                fatal, reason, 100U + reporter, reporter
            );
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread &reporter : reporters) reporter.join();
    uint32_t winners = 0;
    uint32_t winner = UINT32_MAX;
    for (uint32_t reporter = 0;
         reporter < kReporterCount; ++reporter) {
        if (won[reporter]) {
            ++winners;
            winner = reporter;
        }
    }
    const int64_t first_raw = fatal.state;
    const DecodedExecFatal decoded = DecodeExecFatal(first_raw);
    Check(
        winners == 1 && decoded.valid &&
            decoded.reporter_owner == winner &&
            decoded.task_id == 100U + winner &&
            decoded.reason == static_cast<ExecFatalReason>(1U + winner),
        "concurrent fatal reporters preserve exactly one first record"
    );
    Check(
        !PublishExecFatal<ProtocolTestOps>(
            fatal, ExecFatalReason::InvalidBuildInput, 999, 9
        ) && fatal.state == first_raw,
        "fatal is monotonic and never overwrites the first failure"
    );
    fatal.state = static_cast<int64_t>(1ULL << 60U);
    Check(
        ExecFatalPublished<ProtocolTestOps>(fatal) &&
            !DecodeExecFatal(fatal.state).valid,
        "malformed nonzero fatal record still stops all roles"
    );
}

void TestFatalStopsWaitingAndInflightTokens() {
    SharedExecFatalControl fatal{};
    ExecutionToken waiting{};
    ExecutionToken inflight{};
    InitializeFatal(fatal);
    InitializeToken(waiting);
    InitializeToken(inflight);
    waiting.control.phase = ExecTokenPhase::WaitingFanin;
    inflight.control.phase = ExecTokenPhase::EngineInflight;
    Check(
        PublishExecFatal<ProtocolTestOps>(
            fatal, ExecFatalReason::ClaimedPayloadInvalid, 42, 3
        ),
        "fatal trigger is published for token convergence"
    );
    ReadyTable ready{};
    Check(
        !TryMarkExecutionTokenEngineInflight<ProtocolTestOps>(
            waiting, ready, fatal
        ) && waiting.control.phase == ExecTokenPhase::Faulted,
        "waiting token stops before engine launch"
    );
    EngineCompletion incomplete{false};
    Check(
        !TryMarkExecutionTokenCompleting<ProtocolTestOps>(
            inflight, incomplete, fatal
        ) && inflight.control.phase ==
                ExecTokenPhase::EngineInflight,
        "inflight token waits for the real engine completion after fatal"
    );
    EngineCompletion complete{true};
    Check(
        !TryMarkExecutionTokenCompleting<ProtocolTestOps>(
            inflight, complete, fatal
        ) && inflight.control.phase == ExecTokenPhase::Faulted,
        "completed engine converges without vend, flag or DONE"
    );
}

void TestCompletionFailureIsTerminalAndIdempotent() {
    const ExecPayloadSpec spec = MakeSpec();
    const SyntheticPayloadSource source = MakeSource();
    ReadyTable ready{};
    for (int32_t producer : source.fanin) {
        ready.ready[static_cast<uint32_t>(producer)].store(true);
    }
    EngineCompletion engine_completion{true};

    for (uint32_t failure_case = 0; failure_case < 3;
         ++failure_case) {
        SharedExecCell cell{};
        ExecutionToken token{};
        SharedExecFatalControl fatal{};
        InitializeCell(cell);
        InitializeToken(token);
        InitializeFatal(fatal);
        Check(
            BuildAndPublishExecPayload<ProtocolTestOps>(
                cell, 1, spec, source, fatal
            ) == ExecBuildResult::Published &&
            ClaimAndBindExecPayload<ProtocolTestOps>(
                cell, spec.task_id, 2,
                ExecEngineClass::Aiv, token, fatal
            ) == ExecClaimResult::Claimed &&
            TryMarkExecutionTokenEngineInflight<ProtocolTestOps>(
                token, ready, fatal
            ) &&
            TryMarkExecutionTokenCompleting<ProtocolTestOps>(
                token, engine_completion, fatal
            ),
            "failure case reaches completion boundary"
        );
        CompletionRecorder completion{};
        if (failure_case == 0) {
            completion.allow_vend = false;
            Check(
                PublishExecDoneAfterCompletion<ProtocolTestOps>(
                    cell, token, completion, fatal
                ) == ExecDoneResult::VendPublishFailed &&
                    token.control.phase ==
                        ExecTokenPhase::Completing,
                "vend failure remains before vend publication"
            );
            completion.allow_vend = true;
        } else if (failure_case == 1) {
            completion.allow_flag = false;
            Check(
                PublishExecDoneAfterCompletion<ProtocolTestOps>(
                    cell, token, completion, fatal
                ) == ExecDoneResult::FlagPublishFailed &&
                    token.control.phase ==
                        ExecTokenPhase::VendPublished,
                "flag failure preserves successful vend publication"
            );
            completion.allow_flag = true;
        } else {
            const int64_t claimed = cell.control.state;
            cell.control.state = static_cast<int64_t>(
                EncodeExecState(
                    ExecPhase::Claimed,
                    token.control.build_owner, 3,
                    ExecEngineClass::Aiv,
                    token.control.payload_lines,
                    token.control.task_id
                )
            );
            Check(
                PublishExecDoneAfterCompletion<ProtocolTestOps>(
                    cell, token, completion, fatal
                ) == ExecDoneResult::StateConflict &&
                    token.control.phase ==
                        ExecTokenPhase::CompletionPublished,
                "DONE conflict preserves completed external publication"
            );
            cell.control.state = claimed;
        }

        const uint32_t event_count = completion.event_count;
        const uint32_t vend_calls = completion.vend_calls;
        const uint32_t flag_calls = completion.flag_calls;
        Check(
            PublishExecDoneAfterCompletion<ProtocolTestOps>(
                cell, token, completion, fatal
            ) == ExecDoneResult::FatalObserved,
            "terminal fatal blocks any completion retry"
        );
        Check(
            completion.event_count == event_count &&
                completion.vend_calls == vend_calls &&
                completion.flag_calls == flag_calls,
            "fatal retry repeats no completion side effect"
        );
        if (failure_case == 0) {
            Check(
                completion.event_count == 0 &&
                    completion.vend_calls == 1 &&
                    completion.flag_calls == 0,
                "failed vend publishes no flag"
            );
        } else if (failure_case == 1) {
            Check(
                completion.vend_calls == 1 &&
                    completion.flag_calls == 1 &&
                    completion.event_count == 1 &&
                    completion.events[0] == 1,
                "flag failure preserves the single successful vend"
            );
        } else {
            Check(
                completion.vend_calls == 1 &&
                    completion.flag_calls == 1 &&
                    completion.event_count == 2,
                "DONE conflict preserves one vend and one flag"
            );
        }
        Check(
            ExecFatalPublished<ProtocolTestOps>(fatal),
            "every irreversible completion failure is terminal"
        );
    }
}

void TestEngineRoutingAndInputValidation() {
    SharedExecCell cell{};
    ExecutionToken token{};
    SharedExecFatalControl fatal{};
    InitializeCell(cell);
    InitializeToken(token);
    InitializeFatal(fatal);
    const SyntheticPayloadSource source = MakeSource();
    const ExecPayloadSpec spec = MakeSpec(
        42, ExecEngineClass::Aic
    );
    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            cell, 1, spec, source, fatal
        ) == ExecBuildResult::Published,
        "AIC payload is published"
    );
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            cell, spec.task_id, 2, ExecEngineClass::Aiv,
            token, fatal
        ) == ExecClaimResult::Incompatible,
        "AIV rejects an AIC payload before CAS"
    );
    Check(
        ProtocolTestOps::invalidate_calls.load() == 0 &&
            ProtocolTestOps::payload_loads.load() == 0 &&
            CurrentState(cell).phase == ExecPhase::Built,
        "incompatible executor leaves the task available"
    );
    Check(
        ClaimAndBindExecPayload<ProtocolTestOps>(
            cell, spec.task_id, 3, ExecEngineClass::Aic,
            token, fatal
        ) == ExecClaimResult::Claimed,
        "compatible AIC executor claims the task"
    );

    SharedExecCell invalid{};
    InitializeCell(invalid);
    ExecPayloadSpec bad = spec;
    bad.tensor_count = kExecMaxTensors + 1;
    ProtocolTestOps::ResetTrace();
    Check(
        BuildAndPublishExecPayload<ProtocolTestOps>(
            invalid, 1, bad, source, fatal
        ) == ExecBuildResult::InvalidInput,
        "oversized payload is rejected before BUILDING"
    );
    Check(
        CurrentState(invalid).phase == ExecPhase::Empty &&
            ProtocolTestOps::built_cas_event.load() == 0 &&
            ProtocolTestOps::fatal_cas_calls.load() == 1 &&
            ProtocolTestOps::payload_stores.load() == 0,
        "invalid build input leaves the cell untouched"
    );
}

}  // namespace

int main() {
    TestLayoutAndStateAbi();
    TestPublishBindAndComplete();
    TestHalfBuiltPayloadIsInvisible();
    TestFlushedPayloadIsInvisibleBeforeBuilt();
    TestConcurrentExecutorsHaveOneWinner();
    TestConcurrentBuildersPublishExactlyOnce();
    TestBusyExecutorDoesNotTouchCell();
    TestTaskIdentityGuardsClaimAndCompletion();
    TestClaimedPayloadCannotBeRebuilt();
    TestInvalidControlAndPayloadFailClosed();
    TestCorruptCountsRemainBounded();
    TestFaninGatesExecutionAndRejectsFutureProducer();
    TestBuildRejectsInvalidFaninBeforePublication();
    TestBuildingFailurePublishesFatalAndCutsOffWork();
    TestFatalFirstFailureWins();
    TestFatalStopsWaitingAndInflightTokens();
    TestCompletionFailureIsTerminalAndIdempotent();
    TestEngineRoutingAndInputValidation();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] cross-core protocol: %d checks failed\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] cross-core shared execution protocol\n");
    return 0;
}
