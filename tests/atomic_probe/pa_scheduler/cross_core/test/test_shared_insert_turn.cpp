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

#include "pa_scheduler_core.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <new>
#include <sys/mman.h>
#include <thread>

namespace {

using namespace pa_scheduler;

static_assert(kSharedInsertTurnCapacity == 128, "completion-chain test expects the full legacy sidecar");

constexpr uint32_t kSequentialTasks = 260;
using LegacyTurnSnapshot = std::array<int64_t, kSharedInsertTurnCapacity>;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] shared per-task insert completion: %s\n", message);
    ++g_failures;
}

struct CompletionTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline SchedulerState *observed_state = nullptr;
    static inline std::atomic<uint64_t> load_calls{0};
    static inline std::atomic<uint64_t> cas_calls{0};
    static inline std::atomic<uint64_t> fetch_add_calls{0};
    static inline std::atomic<uint64_t> legacy_turn_touches{0};
    static inline std::atomic<uintptr_t> last_load_address{0};
    static inline std::atomic<uintptr_t> last_cas_address{0};
    static inline std::atomic<uintptr_t> last_fetch_add_address{0};

    static bool IsLegacyTurnAddress(const volatile int64_t *address) {
        if (observed_state == nullptr) {
            return false;
        }
        for (uint32_t lane = 0; lane < kSharedInsertTurnCapacity; ++lane) {
            if (address == &SharedInsertTurnLine(observed_state->shared_map, lane).value) {
                return true;
            }
        }
        return false;
    }

    static int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(address, static_cast<int32_t>(0), __ATOMIC_ACQUIRE);
    }

    static int64_t Load(volatile int64_t *address) {
        load_calls.fetch_add(1, std::memory_order_relaxed);
        last_load_address.store(reinterpret_cast<uintptr_t>(address), std::memory_order_relaxed);
        if (IsLegacyTurnAddress(address)) {
            legacy_turn_touches.fetch_add(1, std::memory_order_relaxed);
        }
        return __atomic_fetch_add(address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE);
    }

    static int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static uint64_t Exchange(volatile uint64_t *address, uint64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        cas_calls.fetch_add(1, std::memory_order_relaxed);
        last_cas_address.store(reinterpret_cast<uintptr_t>(address), std::memory_order_relaxed);
        if (IsLegacyTurnAddress(address)) {
            legacy_turn_touches.fetch_add(1, std::memory_order_relaxed);
        }
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        fetch_add_calls.fetch_add(1, std::memory_order_relaxed);
        last_fetch_add_address.store(
            reinterpret_cast<uintptr_t>(address),
            std::memory_order_relaxed
        );
        if (IsLegacyTurnAddress(address)) {
            legacy_turn_touches.fetch_add(1, std::memory_order_relaxed);
        }
        return __atomic_fetch_add(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static int64_t FetchMax(volatile int64_t *address, int64_t value, uint64_t &retries) {
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

    static void StoreBarrier() { std::atomic_thread_fence(std::memory_order_seq_cst); }

    static void FlushRegion(void *, uint64_t) { StoreBarrier(); }

    static void InvalidateRegion(const void *, uint64_t) { StoreBarrier(); }

    static uint64_t Now() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count()
        );
    }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value) {
        asm volatile("" : "+r"(value));
        return Now();
    }

    static void SpinHint() { std::this_thread::yield(); }

    static void ResetTrace(SchedulerState &state) {
        observed_state = &state;
        load_calls.store(0, std::memory_order_relaxed);
        cas_calls.store(0, std::memory_order_relaxed);
        fetch_add_calls.store(0, std::memory_order_relaxed);
        legacy_turn_touches.store(0, std::memory_order_relaxed);
        last_load_address.store(0, std::memory_order_relaxed);
        last_cas_address.store(0, std::memory_order_relaxed);
        last_fetch_add_address.store(0, std::memory_order_relaxed);
    }
};

SchedulerState *MapSparseSchedulerState() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(nullptr, sizeof(SchedulerState), PROT_READ | PROT_WRITE, flags, -1, 0);
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    return ::new (memory) SchedulerState;
}

void UnmapSparseSchedulerState(SchedulerState *state) {
    if (state != nullptr) {
        (void)munmap(state, sizeof(*state));
    }
}

LegacyTurnSnapshot SeedLegacyTurns(SchedulerState &state) {
    LegacyTurnSnapshot snapshot{};
    for (uint32_t lane = 0; lane < kSharedInsertTurnCapacity; ++lane) {
        snapshot[lane] = -10000 - static_cast<int64_t>(lane);
        SharedInsertTurnLine(state.shared_map, lane).value = snapshot[lane];
    }
    return snapshot;
}

bool LegacyTurnsMatch(const SchedulerState &state, const LegacyTurnSnapshot &snapshot) {
    for (uint32_t lane = 0; lane < kSharedInsertTurnCapacity; ++lane) {
        const volatile int64_t *address =
            &SharedInsertTurnLine(const_cast<SharedTensorMapSidecar &>(state.shared_map), lane).value;
        if (__atomic_load_n(address, __ATOMIC_ACQUIRE) != snapshot[lane]) {
            return false;
        }
    }
    return true;
}

void ResetCompletionWords(SchedulerState &state, uint32_t count) {
    state.fatal.value = 0;
    for (uint32_t task = 0; task < count; ++task) {
        state.tasks[task].deps_prepared =
            SharedInsertCompletionInitialValue(task);
        state.claim_tournament[task]
            .root.insert_completion.value =
            SharedInsertCompletionInitialValue(task);
    }
}

volatile int64_t &CompletionWord(
    SchedulerState &state, uint32_t task
) {
    return state.claim_tournament[task]
        .root.insert_completion.value;
}

bool TaskCellCompletionCanariesMatch(
    const SchedulerState &state, uint32_t count
) {
    for (uint32_t task = 0; task < count; ++task) {
        if (state.tasks[task].deps_prepared !=
            SharedInsertCompletionInitialValue(task)) {
            return false;
        }
    }
    return true;
}

bool AddressEquals(uintptr_t observed, volatile int64_t *expected) {
    return observed == reinterpret_cast<uintptr_t>(expected);
}

void TestSequentialCompletionChain(SchedulerState &state) {
    ResetCompletionWords(state, kSequentialTasks + 1U);
    const LegacyTurnSnapshot legacy = SeedLegacyTurns(state);
    bool exact = true;

    for (uint32_t task = 0; task < kSequentialTasks; ++task) {
        CompletionTestOps::ResetTrace(state);
        LocalStats wait_stats{};
        int64_t ready_observed = INT64_MIN;
        uint64_t load_count = UINT64_MAX;
        const bool ready = WaitForSharedTaskInsertTurn<CompletionTestOps>(
            &state, static_cast<int32_t>(task), wait_stats, ready_observed, load_count
        );
        exact &= ready && state.fatal.value == 0;
        if (task == 0) {
            exact &= ready_observed == -1 && load_count == 0 &&
                     CompletionTestOps::load_calls.load(std::memory_order_relaxed) == 0;
        } else {
            exact &= ready_observed == static_cast<int64_t>(task - 1U) && load_count == 1 &&
                     CompletionTestOps::load_calls.load(std::memory_order_relaxed) == 1 &&
                     AddressEquals(
                         CompletionTestOps::last_load_address.load(std::memory_order_relaxed),
                         &CompletionWord(state, task - 1U)
                     );
        }
        exact &= CompletionTestOps::cas_calls.load(std::memory_order_relaxed) == 0 &&
                 CompletionTestOps::legacy_turn_touches.load(std::memory_order_relaxed) == 0;

        CompletionTestOps::ResetTrace(state);
        LocalStats publish_stats{};
        const bool published = HandoffSharedTaskInsertTurn<CompletionTestOps>(
            &state, static_cast<int32_t>(task), publish_stats
        );
        exact &=
            published && CompletionWord(state, task) == static_cast<int64_t>(task) &&
            CompletionTestOps::cas_calls.load(std::memory_order_relaxed) == 0 &&
            CompletionTestOps::fetch_add_calls.load(std::memory_order_relaxed) == 1 &&
            AddressEquals(
                CompletionTestOps::last_fetch_add_address.load(std::memory_order_relaxed),
                &CompletionWord(state, task)
            ) &&
            CompletionTestOps::legacy_turn_touches.load(std::memory_order_relaxed) == 0 &&
            LegacyTurnsMatch(state, legacy);
    }

    for (uint32_t task = 0; task < kSequentialTasks; ++task) {
        exact &= CompletionWord(state, task) == static_cast<int64_t>(task);
    }
    exact &= CompletionWord(state, kSequentialTasks) ==
        SharedInsertCompletionInitialValue(kSequentialTasks);
    exact &= TaskCellCompletionCanariesMatch(
        state, kSequentialTasks + 1U
    );
    Check(
        exact, "task 0 skips predecessor; every N waits only N-1, "
               "publishes only N, and never touches legacy turns"
    );
}

void TestPendingOwnerWakesOnPredecessor(SchedulerState &state) {
    ResetCompletionWords(state, 2);
    const LegacyTurnSnapshot legacy = SeedLegacyTurns(state);
    CompletionTestOps::ResetTrace(state);
    std::atomic<bool> wait_finished{false};
    bool wait_ok = false;
    int64_t ready_observed = INT64_MIN;
    uint64_t load_count = 0;
    LocalStats wait_stats{};
    std::thread waiter([&]() {
        wait_ok = WaitForSharedTaskInsertTurn<CompletionTestOps>(&state, 1, wait_stats, ready_observed, load_count);
        wait_finished.store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (CompletionTestOps::load_calls.load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool observed_pending = CompletionTestOps::load_calls.load(std::memory_order_acquire) != 0 &&
                                  !wait_finished.load(std::memory_order_acquire);

    LocalStats publish_stats{};
    const bool published = HandoffSharedTaskInsertTurn<CompletionTestOps>(
        &state, 0, publish_stats
    );
    waiter.join();

    Check(
        observed_pending && published && wait_ok && ready_observed == 0 && load_count >= 2 &&
            CompletionWord(state, 0) == 0 &&
            CompletionWord(state, 1) ==
                SharedInsertCompletionInitialValue(1) &&
            TaskCellCompletionCanariesMatch(state, 2) &&
            state.fatal.value == 0 &&
            CompletionTestOps::legacy_turn_touches.load(std::memory_order_relaxed) == 0 &&
            LegacyTurnsMatch(state, legacy),
        "task 1 remains pending until task 0 publishes its "
        "per-task completion word"
    );
}

void TestEmptyWriterStillCompletes(SchedulerState &state) {
    ResetCompletionWords(state, 2);
    const LegacyTurnSnapshot legacy = SeedLegacyTurns(state);
    bool exact = true;

    for (int32_t task = 0; task < 2; ++task) {
        TaskArgs args;
        ConstructTaskArgs(args);
        SubmitContext context{};
        context.task_id = task;
        context.won = true;
        context.result.task_id = task;
        context.shared_result.Reset(task);
        SharedTaskWriterDelta delta{};
        LocalStats stats{};
        CompletionTestOps::ResetTrace(state);
        exact &= PrepareSharedTaskWriterDelta(args, context, delta) && delta.ordinary_count == 0 &&
                 !delta.writer_intent_required &&
                 PublishSharedTaskWriterDelta<CompletionTestOps>(&state, context, delta, stats) &&
                 CompletionWord(state, static_cast<uint32_t>(task)) == task &&
                 CompletionTestOps::fetch_add_calls.load(std::memory_order_relaxed) == 1 &&
                 AddressEquals(
                     CompletionTestOps::last_fetch_add_address.load(std::memory_order_relaxed),
                     &CompletionWord(state, static_cast<uint32_t>(task))
                 ) &&
                 CompletionTestOps::legacy_turn_touches.load(std::memory_order_relaxed) == 0;
    }

    Check(
        exact && state.fatal.value == 0 &&
            TaskCellCompletionCanariesMatch(state, 2) &&
            LegacyTurnsMatch(state, legacy),
        "empty metadata transactions still publish one "
        "completion per task without a sidecar baton"
    );
}

void TestOutputsPublishBeforePredecessorWait(
    SchedulerState &state
) {
    ResetCompletionWords(state, 2);
    const LegacyTurnSnapshot legacy = SeedLegacyTurns(state);
    SharedOutputCell &cell = state.shared_map.shared_outputs[1];
    cell.published[0].value = -1;
    cell.last_writer[0].value = -1;
    TensorDesc descriptor{};
    descriptor.buffer_addr = 0x510000000ULL;
    descriptor.ndims = 1;
    descriptor.shapes[0] = 16;
    descriptor.strides[0] = 1;

    TaskArgs args;
    ConstructTaskArgs(args);
    SubmitContext context{};
    context.task_id = 1;
    context.won = true;
    context.result.task_id = 1;
    context.result.count = 1;
    context.result.tensors[0] = &descriptor;
    context.shared_result.Reset(1);
    const bool output_ref_ok =
        context.shared_result.AddOutputRef(1, 0);
    SharedTaskWriterDelta delta{};
    const bool delta_ok =
        PrepareSharedTaskWriterDelta(args, context, delta);

    CompletionTestOps::ResetTrace(state);
    std::atomic<bool> publish_finished{false};
    bool publish_ok = false;
    LocalStats task_stats{};
    std::thread owner([&]() {
        publish_ok =
            PublishSharedTaskWriterDelta<CompletionTestOps>(
                &state, context, delta, task_stats
            );
        publish_finished.store(true, std::memory_order_release);
    });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (
        __atomic_load_n(
            &cell.published[0].value, __ATOMIC_ACQUIRE
        ) != 1 &&
        std::chrono::steady_clock::now() < deadline
    ) {
        std::this_thread::yield();
    }
    const bool visible_before_turn =
        __atomic_load_n(
            &cell.published[0].value, __ATOMIC_ACQUIRE
        ) == 1 &&
        __atomic_load_n(
            &CompletionWord(state, 1), __ATOMIC_ACQUIRE
        ) == SharedInsertCompletionInitialValue(1) &&
        !publish_finished.load(std::memory_order_acquire);

    LocalStats predecessor_stats{};
    const bool predecessor_published =
        HandoffSharedTaskInsertTurn<CompletionTestOps>(
            &state, 0, predecessor_stats
        );
    owner.join();

    Check(
        output_ref_ok && delta_ok && visible_before_turn &&
            predecessor_published &&
            publish_ok &&
            state.fatal.value == 0 &&
            CompletionWord(state, 0) == 0 &&
            CompletionWord(state, 1) == 1 &&
            TaskCellCompletionCanariesMatch(state, 2) &&
            cell.tensors[0].buffer_addr ==
                descriptor.buffer_addr &&
            LegacyTurnsMatch(state, legacy),
        "fresh output is visible while task 1 still waits for task 0, "
        "and deps_prepared closes only after serialized metadata"
    );
}

void TestCorruptionAndDuplicateFailClosed(SchedulerState &state) {
    bool exact = true;

    ResetCompletionWords(state, 5);
    LegacyTurnSnapshot legacy = SeedLegacyTurns(state);
    CompletionWord(state, 2) = -9;
    CompletionTestOps::ResetTrace(state);
    LocalStats corrupt_predecessor_stats{};
    int64_t ready_observed = INT64_MIN;
    uint64_t load_count = 0;
    exact &= !WaitForSharedTaskInsertTurn<CompletionTestOps>(
                 &state, 3, corrupt_predecessor_stats, ready_observed, load_count
             ) &&
             state.fatal.value == 1 && CompletionWord(state, 2) == -9 &&
             CompletionTestOps::cas_calls.load(std::memory_order_relaxed) == 0 &&
             CompletionTestOps::legacy_turn_touches.load(std::memory_order_relaxed) == 0 &&
             LegacyTurnsMatch(state, legacy);

    ResetCompletionWords(state, 2);
    legacy = SeedLegacyTurns(state);
    LocalStats first_stats{};
    exact &=
        HandoffSharedTaskInsertTurn<CompletionTestOps>(
            &state, 0, first_stats
        );
    state.fatal.value = 0;
    CompletionTestOps::ResetTrace(state);
    LocalStats duplicate_stats{};
    exact &= HandoffSharedTaskInsertTurn<CompletionTestOps>(
                 &state, 0, duplicate_stats
             ) &&
             CompletionWord(state, 0) == 1 &&
             state.fatal.value == 0 &&
             CompletionTestOps::fetch_add_calls.load(std::memory_order_relaxed) == 1 &&
             CompletionTestOps::legacy_turn_touches.load(std::memory_order_relaxed) == 0 &&
             LegacyTurnsMatch(state, legacy);
    LocalStats duplicate_observer_stats{};
    int64_t duplicate_ready = INT64_MIN;
    uint64_t duplicate_loads = 0;
    exact &= !WaitForSharedTaskInsertTurn<CompletionTestOps>(
                 &state, 1, duplicate_observer_stats,
                 duplicate_ready, duplicate_loads
             ) &&
             duplicate_ready == -1 && state.fatal.value == 1;

    ResetCompletionWords(state, 2);
    legacy = SeedLegacyTurns(state);
    CompletionWord(state, 1) = 99;
    CompletionTestOps::ResetTrace(state);
    LocalStats bad_current_stats{};
    exact &= HandoffSharedTaskInsertTurn<CompletionTestOps>(
                 &state, 1, bad_current_stats
             ) &&
             CompletionWord(state, 1) == 100 &&
             state.fatal.value == 0 &&
             CompletionTestOps::fetch_add_calls.load(std::memory_order_relaxed) == 1 &&
             CompletionTestOps::legacy_turn_touches.load(std::memory_order_relaxed) == 0 &&
             LegacyTurnsMatch(state, legacy);
    LocalStats bad_current_observer_stats{};
    int64_t bad_current_ready = INT64_MIN;
    uint64_t bad_current_loads = 0;
    exact &= !WaitForSharedTaskInsertTurn<CompletionTestOps>(
                 &state, 2, bad_current_observer_stats,
                 bad_current_ready, bad_current_loads
             ) &&
             state.fatal.value == 1;

    Check(
        exact && TaskCellCompletionCanariesMatch(state, 2),
        "corrupt predecessor, duplicate completion, and "
               "unexpected current value are rejected by the next owner"
    );
}

}  // namespace

int main() {
    SchedulerState *state = MapSparseSchedulerState();
    if (state == nullptr) {
        return 1;
    }

    TestSequentialCompletionChain(*state);
    TestPendingOwnerWakesOnPredecessor(*state);
    TestEmptyWriterStillCompletes(*state);
    TestOutputsPublishBeforePredecessorWait(*state);
    TestCorruptionAndDuplicateFailClosed(*state);

    UnmapSparseSchedulerState(state);
    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "[FAIL] shared per-task insert completion tests: "
            "%d failure(s)\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared per-task insert completion chain tests\n");
    return 0;
}
