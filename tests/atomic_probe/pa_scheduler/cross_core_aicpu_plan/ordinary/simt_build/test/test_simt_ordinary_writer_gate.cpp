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

#define PTO_FDWIC_SHARED_MAP 1
#define PA_BUILD_SWIMLANE 1

#include "../common/simt_plan_build_protocol.h"
#include "../../scalar_build/common/pa_scheduler_core.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace simt = pa_scheduler::aicpu_plan_simt;
using namespace pa_scheduler;

constexpr uint32_t kTaskCount = 17U;
constexpr uint32_t kDelayedReaderTask = 2U;
constexpr uint32_t kFutureWriterTask = 5U;
constexpr uint64_t kSharedAddress = 0x5A7100000ULL;
constexpr auto kDeadlineDuration = std::chrono::seconds(15);

bool IsWriter(uint32_t task_id)
{
    return task_id == 0U || task_id % 4U == 1U;
}

int32_t PreviousWriter(uint32_t task_id)
{
    for (uint32_t candidate = task_id; candidate != 0U; --candidate) {
        const uint32_t previous = candidate - 1U;
        if (IsWriter(previous)) return static_cast<int32_t>(previous);
    }
    return -1;
}

struct GateOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline SchedulerState *state = nullptr;
    static inline std::array<std::atomic<uint32_t>, kTaskCount>
        metadata_flushes{};
    static inline std::array<std::atomic<uint32_t>, kTaskCount>
        completion_publications{};
    static inline std::atomic<uint32_t> protocol_failures{0U};

    static void ResetObservations(SchedulerState *scheduler)
    {
        state = scheduler;
        protocol_failures.store(0U, std::memory_order_relaxed);
        for (uint32_t task = 0U; task < kTaskCount; ++task) {
            metadata_flushes[task].store(0U, std::memory_order_relaxed);
            completion_publications[task].store(0U, std::memory_order_relaxed);
        }
    }

    static void Fail()
    {
        protocol_failures.fetch_add(1U, std::memory_order_relaxed);
    }

    static int32_t Load(volatile int32_t *address)
    {
        return __atomic_fetch_add(address, int32_t{0}, __ATOMIC_ACQUIRE);
    }

    static int64_t Load(volatile int64_t *address)
    {
        return __atomic_fetch_add(address, int64_t{0}, __ATOMIC_ACQUIRE);
    }

    static uint64_t Load(volatile uint64_t *address)
    {
        return __atomic_fetch_add(address, uint64_t{0}, __ATOMIC_ACQUIRE);
    }

    static int32_t Exchange(volatile int32_t *address, int32_t value)
    {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value)
    {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static uint64_t Exchange(volatile uint64_t *address, uint64_t value)
    {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected, int64_t desired
    )
    {
        int32_t completion_task = -1;
        if (state != nullptr) {
            for (uint32_t task = 0U; task < kTaskCount; ++task) {
                if (address == &state->claim_tournament[task]
                                    .root.insert_completion.value) {
                    completion_task = static_cast<int32_t>(task);
                    break;
                }
            }
        }
        if (completion_task >= 0) {
            const uint32_t task = static_cast<uint32_t>(completion_task);
            const int64_t initial = SharedInsertCompletionInitialValue(task);
            if (expected != initial || desired != static_cast<int64_t>(task)) {
                Fail();
            }
            if (task != 0U &&
                Load(&state->claim_tournament[task - 1U]
                           .root.insert_completion.value) !=
                    static_cast<int64_t>(task - 1U)) {
                Fail();
            }
            const uint32_t expected_flushes = IsWriter(task) ? 1U : 0U;
            if (metadata_flushes[task].load(std::memory_order_acquire) !=
                expected_flushes) {
                Fail();
            }
        }

        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        if (completion_task >= 0 && observed == expected) {
            completion_publications[static_cast<uint32_t>(completion_task)]
                .fetch_add(1U, std::memory_order_release);
        }
        return observed;
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value)
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    )
    {
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0U;
        while (current < value) {
            if (__atomic_compare_exchange_n(
                    address, &current, value, true,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
                )) {
                break;
            }
            ++retries;
        }
        return current;
    }

    static int64_t LoadControl(const volatile int64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static int64_t FetchAddControl(volatile int64_t *address, int64_t value)
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static void PublishControl(volatile int64_t *address, int64_t value)
    {
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void FlushRegion(void *address, uint64_t bytes)
    {
        if (state != nullptr && bytes == sizeof(SharedRegionPayload)) {
            const uintptr_t current = reinterpret_cast<uintptr_t>(address);
            const uintptr_t begin = reinterpret_cast<uintptr_t>(
                &state->shared_map.slots[0]
            );
            const uintptr_t end = reinterpret_cast<uintptr_t>(
                &state->shared_map.slots[kMapCapacity]
            );
            if (current >= begin && current < end) {
                const uintptr_t relative = current - begin;
                const uint32_t slot_index = static_cast<uint32_t>(
                    relative / sizeof(SharedRegionSlot)
                );
                SharedRegionSlot &slot = state->shared_map.slots[slot_index];
                if (address != static_cast<void *>(&slot.payload)) {
                    Fail();
                } else {
                    const SharedRegionValue &value = slot.payload.value;
                    if (value.producer < 0 ||
                        value.producer >= static_cast<int32_t>(kTaskCount) ||
                        !IsWriter(static_cast<uint32_t>(value.producer)) ||
                        value.buffer_addr != kSharedAddress ||
                        value.lo != 0U || value.hi != 4096U ||
                        value.reserved != 0) {
                        Fail();
                    } else {
                        const uint32_t task =
                            static_cast<uint32_t>(value.producer);
                        if (task != 0U &&
                            Load(&state->claim_tournament[task - 1U]
                                       .root.insert_completion.value) !=
                                static_cast<int64_t>(task - 1U)) {
                            Fail();
                        }
                        if (Load(&state->claim_tournament[task]
                                     .root.insert_completion.value) !=
                            SharedInsertCompletionInitialValue(task)) {
                            Fail();
                        }
                        metadata_flushes[task].fetch_add(
                            1U, std::memory_order_release
                        );
                    }
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void InvalidateRegion(const void *, uint64_t)
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static uint64_t Now()
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value)
    {
        asm volatile("" : "+r"(value));
        return Now();
    }

    static void SpinHint()
    {
        std::this_thread::yield();
    }
};

struct AlignedCellFree {
    void operator()(plan::RuntimeTaskPlanCell *pointer) const
    {
        std::free(pointer);
    }
};

struct PlanFixture {
    alignas(plan::kAtomicIsolationBytes) plan::RuntimePlanControl control{};
    std::unique_ptr<plan::RuntimeTaskPlanCell, AlignedCellFree> cells;
    plan::RuntimePlanView view{};

    PlanFixture()
    {
        void *raw = nullptr;
        const size_t bytes = sizeof(plan::RuntimeTaskPlanCell) * kTaskCount;
        if (posix_memalign(&raw, plan::kAtomicIsolationBytes, bytes) != 0) {
            return;
        }
        std::memset(raw, 0, bytes);
        cells.reset(static_cast<plan::RuntimeTaskPlanCell *>(raw));
        control.planned_frontier.value = kTaskCount;
        control.closed_task_count.value = kTaskCount;
        control.build_next.value = 0;
        control.build_workers_done.value = 0;
        control.build_release.value = plan::kBuildReleasePending;
        control.fatal.value = 0;
        view = {&control, cells.get(), kTaskCount};
    }

    bool Valid() const
    {
        return cells != nullptr;
    }
};

SchedulerState *MapSchedulerState()
{
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(SchedulerState), PROT_READ | PROT_WRITE,
        flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    return ::new (memory) SchedulerState;
}

void UnmapSchedulerState(SchedulerState *state)
{
    if (state != nullptr) {
        (void)munmap(state, sizeof(SchedulerState));
    }
}

void ResetSchedulerState(SchedulerState &state)
{
    state.fatal.value = 0;
    state.heap_window = kHeapWindow;
    state.shared_map.reclaim_upto.value = -1;
    for (uint32_t bucket = 0U; bucket < kMapBuckets; ++bucket) {
        state.shared_map.buckets[bucket].head.value = 0;
        state.shared_map.buckets[bucket].tail.value = 0;
    }
    for (uint32_t slot = 0U; slot < kMapCapacity; ++slot) {
        state.shared_map.slots[slot].seq.value = kSharedMapEmptySeq;
    }
    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        state.claim_tournament[task].root.insert_completion.value =
            SharedInsertCompletionInitialValue(task);
        SharedOutputCell &outputs = state.shared_map.shared_outputs[task];
        for (uint32_t output = 0U;
             output < kSharedOutputMaxPerTask; ++output) {
            outputs.published[output].value = -1;
            outputs.last_writer[output].value = -1;
        }
    }
}

TensorDesc MakeTensor()
{
    TensorDesc tensor{};
    tensor.buffer_addr = kSharedAddress;
    tensor.buffer_size = 4096U;
    tensor.owner_task_id = kInvalidTaskId;
    tensor.ndims = 1;
    tensor.dtype = DataType::Float32;
    tensor.manual_dep = false;
    tensor.is_contiguous = true;
    tensor.shapes[0] = 1024;
    tensor.strides[0] = 1;
    return tensor;
}

bool WaitUntil(
    std::chrono::steady_clock::time_point deadline,
    const std::function<bool()> &predicate
)
{
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

bool CheckFinalMap(SchedulerState &state)
{
    const uint32_t bucket = TensorMapHash(kSharedAddress);
    uint32_t writer_count = 0U;
    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        writer_count += IsWriter(task) ? 1U : 0U;
    }
    if (GateOps::Load(&state.shared_map.buckets[bucket].head.value) != 0 ||
        GateOps::Load(&state.shared_map.buckets[bucket].tail.value) !=
            static_cast<int64_t>(writer_count)) {
        return false;
    }
    uint32_t cursor = 0U;
    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        if (!IsWriter(task)) continue;
        SharedRegionValue snapshot{};
        if (!SharedReadRegionSlot<GateOps>(
                state.shared_map, bucket, cursor, snapshot
            ) ||
            snapshot.buffer_addr != kSharedAddress ||
            snapshot.lo != 0U || snapshot.hi != 4096U ||
            snapshot.producer != static_cast<int32_t>(task) ||
            snapshot.reserved != 0) {
            return false;
        }
        ++cursor;
    }
    return cursor == writer_count;
}

bool RunFourLeaderOrdinaryWriterGate()
{
    std::unique_ptr<SchedulerState, void (*)(SchedulerState *)> state(
        MapSchedulerState(), UnmapSchedulerState
    );
    if (state == nullptr) return false;
    ResetSchedulerState(*state);
    GateOps::ResetObservations(state.get());

    PlanFixture fixture;
    if (!fixture.Valid()) return false;
    uint32_t attached_task_count = UINT32_MAX;
    if (!simt::AttachClosedPlan<GateOps>(
            fixture.view, attached_task_count
        ) || attached_task_count != kTaskCount) {
        return false;
    }

    std::vector<TensorDesc> tensors(kTaskCount);
    std::vector<TaskArgs> args(kTaskCount);
    std::vector<LocalStats> stats(kTaskCount);
    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        tensors[task] = MakeTensor();
        ConstructTaskArgs(args[task]);
        AddGmTensor(
            args[task], tensors[task],
            IsWriter(task)
                ? (task == 0U ? TensorArgType::OutputExisting
                              : TensorArgType::Inout)
                : TensorArgType::Input
        );
    }

    std::array<std::thread, simt::kBuilderLeaders> leaders;
    std::array<simt::BuilderLeaderLocalState, simt::kBuilderLeaders>
        leader_states{};
    std::array<std::atomic<uint32_t>, simt::kBuilderLeaders> arrivals{};
    std::array<std::atomic<uint32_t>, kTaskCount> visits{};
    std::array<std::atomic<uint32_t>, kTaskCount> entered{};
    std::atomic<bool> ok{true};
    std::atomic<bool> start{false};
    std::atomic<bool> release_task_zero{false};
    std::atomic<bool> future_writer_visible_to_delayed_reader{false};
    std::atomic<uint32_t> last_arrivals{0U};
    const auto deadline = std::chrono::steady_clock::now() +
                          kDeadlineDuration;

    const auto fail = [&] {
        ok.store(false, std::memory_order_release);
        GateOps::PublishControl(&fixture.control.fatal.value, 1);
        release_task_zero.store(true, std::memory_order_release);
        start.store(true, std::memory_order_release);
    };

    for (uint32_t leader = 0U; leader < simt::kBuilderLeaders; ++leader) {
        leaders[leader] = std::thread([&, leader] {
            if (!WaitUntil(deadline, [&] {
                    return start.load(std::memory_order_acquire) ||
                           !ok.load(std::memory_order_acquire);
                })) {
                fail();
                return;
            }
            while (ok.load(std::memory_order_acquire)) {
                const plan::BuildReservation reservation =
                    simt::TakeAttachedBuildTicket<GateOps>(
                        fixture.view, attached_task_count
                    );
                if (reservation.status ==
                    plan::BuildReservationStatus::Closed) {
                    arrivals[leader].fetch_add(1U, std::memory_order_relaxed);
                    const plan::BuildArrivalStatus arrival =
                        simt::ArriveBuilderLeaderOnce<GateOps>(
                            fixture.view, leader, leader_states[leader]
                        );
                    if (arrival == plan::BuildArrivalStatus::Last) {
                        last_arrivals.fetch_add(1U, std::memory_order_relaxed);
                        if (!simt::PublishBuildRelease<GateOps>(
                                fixture.view, kTaskCount
                            )) {
                            fail();
                        }
                    } else if (arrival !=
                               plan::BuildArrivalStatus::Arrived) {
                        fail();
                    }
                    return;
                }
                if (reservation.status !=
                        plan::BuildReservationStatus::Reserved ||
                    reservation.task_id >= kTaskCount) {
                    fail();
                    return;
                }
                const uint32_t task = reservation.task_id;
                if (visits[task].fetch_add(
                        1U, std::memory_order_acq_rel
                    ) != 0U) {
                    fail();
                    return;
                }
                entered[task].store(1U, std::memory_order_release);
                if (task == 0U && !WaitUntil(deadline, [&] {
                        return release_task_zero.load(
                                   std::memory_order_acquire
                               ) || !ok.load(std::memory_order_acquire);
                    })) {
                    fail();
                    return;
                }
                if (!ok.load(std::memory_order_acquire)) return;

                SubmitContext context{};
                context.task_id = static_cast<int32_t>(task);
                context.won = true;
                context.register_mask = IsWriter(task) ? 1U : 0U;
                context.result.task_id = static_cast<int32_t>(task);
                context.shared_result.Reset(static_cast<int32_t>(task));
                SharedTaskWriterDelta delta{};
                if (!PrepareSharedTaskWriterDelta(
                        args[task], context, delta
                    ) ||
                    delta.ordinary_count != (IsWriter(task) ? 1U : 0U) ||
                    !PublishSharedTaskWriterDelta<GateOps>(
                        state.get(), context, delta, stats[task]
                    )) {
                    fail();
                    return;
                }

                // 固定让 reader 2 在 writer 5 已经插入同一个 key 后再查询。
                // 这不是依赖等待，而是测试编排：lookup 必须越过 map 中的
                // future entry，仍只返回 max(writer < consumer)==1。
                if (task == kDelayedReaderTask) {
                    if (!WaitUntil(deadline, [&] {
                            return GateOps::Load(
                                       &state->claim_tournament[
                                           kFutureWriterTask
                                       ].root.insert_completion.value
                                   ) == static_cast<int64_t>(
                                            kFutureWriterTask
                                        ) ||
                                   !ok.load(std::memory_order_acquire);
                        })) {
                        fail();
                        return;
                    }
                    future_writer_visible_to_delayed_reader.store(
                        true, std::memory_order_release
                    );
                }

                bool lookup_ok = false;
                uint32_t ordinary_lookups = UINT32_MAX;
                int32_t fanin[kMaxFanin] = {};
                const uint32_t fanin_count =
                    CollectSharedFanin<GateOps, false, true>(
                        state->shared_map, args[task],
                        static_cast<int32_t>(task),
                        static_cast<int32_t>(state->heap_window),
                        stats[task], fanin, lookup_ok,
                        ordinary_lookups, &state->fatal.value
                    );
                const int32_t expected_writer = PreviousWriter(task);
                if (!lookup_ok || ordinary_lookups != 1U ||
                    (expected_writer < 0
                         ? fanin_count != 0U
                         : fanin_count != 1U ||
                               fanin[0] != expected_writer ||
                               fanin[0] >= static_cast<int32_t>(task))) {
                    fail();
                    return;
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    const uint32_t first_wave =
        kTaskCount < simt::kBuilderLeaders
            ? kTaskCount
            : simt::kBuilderLeaders;
    if (!WaitUntil(deadline, [&] {
            for (uint32_t task = 0U; task < first_wave; ++task) {
                if (entered[task].load(std::memory_order_acquire) == 0U) {
                    return false;
                }
            }
            return true;
        })) {
        fail();
    }
    // task0 人为挂起时，task1..3 已经拿到 ticket，但不得越过严格前驱链
    // 写普通 TensorMap，也不得发布自己的 completion。
    for (uint32_t task = 0U; task < first_wave; ++task) {
        if (GateOps::metadata_flushes[task].load(
                std::memory_order_acquire
            ) != 0U ||
            GateOps::completion_publications[task].load(
                std::memory_order_acquire
            ) != 0U) {
            fail();
        }
    }
    release_task_zero.store(true, std::memory_order_release);
    for (std::thread &leader : leaders) leader.join();

    bool exact = ok.load(std::memory_order_acquire) &&
        state->fatal.value == 0 &&
        GateOps::protocol_failures.load(std::memory_order_acquire) == 0U &&
        last_arrivals.load(std::memory_order_acquire) == 1U &&
        GateOps::LoadControl(&fixture.control.build_next.value) ==
            static_cast<int64_t>(simt::ExpectedBuildTicketCount(kTaskCount)) &&
        GateOps::LoadControl(&fixture.control.build_workers_done.value) ==
            static_cast<int64_t>(simt::kBuilderLeaders) &&
        GateOps::LoadControl(&fixture.control.build_release.value) ==
            static_cast<int64_t>(kTaskCount) &&
        future_writer_visible_to_delayed_reader.load(
            std::memory_order_acquire
        ) &&
        CheckFinalMap(*state);
    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        exact &= visits[task].load(std::memory_order_acquire) == 1U;
        exact &= GateOps::Load(
            &state->claim_tournament[task].root.insert_completion.value
        ) == static_cast<int64_t>(task);
        exact &= GateOps::metadata_flushes[task].load(
            std::memory_order_acquire
        ) == (IsWriter(task) ? 1U : 0U);
        exact &= GateOps::completion_publications[task].load(
            std::memory_order_acquire
        ) == 1U;
        exact &= stats[task].result.map_inserts ==
            (IsWriter(task) ? 1U : 0U);
    }
    for (uint32_t leader = 0U; leader < simt::kBuilderLeaders; ++leader) {
        exact &= arrivals[leader].load(std::memory_order_acquire) == 1U;
        exact &= leader_states[leader].arrival_recorded;
    }
    return exact;
}

bool TestDuplicateAndOutOfOrderFailClosed()
{
    std::unique_ptr<SchedulerState, void (*)(SchedulerState *)> state(
        MapSchedulerState(), UnmapSchedulerState
    );
    if (state == nullptr) return false;

    ResetSchedulerState(*state);
    GateOps::ResetObservations(state.get());
    LocalStats duplicate_stats{};
    constexpr uint32_t kDuplicateTask = 2U;
    state->claim_tournament[kDuplicateTask - 1U]
        .root.insert_completion.value =
        static_cast<int64_t>(kDuplicateTask - 1U);
    int64_t duplicate_observed = INT64_MIN;
    const bool first = HandoffSharedTaskInsertTurn<GateOps>(
        state.get(), static_cast<int32_t>(kDuplicateTask),
        duplicate_stats, duplicate_observed
    );
    duplicate_observed = INT64_MIN;
    const bool duplicate = HandoffSharedTaskInsertTurn<GateOps>(
        state.get(), static_cast<int32_t>(kDuplicateTask),
        duplicate_stats, duplicate_observed
    );
    const bool duplicate_failed = first && !duplicate &&
        duplicate_observed == static_cast<int64_t>(kDuplicateTask) &&
        state->fatal.value != 0 &&
        GateOps::protocol_failures.load(std::memory_order_acquire) == 0U;

    ResetSchedulerState(*state);
    GateOps::ResetObservations(state.get());
    // predecessor 只能是 pending(N-1) 或 exact(N-1)。任意其他值都不是
    // “再等一会儿”的乱序，而是必须立即 fail-closed 的协议损坏。
    state->claim_tournament[0].root.insert_completion.value = 7;
    LocalStats out_of_order_stats{};
    int64_t observed = INT64_MIN;
    uint64_t loads = 0U;
    const bool out_of_order = WaitForSharedTaskInsertTurn<GateOps>(
        state.get(), 1, out_of_order_stats, observed, loads
    );
    const bool out_of_order_failed = !out_of_order && observed == -1 &&
        loads == 0U && state->fatal.value != 0 &&
        GateOps::metadata_flushes[1].load(std::memory_order_acquire) == 0U &&
        GateOps::Load(
            &state->claim_tournament[1].root.insert_completion.value
        ) == SharedInsertCompletionInitialValue(1U);
    return duplicate_failed && out_of_order_failed;
}

}  // namespace

int main()
{
    const bool four_leader = RunFourLeaderOrdinaryWriterGate();
    const bool failures = TestDuplicateAndOutOfOrderFailClosed();
    if (!four_leader || !failures) {
        std::fprintf(
            stderr,
            "FAIL SIMT ordinary writer gate: four_leader=%d failures=%d\n",
            four_leader ? 1 : 0, failures ? 1 : 0
        );
        return 1;
    }
    std::puts(
        "PASS SIMT ordinary writer gate: 4 leaders, ordinary payload, "
        "writer<consumer lookup, zero-writer handoff, fail-closed ordering"
    );
    return 0;
}
