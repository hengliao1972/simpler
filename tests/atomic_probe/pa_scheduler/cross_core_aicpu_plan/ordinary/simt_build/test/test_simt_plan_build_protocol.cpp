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

#include "../common/simt_plan_build_protocol.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler::aicpu_plan;
using namespace pa_scheduler::aicpu_plan_simt;

constexpr uint32_t kSyntheticTensorCount = 5U;
constexpr uint32_t kSyntheticScalarCount = 3U;
constexpr uint32_t kSyntheticDependencyCount = 2U;
constexpr uint32_t kCorruptedTaskId = 1U;
constexpr auto kTestTimeout = std::chrono::seconds(30);

enum class CorruptionKind : uint8_t {
    None,
    Control,
    Payload,
};

struct AlignedFree {
    void operator()(RuntimeTaskPlanCell *pointer) const
    {
        std::free(pointer);
    }
};

using CellStorage = std::unique_ptr<RuntimeTaskPlanCell, AlignedFree>;

struct CpuPlanOps {
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

    static void StorePayloadWord(volatile uint64_t *address, uint64_t value)
    {
        *address = value;
    }

    static void FlushRegion(void *, uint64_t) {}
    static void InvalidateRegion(const void *, uint64_t) {}

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
};

TensorTag SyntheticTensorTag(uint32_t tensor)
{
    constexpr std::array<TensorTag, kSyntheticTensorCount> tags{
        TensorTag::Input,
        TensorTag::Output,
        TensorTag::Inout,
        TensorTag::OutputExisting,
        TensorTag::NoDependency,
    };
    return tags[tensor];
}

EngineClass EngineForTask(uint32_t task_id)
{
    constexpr std::array<EngineClass, 3> engines{
        EngineClass::Aic,
        EngineClass::Aiv,
        EngineClass::MetadataOnly,
    };
    return engines[task_id % engines.size()];
}

uint32_t FunctionForTask(uint32_t task_id)
{
    // 故意采用非周期散列，门槛不得从 task_id 的固定 PA 周期恢复语义。
    return EngineForTask(task_id) == EngineClass::MetadataOnly
        ? kInvalidFunctionId
        : 17U + ((task_id * 73U + 29U) % 30000U);
}

uint8_t AdapterFlagsForTask(uint32_t task_id)
{
    return static_cast<uint8_t>(0x80U | ((task_id * 11U) & 0x7FU));
}

uint16_t AdapterDataForTask(uint32_t task_id)
{
    return static_cast<uint16_t>((task_id * 257U + 91U) & 0xFFFFU);
}

int16_t CoreNumForTask(uint32_t task_id)
{
    return static_cast<int16_t>(1U + ((task_id * 5U) % 16U));
}

uint8_t RequireSyncForTask(uint32_t task_id)
{
    return static_cast<uint8_t>((task_id >> 1U) & 1U);
}

uint64_t TensorWordForTask(
    uint32_t task_id, uint32_t tensor, uint32_t word
)
{
    return 0x3C6EF372FE94F82BULL ^
           (static_cast<uint64_t>(task_id) << 21U) ^
           (static_cast<uint64_t>(tensor) << 12U) ^ word;
}

uint64_t ScalarForTask(uint32_t task_id, uint32_t scalar)
{
    return 0x6A09E667F3BCC909ULL ^
           (static_cast<uint64_t>(task_id) << 17U) ^ scalar;
}

uint64_t DependencyForTask(uint32_t task_id, uint32_t dependency)
{
    return 0xBB67AE8584CAA73BULL ^
           (static_cast<uint64_t>(task_id) << 19U) ^ dependency;
}

struct SyntheticPlanSource {
    uint32_t task_id;

    TensorTag TensorTagAt(uint32_t tensor) const
    {
        return SyntheticTensorTag(tensor);
    }

    bool TensorIsReference(uint32_t) const { return false; }

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const
    {
        return TensorWordForTask(task_id, tensor, word);
    }

    uint64_t Scalar(uint32_t scalar) const
    {
        return ScalarForTask(task_id, scalar);
    }

    uint64_t ExplicitDependency(uint32_t dependency) const
    {
        return DependencyForTask(task_id, dependency);
    }
};

struct PlanFixture {
    alignas(kAtomicIsolationBytes) RuntimePlanControl control{};
    RuntimePlanView view{};
    CellStorage cells;
    uint32_t capacity = 0U;

    explicit PlanFixture(uint32_t requested_capacity)
        : capacity(requested_capacity)
    {
        void *raw = nullptr;
        const size_t bytes = static_cast<size_t>(capacity) *
                             sizeof(RuntimeTaskPlanCell);
        if (posix_memalign(&raw, kAtomicIsolationBytes, bytes) != 0) {
            return;
        }
        std::memset(raw, 0, bytes);
        cells.reset(static_cast<RuntimeTaskPlanCell *>(raw));
        control.closed_task_count.value = kPlanOpenTaskCount;
        control.build_release.value = kBuildReleasePending;
        view = RuntimePlanView{&control, cells.get(), capacity};
    }

    bool Valid() const { return cells != nullptr; }
};

bool PublishSyntheticPlan(PlanFixture &fixture, uint32_t task_count)
{
    if (!fixture.Valid() || task_count > fixture.capacity) return false;
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        const SyntheticPlanSource source{task_id};
        const RuntimeTaskPlanSpec spec{
            task_id,
            FunctionForTask(task_id),
            kSyntheticTensorCount,
            kSyntheticScalarCount,
            kSyntheticDependencyCount,
            /*output_count=*/1U,
            EngineForTask(task_id),
            AdapterFlagsForTask(task_id),
            CoreNumForTask(task_id),
            RequireSyncForTask(task_id),
            /*reserved=*/0U,
            AdapterDataForTask(task_id),
            /*tensor_reference_mask=*/0U,
        };
        if (PublishRuntimeTaskPlan<CpuPlanOps>(fixture.view, spec, source) !=
                PlanPublishResult::Published ||
            !AdvancePlannedFrontier<CpuPlanOps>(fixture.view, task_id)) {
            return false;
        }
    }
    return CloseRuntimePlan<CpuPlanOps>(fixture.view, task_count);
}

bool ValidateCanonicalPlanFields(
    const PlanFixture &fixture, uint32_t task_id,
    const RuntimeTaskPlanHeader &header,
    const RuntimeTaskPlanLayout &layout
)
{
    if (header.task_id != task_id ||
        header.function_id != FunctionForTask(task_id) ||
        header.tensor_count != kSyntheticTensorCount ||
        header.scalar_count != kSyntheticScalarCount ||
        header.explicit_dep_count != kSyntheticDependencyCount ||
        header.output_count != 1U ||
        header.engine_class != static_cast<uint8_t>(EngineForTask(task_id)) ||
        header.adapter_flags != AdapterFlagsForTask(task_id) ||
        header.core_num != CoreNumForTask(task_id) ||
        header.require_sync_start != RequireSyncForTask(task_id) ||
        header.reserved0 != 0U ||
        header.adapter_data != AdapterDataForTask(task_id) ||
        header.tensor_reference_mask != 0U ||
        header.abi_version != kRuntimePlanAbiVersion) {
        return false;
    }
    for (uint32_t tensor = 0U; tensor < kSyntheticTensorCount; ++tensor) {
        if (header.tensor_tags[tensor] !=
            static_cast<uint8_t>(SyntheticTensorTag(tensor))) {
            return false;
        }
        uint32_t offset = 0U;
        if (!RuntimeTaskPlanTensorWordOffset(header, tensor, offset)) {
            return false;
        }
        const uint32_t meaningful = TensorMeaningfulWords(
            SyntheticTensorTag(tensor), /*reference=*/false
        );
        for (uint32_t word = 0U; word < kTensorCanonicalWords; ++word) {
            const uint64_t expected = word < meaningful
                ? TensorWordForTask(task_id, tensor, word)
                : 0U;
            if (fixture.cells.get()[task_id].payload.words[offset + word] !=
                expected) {
                return false;
            }
        }
    }
    for (uint32_t tensor = kSyntheticTensorCount;
         tensor < kMaxTaskTensors; ++tensor) {
        if (header.tensor_tags[tensor] != 0U) return false;
    }
    for (uint32_t scalar = 0U; scalar < kSyntheticScalarCount; ++scalar) {
        if (fixture.cells.get()[task_id].payload.words[
                layout.scalar_word_offset + scalar
            ] != ScalarForTask(task_id, scalar)) {
            return false;
        }
    }
    for (uint32_t dependency = 0U;
         dependency < kSyntheticDependencyCount; ++dependency) {
        if (fixture.cells.get()[task_id].payload.words[
                layout.explicit_dep_word_offset + dependency
            ] != DependencyForTask(task_id, dependency)) {
            return false;
        }
    }
    return true;
}

void CorruptPublishedCell(PlanFixture &fixture, CorruptionKind corruption)
{
    if (corruption == CorruptionKind::Control) {
        fixture.cells.get()[kCorruptedTaskId].control.value =
            static_cast<int64_t>(
                static_cast<uint64_t>(
                    fixture.cells.get()[kCorruptedTaskId].control.value
                ) | (uint64_t{1} << 63U)
            );
    } else if (corruption == CorruptionKind::Payload) {
        volatile uint64_t &word =
            fixture.cells.get()[kCorruptedTaskId].payload.words[2];
        word = (word & ~uint64_t{0xFFU}) | uint64_t{0xFFU};
    }
}

void PublishFatal(PlanFixture &fixture)
{
    CpuPlanOps::PublishControl(&fixture.control.fatal.value, 1);
}

template <typename Predicate>
bool WaitUntil(
    const std::chrono::steady_clock::time_point &deadline,
    Predicate predicate
)
{
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

bool TestLocalLeaderIdentityContract()
{
    BuilderLeaderLocalState valid{};
    BuilderLeaderLocalState invalid{};
    return BuilderLeaderIdValid(0U) &&
           BuilderLeaderIdValid(kBuilderLeaders - 1U) &&
           !BuilderLeaderIdValid(kBuilderLeaders) &&
           RecordBuilderLeaderArrivalLocally(0U, valid) ==
               LocalLeaderArrivalStatus::First &&
           RecordBuilderLeaderArrivalLocally(0U, valid) ==
               LocalLeaderArrivalStatus::Duplicate &&
           RecordBuilderLeaderArrivalLocally(kBuilderLeaders, invalid) ==
               LocalLeaderArrivalStatus::InvalidLeader &&
           !invalid.arrival_recorded;
}

bool RunFourLeaderBuild(
    uint32_t task_count, bool delay_low_predecessor,
    CorruptionKind corruption
)
{
    const bool expect_fatal = corruption != CorruptionKind::None;
    if (expect_fatal && task_count <= kCorruptedTaskId) return false;

    PlanFixture fixture(task_count == 0U ? 1U : task_count);
    if (!PublishSyntheticPlan(fixture, task_count)) return false;

    // Attach 只执行一次。四个 leader 之后只使用缓存的 N，每票
    // 热路只保留 build_next FetchAdd。
    uint32_t attached_task_count = UINT32_MAX;
    if (!AttachClosedPlan<CpuPlanOps>(
            fixture.view, attached_task_count
        ) || attached_task_count != task_count) {
        return false;
    }
    CorruptPublishedCell(fixture, corruption);

    std::vector<std::atomic<uint32_t>> visits(task_count);
    std::vector<std::atomic<int64_t>> completion(task_count);
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        visits[task_id].store(0U, std::memory_order_relaxed);
        completion[task_id].store(
            task_id == 0U ? -1 : static_cast<int64_t>(task_id) - 1,
            std::memory_order_relaxed
        );
    }

    std::array<std::atomic<uint32_t>, kBuilderLeaders> participations{};
    std::array<std::atomic<uint32_t>, kBuilderLeaders> first_tickets{};
    std::array<std::atomic<uint32_t>, kBuilderLeaders> arrival_calls{};
    std::array<BuilderLeaderLocalState, kBuilderLeaders> local_states{};
    std::atomic<bool> protocol_ok{true};
    std::atomic<bool> start{false};
    std::atomic<bool> process_first_ticket{false};
    std::atomic<bool> release_task_zero{!delay_low_predecessor};
    std::atomic<bool> task_zero_blocked{false};
    std::atomic<uint32_t> leaders_ready{0U};
    std::atomic<uint32_t> first_ticket_total{0U};
    std::atomic<uint32_t> predecessor_waiters{0U};
    std::atomic<uint32_t> last_arrivals{0U};
    std::atomic<uint32_t> rejected_control{0U};
    std::atomic<uint32_t> rejected_payload{0U};
    const auto deadline = std::chrono::steady_clock::now() + kTestTimeout;

    const auto fail_protocol = [&] {
        protocol_ok.store(false, std::memory_order_release);
        PublishFatal(fixture);
        release_task_zero.store(true, std::memory_order_release);
        process_first_ticket.store(true, std::memory_order_release);
        start.store(true, std::memory_order_release);
    };

    const auto wait_for_completion = [&](uint32_t predecessor) {
        while (completion[predecessor].load(std::memory_order_acquire) !=
               static_cast<int64_t>(predecessor)) {
            if (!protocol_ok.load(std::memory_order_acquire) ||
                CpuPlanOps::LoadControl(&fixture.control.fatal.value) != 0) {
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                fail_protocol();
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    };

    std::array<std::thread, kBuilderLeaders> leaders;
    for (uint32_t leader = 0U; leader < kBuilderLeaders; ++leader) {
        leaders[leader] = std::thread([&, leader] {
            if (!IsBuilderLeader(leader * kWarpSize) ||
                BuilderLeaderId(leader * kWarpSize) != leader ||
                participations[leader].fetch_add(
                    1U, std::memory_order_acq_rel
                ) != 0U) {
                fail_protocol();
                return;
            }
            leaders_ready.fetch_add(1U, std::memory_order_release);
            if (!WaitUntil(deadline, [&] {
                    return start.load(std::memory_order_acquire) ||
                           !protocol_ok.load(std::memory_order_acquire);
                })) {
                fail_protocol();
                return;
            }

            BuildReservation reservation =
                TakeAttachedBuildTicket<CpuPlanOps>(
                    fixture.view, attached_task_count
                );
            first_tickets[leader].fetch_add(1U, std::memory_order_acq_rel);
            first_ticket_total.fetch_add(1U, std::memory_order_release);
            if (!WaitUntil(deadline, [&] {
                    return process_first_ticket.load(std::memory_order_acquire) ||
                           !protocol_ok.load(std::memory_order_acquire);
                })) {
                fail_protocol();
                return;
            }

            while (protocol_ok.load(std::memory_order_acquire)) {
                if (reservation.status == BuildReservationStatus::Closed) {
                    // hot ticket 不读 fatal；在 arrival 边界统一观察。
                    if (CpuPlanOps::LoadControl(
                            &fixture.control.fatal.value
                        ) != 0) {
                        return;
                    }
                    arrival_calls[leader].fetch_add(
                        1U, std::memory_order_acq_rel
                    );
                    const BuildArrivalStatus arrival =
                        ArriveBuilderLeaderOnce<CpuPlanOps>(
                            fixture.view, leader, local_states[leader]
                        );
                    if (arrival == BuildArrivalStatus::Last) {
                        last_arrivals.fetch_add(
                            1U, std::memory_order_acq_rel
                        );
                        const bool final_insert_ok = task_count == 0U ||
                            completion[task_count - 1U].load(
                                std::memory_order_acquire
                            ) == static_cast<int64_t>(task_count - 1U);
                        if (!final_insert_ok ||
                            !PublishBuildRelease<CpuPlanOps>(
                                fixture.view, task_count
                            )) {
                            fail_protocol();
                        }
                    } else if (arrival != BuildArrivalStatus::Arrived) {
                        fail_protocol();
                    }
                    return;
                }
                if (reservation.status != BuildReservationStatus::Reserved ||
                    reservation.task_id >= task_count) {
                    fail_protocol();
                    return;
                }

                RuntimeTaskPlanHeader header{};
                RuntimeTaskPlanLayout layout{};
                const PlanAcquireResult acquire =
                    AcquireRuntimeTaskPlan<CpuPlanOps>(
                        fixture.view, reservation.task_id, header, layout
                    );
                if (acquire != PlanAcquireResult::Acquired) {
                    if (expect_fatal &&
                        acquire == PlanAcquireResult::FatalObserved &&
                        CpuPlanOps::LoadControl(
                            &fixture.control.fatal.value
                        ) != 0) {
                        return;
                    }
                    const bool expected_control =
                        corruption == CorruptionKind::Control &&
                        reservation.task_id == kCorruptedTaskId &&
                        acquire == PlanAcquireResult::InvalidControl;
                    const bool expected_payload =
                        corruption == CorruptionKind::Payload &&
                        reservation.task_id == kCorruptedTaskId &&
                        acquire == PlanAcquireResult::InvalidPayload;
                    if (expected_control) {
                        rejected_control.fetch_add(
                            1U, std::memory_order_acq_rel
                        );
                        PublishFatal(fixture);
                    } else if (expected_payload) {
                        rejected_payload.fetch_add(
                            1U, std::memory_order_acq_rel
                        );
                        PublishFatal(fixture);
                    } else {
                        fail_protocol();
                    }
                    return;
                }
                if (!ValidateCanonicalPlanFields(
                        fixture, reservation.task_id, header, layout
                    ) || visits[reservation.task_id].fetch_add(
                        1U, std::memory_order_acq_rel
                    ) != 0U) {
                    fail_protocol();
                    return;
                }

                if (reservation.task_id == 0U && delay_low_predecessor) {
                    task_zero_blocked.store(true, std::memory_order_release);
                    if (!WaitUntil(deadline, [&] {
                            return release_task_zero.load(
                                       std::memory_order_acquire
                                   ) ||
                                   CpuPlanOps::LoadControl(
                                       &fixture.control.fatal.value
                                   ) != 0;
                        })) {
                        fail_protocol();
                        return;
                    }
                    if (CpuPlanOps::LoadControl(
                            &fixture.control.fatal.value
                        ) != 0) {
                        return;
                    }
                }

                if (reservation.task_id != 0U) {
                    predecessor_waiters.fetch_add(
                        1U, std::memory_order_acq_rel
                    );
                    if (!wait_for_completion(reservation.task_id - 1U)) {
                        return;
                    }
                }
                int64_t expected = reservation.task_id == 0U
                    ? -1
                    : static_cast<int64_t>(reservation.task_id) - 1;
                if (!completion[reservation.task_id].compare_exchange_strong(
                        expected, static_cast<int64_t>(reservation.task_id),
                        std::memory_order_acq_rel
                    )) {
                    fail_protocol();
                    return;
                }
                reservation = TakeAttachedBuildTicket<CpuPlanOps>(
                    fixture.view, attached_task_count
                );
            }
        });
    }

    if (!WaitUntil(deadline, [&] {
            return leaders_ready.load(std::memory_order_acquire) ==
                   kBuilderLeaders;
        })) {
        fail_protocol();
    }
    start.store(true, std::memory_order_release);
    if (!WaitUntil(deadline, [&] {
            return first_ticket_total.load(std::memory_order_acquire) ==
                   kBuilderLeaders;
        })) {
        fail_protocol();
    }
    process_first_ticket.store(true, std::memory_order_release);

    if (delay_low_predecessor) {
        const uint32_t expected_waiters = task_count == 0U
            ? 0U
            : (task_count - 1U < kBuilderLeaders - 1U
                   ? task_count - 1U
                   : kBuilderLeaders - 1U);
        if (!WaitUntil(deadline, [&] {
                return task_zero_blocked.load(std::memory_order_acquire) &&
                       predecessor_waiters.load(std::memory_order_acquire) >=
                           expected_waiters;
            })) {
            fail_protocol();
        }
        // task0 未发布 completion 时，其他 leader 即使已取得高 task
        // 也不得提前 arrival/release。
        if (task_count == 0U ||
            completion[0U].load(std::memory_order_acquire) != -1 ||
            CpuPlanOps::LoadControl(
                &fixture.control.build_workers_done.value
            ) != 0 ||
            CpuPlanOps::LoadControl(
                &fixture.control.build_release.value
            ) != kBuildReleasePending) {
            fail_protocol();
        }
        release_task_zero.store(true, std::memory_order_release);
    }

    for (std::thread &leader : leaders) leader.join();

    for (uint32_t leader = 0U; leader < kBuilderLeaders; ++leader) {
        if (participations[leader].load(std::memory_order_relaxed) != 1U ||
            first_tickets[leader].load(std::memory_order_relaxed) != 1U) {
            return false;
        }
    }

    if (expect_fatal) {
        const uint32_t expected_control =
            corruption == CorruptionKind::Control ? 1U : 0U;
        const uint32_t expected_payload =
            corruption == CorruptionKind::Payload ? 1U : 0U;
        if (!protocol_ok.load(std::memory_order_relaxed) ||
            rejected_control.load(std::memory_order_relaxed) !=
                expected_control ||
            rejected_payload.load(std::memory_order_relaxed) !=
                expected_payload ||
            CpuPlanOps::LoadControl(&fixture.control.fatal.value) == 0 ||
            CpuPlanOps::LoadControl(
                &fixture.control.build_release.value
            ) != kBuildReleasePending ||
            last_arrivals.load(std::memory_order_relaxed) != 0U) {
            return false;
        }
        for (uint32_t leader = 0U; leader < kBuilderLeaders; ++leader) {
            if (arrival_calls[leader].load(std::memory_order_relaxed) != 0U ||
                local_states[leader].arrival_recorded) {
                return false;
            }
        }
        return true;
    }

    if (!protocol_ok.load(std::memory_order_relaxed) ||
        CpuPlanOps::LoadControl(&fixture.control.fatal.value) != 0 ||
        last_arrivals.load(std::memory_order_relaxed) != 1U ||
        CpuPlanOps::LoadControl(&fixture.control.build_next.value) !=
            static_cast<int64_t>(ExpectedBuildTicketCount(task_count)) ||
        CpuPlanOps::LoadControl(
            &fixture.control.build_workers_done.value
        ) != static_cast<int64_t>(kBuilderLeaders) ||
        CpuPlanOps::LoadControl(&fixture.control.build_release.value) !=
            static_cast<int64_t>(task_count)) {
        return false;
    }
    for (uint32_t leader = 0U; leader < kBuilderLeaders; ++leader) {
        if (arrival_calls[leader].load(std::memory_order_relaxed) != 1U ||
            !local_states[leader].arrival_recorded) {
            return false;
        }
    }
    for (uint32_t task_id = 0U; task_id < task_count; ++task_id) {
        if (visits[task_id].load(std::memory_order_relaxed) != 1U ||
            completion[task_id].load(std::memory_order_relaxed) !=
                static_cast<int64_t>(task_id)) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main()
{
    if (!TestLocalLeaderIdentityContract()) {
        std::cerr << "FAIL local leader identity/once contract\n";
        return 1;
    }
    std::cout << "PASS local leader identity/once contract\n";

    const std::array<uint32_t, 8> task_counts{
        0U, 1U, 3U, 4U, 5U, 41U, 257U, 1280U
    };
    for (uint32_t task_count : task_counts) {
        const bool delay_low_predecessor = task_count == 41U;
        if (!RunFourLeaderBuild(
                task_count, delay_low_predecessor,
                CorruptionKind::None
            )) {
            std::cerr << "FAIL four-leader closed Plan: N="
                      << task_count << '\n';
            return 1;
        }
        std::cout << "PASS four-leader closed Plan: N="
                  << task_count << " tickets="
                  << ExpectedBuildTicketCount(task_count)
                  << (delay_low_predecessor ? " delayed-task0" : "")
                  << '\n';
    }

    for (CorruptionKind corruption : {
             CorruptionKind::Control, CorruptionKind::Payload
         }) {
        if (!RunFourLeaderBuild(
                /*task_count=*/8U, /*delay_low_predecessor=*/false,
                corruption
            )) {
            std::cerr << "FAIL malformed full-leader convergence: "
                      << (corruption == CorruptionKind::Control
                              ? "control"
                              : "payload")
                      << '\n';
            return 1;
        }
        std::cout << "PASS malformed full-leader fatal/no-release: "
                  << (corruption == CorruptionKind::Control
                          ? "control"
                          : "payload")
                  << '\n';
    }
    return 0;
}
