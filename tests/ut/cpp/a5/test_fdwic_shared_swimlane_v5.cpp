/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE file.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <type_traits>

#include "inner_kernel.h"
#include "runtime.h"
#include "dist_engine/common/swimlane.h"
#include "dist_engine/common/runtime_state.h"
#include "dist_engine/aicore/tensor_map.h"
#include "dist_engine/aicore/core_state.h"
#include "dist_engine/aicore/run_state.h"
#include "dist_engine/aicore/onboard_entry.h"
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)

Runtime::Runtime() { std::memset(this, 0, sizeof(*this)); }
[[noreturn]] void assert_impl(const char *, const char *, int) { std::abort(); }
volatile uint8_t *sim_get_reg_base() { return nullptr; }
extern "C" void aicpu_orchestration_entry(const L2TaskArgs &) {}

extern "C" int fdwic_swimlane_host_init(Runtime *runtime, int num_cores, int level, const char *output_prefix);
extern "C" int fdwic_swimlane_host_export(Runtime *runtime);
extern "C" void fdwic_swimlane_host_finalize(Runtime *runtime);

namespace {

static_assert(PTO_FDWIC_SHARED_MAP == 1, "this target locks the shared trace ABI");
static_assert(kFdwicSwimlaneVersion == 5);
static_assert(kFdwicSwimlaneTraceSchemaVersion == 5);
static_assert(sizeof(FdwicCompactSwimlaneRecord) == 16);
static_assert(sizeof(FdwicSharedSubmitClaimRecord) == 32);
static_assert(sizeof(FdwicSwimlaneRecord) == 32);
static_assert(!std::is_same_v<FdwicSwimlaneStorageRecord, FdwicSwimlaneRecord>);
static_assert(kFdwicSwimlaneWorkerBytes == 593920);
static_assert(static_cast<uint32_t>(FdwicSwimlanePhase::Dcci) == 24);
static_assert(static_cast<uint32_t>(FdwicSwimlanePhase::SharedRegisterWaitInsertTurnBypassLoad) == 25);
static_assert(static_cast<uint32_t>(FdwicSwimlanePhase::Count) == 26);
static_assert(static_cast<uint32_t>(FdwicAtomicSite::SharedInsertTurnPoll) == 19);
static_assert(static_cast<uint32_t>(FdwicAtomicSite::SharedInsertTurnHandoff) == 20);
static_assert(static_cast<uint32_t>(FdwicAtomicSite::SharedClaimTournamentLocal) == 40);
static_assert(static_cast<uint32_t>(FdwicAtomicSite::SharedClaimTournamentRoot) == 41);
static_assert(static_cast<uint32_t>(FdwicAtomicSite::Count) == 42);
static_assert(static_cast<uint32_t>(FdwicAtomicOp::CompareExchange) == 4);

class FdwicSharedSwimlaneV5Test : public ::testing::Test {
protected:
    void SetUp() override {
        storage_ = std::aligned_alloc(64, kFdwicSwimlaneWorkerBytes);
        ASSERT_NE(storage_, nullptr);
        std::memset(storage_, 0, kFdwicSwimlaneWorkerBytes);
        self_ = {};
        self_.core_idx = 0;
        self_.block_id = 0;
        self_.lane = 0;
        core_ = {};
        g_self = &self_;
        g_fdwic_swimlane_level = kFdwicAtomicSwimlaneLevel;
        g_fdwic_swimlane_core = &core_;
        g_fdwic_swimlane_records = reinterpret_cast<FdwicSwimlaneStorageRecord *>(
            static_cast<uint8_t *>(storage_) + kFdwicSharedSubmitClaimBytesPerCore
        );
        g_fdwic_swimlane_records_per_core = kFdwicSwimlaneDefaultRecordsPerCore;
        g_fdwic_swimlane_record_count = 0;
        g_fdwic_swimlane_dropped_records = 0;
        g_fdwic_swimlane_shared_submit_count = 0;
        g_fdwic_atomic_calls = 0;
        g_fdwic_poll_calls = 0;
        g_fdwic_poll_batch_records = 0;
        g_fdwic_atomic_counter_overflow = false;
        g_fdwic_dcci_calls = 0;
        g_fdwic_dcci_lines = 0;
        g_fdwic_dcci_records = 0;
        g_fdwic_dcci_counter_overflow = false;
    }

    void TearDown() override {
        g_self = nullptr;
        g_dist_ptr = nullptr;
        g_fdwic_swimlane_level = 0;
        g_fdwic_swimlane_core = nullptr;
        g_fdwic_swimlane_records = nullptr;
        g_fdwic_swimlane_records_per_core = 0;
        std::free(storage_);
    }

    FdwicSharedSubmitClaimRecord *endpoints() { return reinterpret_cast<FdwicSharedSubmitClaimRecord *>(storage_); }

    FdwicCompactSwimlaneRecord *generic() {
        return reinterpret_cast<FdwicCompactSwimlaneRecord *>(
            static_cast<uint8_t *>(storage_) + kFdwicSharedSubmitClaimBytesPerCore
        );
    }

    void *storage_ = nullptr;
    DistCore self_{};
    FdwicSwimlaneCoreState core_{};
};

void *host_test_device_malloc(size_t size) { return std::calloc(1, size); }

void host_test_device_free(void *pointer) { std::free(pointer); }

int host_test_copy_to_device(void *device, const void *host, size_t size) {
    if (device == nullptr || host == nullptr) return -1;
    std::memcpy(device, host, size);
    return 0;
}

int host_test_copy_from_device(void *host, const void *device, size_t size) {
    if (device == nullptr || host == nullptr) return -1;
    std::memcpy(host, device, size);
    return 0;
}

class ScopedTemporaryDirectory {
public:
    ScopedTemporaryDirectory() {
        char path[] = "/tmp/fdwic-shared-v5-XXXXXX";
        const char *created = ::mkdtemp(path);
        if (created != nullptr) path_ = created;
    }

    ~ScopedTemporaryDirectory() {
        std::error_code error;
        if (!path_.empty()) std::filesystem::remove_all(path_, error);
    }

    const std::string &path() const { return path_; }

private:
    std::string path_;
};

bool shared_trace_task_winner(uint32_t core, uint32_t task_id) {
    switch (task_id % 5U) {
    case 0:
        return core == 0;
    case 1:
    case 3:
        return core == 0;
    case 2:
        return core == 32;
    case 4:
        return core == 32;
    default:
        return false;
    }
}

int32_t shared_trace_task_func(uint32_t task_id) {
    const uint32_t kind = task_id % 5U;
    return kind == 0 ? -1 : static_cast<int32_t>(kind - 1U);
}

bool shared_trace_task_attempted(uint32_t core, uint32_t task_id) {
    const uint32_t kind = task_id % 5U;
    if (kind == 0) {
        return core < kFdwicSharedWorkers;
    }
    return core < 32U ? kind == 1U || kind == 3U : kind == 2U || kind == 4U;
}

bool shared_trace_task_root_contender(uint32_t core, uint32_t task_id) {
    const uint32_t kind = task_id % 5U;
    if (!shared_trace_task_attempted(core, task_id)) return false;
    if (kind == 0) return core < kFdwicSharedAllocClaimTournamentGroups;
    if (kind == 1 || kind == 3) return core < kFdwicSharedAicClaimTournamentGroups;
    return core >= kFdwicSharedAicWorkers &&
           core < kFdwicSharedAicWorkers + kFdwicSharedAivClaimTournamentGroups;
}

enum class SharedTraceFault {
    None,
    MissingTournamentLocal,
    ExtraTournamentLocal,
    MissingTournamentRoot,
    ExtraTournamentRoot,
    MissingDcci,
    WrongDcciLines,
};

SharedTraceFault shared_trace_task_fault(SharedTraceFault fault, uint32_t core, uint32_t task_id) {
    if (fault == SharedTraceFault::ExtraTournamentLocal) {
        return core == 32 && task_id == 1 ? fault : SharedTraceFault::None;
    }
    if (fault == SharedTraceFault::ExtraTournamentRoot) {
        return core == 6 && task_id == 1 ? fault : SharedTraceFault::None;
    }
    return core == 0 && task_id == 1 ? fault : SharedTraceFault::None;
}

bool write_shared_generic_span(
    DistCore *self, int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase, uint64_t begin, uint64_t end,
    uint32_t aux = 0
) {
    return fdwic_swimlane_detail_write_record(self, task_id, func_id, phase, begin, end, /*flags=*/0, aux);
}

bool write_shared_business_dcci(
    DistCore *self, uint32_t task_id, uint64_t submit_begin, SharedTraceFault fault
) {
    uint64_t offset = 25U;
    const auto write = [&](FdwicDcciSite site, uint32_t lines) {
        const uint64_t begin = submit_begin + offset++;
        return fdwic_swimlane_record_dcci(
            self, static_cast<int32_t>(task_id), -1, site, /*trailing_dsb=*/true, lines, begin, begin + 1U
        );
    };
    switch (task_id % 5U) {
    case 0:
        return write(FdwicDcciSite::SharedOutputDescriptorFlush, 6);
    case 1:
        if (fault == SharedTraceFault::MissingDcci) return true;
        return write(
            FdwicDcciSite::SharedOutputDescriptorFlush,
            fault == SharedTraceFault::WrongDcciLines ? 3U : 2U
        );
    case 2:
        return write(FdwicDcciSite::SharedOutputDescriptorFlush, 6) &&
               write(FdwicDcciSite::SharedWinnerBuildDescriptorInvalidate, 2);
    case 3:
        return write(FdwicDcciSite::SharedOutputDescriptorFlush, 2) &&
               write(FdwicDcciSite::SharedWinnerBuildDescriptorInvalidate, 2);
    case 4:
        if (!write(FdwicDcciSite::SharedWriterHistoryFlush, 1)) return false;
        for (uint32_t record = 0; record < 3; ++record) {
            if (!write(FdwicDcciSite::SharedFaninHistoryInvalidate, 1)) return false;
        }
        for (uint32_t record = 0; record < 6; ++record) {
            if (!write(FdwicDcciSite::SharedWinnerBuildDescriptorInvalidate, 2)) return false;
        }
        return true;
    default:
        return false;
    }
}

bool populate_shared_trace_core(Runtime *runtime, uint32_t core, SharedTraceFault fault = SharedTraceFault::None) {
    fdwic_swimlane_attach(runtime);
    auto self_storage = std::make_unique<DistCore>();
    DistCore *self = self_storage.get();
    self->core_idx = static_cast<int32_t>(core);
    if (core < 32) {
        self->block_id = static_cast<int32_t>(core);
        self->lane = 0;
    } else {
        const uint32_t aiv_ordinal = core - 32U;
        self->block_id = static_cast<int32_t>(aiv_ordinal / 2U);
        self->lane = 1 + static_cast<int32_t>(aiv_ordinal % 2U);
    }
    g_self = self;
    fdwic_swimlane_reset_core(self);
    if (g_fdwic_swimlane_core == nullptr || g_fdwic_swimlane_records == nullptr) return false;

    const uint64_t base = fdwic_swimlane_detail_now();
    const bool detailed = runtime->dist.swimlane_level >= kFdwicAtomicSwimlaneLevel;
    if (detailed &&
        (!fdwic_swimlane_record_dcci(
             self, -1, -1, FdwicDcciSite::StartupConfigInvalidate, FdwicDcciOp::Invalidate,
             /*trailing_dsb=*/true, /*call_count=*/1, /*line_count=*/1, base + 10U, base + 11U
         ) ||
         !fdwic_swimlane_detail_write_record(
             self, -1, -1, FdwicSwimlanePhase::ClockBaseline, base + 20U, base + 21U, /*flags=*/0, /*aux=*/0
         ) ||
         !fdwic_swimlane_detail_write_record(
             self, -1, -1, FdwicSwimlanePhase::ClockBaseline, base + 22U, base + 23U, kFdwicClockAtomicDependency,
             /*aux=*/0
         ))) {
        return false;
    }

    for (uint32_t task_id = 0; task_id < kFdwicSharedTracePhase1TaskCount; ++task_id) {
        const bool winner = shared_trace_task_winner(core, task_id);
        const bool attempted = shared_trace_task_attempted(core, task_id);
        const SharedTraceFault task_fault = shared_trace_task_fault(fault, core, task_id);
        const int32_t func_id = winner ? shared_trace_task_func(task_id) : -1;
        const uint64_t submit_begin = base + 100U + static_cast<uint64_t>(task_id) * 100U;
        const uint64_t claim_begin = submit_begin + 5U;
        const uint64_t claim_end = submit_begin + 10U;
        const uint64_t submit_end = submit_begin + 90U;
        if (!fdwic_swimlane_record_shared_claim(self, task_id, claim_begin, claim_end, winner) ||
            !fdwic_swimlane_record_shared_submit(self, task_id, submit_begin, submit_end)) {
            return false;
        }
        bool write_local = attempted;
        if (task_fault == SharedTraceFault::MissingTournamentLocal) write_local = false;
        if (task_fault == SharedTraceFault::ExtraTournamentLocal) write_local = true;
        if (detailed && write_local &&
            !fdwic_swimlane_record_captured_atomic(
                static_cast<int32_t>(task_id), FdwicAtomicSite::SharedClaimTournamentLocal,
                FdwicAtomicOp::CompareExchange, claim_begin + 1U, claim_begin + 2U,
                /*result_used=*/true, /*return_ready=*/false
            )) {
            return false;
        }
        bool write_root = shared_trace_task_root_contender(core, task_id);
        if (task_fault == SharedTraceFault::MissingTournamentRoot) write_root = false;
        if (task_fault == SharedTraceFault::ExtraTournamentRoot) write_root = true;
        if (detailed && write_root &&
            !fdwic_swimlane_record_captured_atomic(
                static_cast<int32_t>(task_id), FdwicAtomicSite::SharedClaimTournamentRoot,
                FdwicAtomicOp::CompareExchange, claim_begin + 3U, claim_begin + 4U,
                /*result_used=*/true, /*return_ready=*/false
            )) {
            return false;
        }
        if (!winner) continue;

        if (!write_shared_generic_span(
                self, task_id, func_id, FdwicSwimlanePhase::Materialize, submit_begin + 12U, submit_begin + 35U
            ) ||
            !write_shared_generic_span(
                self, task_id, func_id, FdwicSwimlanePhase::SharedMaterializePublishTaskOutputs, submit_begin + 13U,
                submit_begin + 24U
            ) ||
            !write_shared_generic_span(
                self, task_id, func_id, FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsCopy, submit_begin + 13U,
                submit_begin + 18U
            ) ||
            !write_shared_generic_span(
                self, task_id, func_id, FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsFlush,
                submit_begin + 18U, submit_begin + 24U
            ) ||
            !write_shared_generic_span(
                self, task_id, func_id, FdwicSwimlanePhase::Register, submit_begin + 35U, submit_begin + 65U
            ) ||
            !write_shared_generic_span(
                self, task_id, func_id, FdwicSwimlanePhase::SharedRegisterWaitInsertTurnBypassLoad,
                submit_begin + 35U, submit_begin + 40U, task_id == 0 ? 0U : 1U
            )) {
            return false;
        }
        if (!write_shared_generic_span(
                self, task_id, func_id, FdwicSwimlanePhase::SharedRegisterPublishMetadata, submit_begin + 40U,
                submit_begin + 50U
            )) {
            return false;
        }
        if (detailed &&
            !fdwic_swimlane_record_captured_atomic(
                static_cast<int32_t>(task_id), FdwicAtomicSite::SharedInsertTurnHandoff, FdwicAtomicOp::CompareExchange,
                submit_begin + 55U, submit_begin + 60U,
                /*result_used=*/true, /*return_ready=*/false
            )) {
            return false;
        }
        if (task_id % 5U == 0) {
            if (!write_shared_generic_span(
                    self, task_id, func_id, FdwicSwimlanePhase::AllocComplete, submit_begin + 65U, submit_begin + 72U
                )) {
                return false;
            }
        } else if (!write_shared_generic_span(
                       self, task_id, func_id, FdwicSwimlanePhase::Fanin, submit_begin + 65U, submit_begin + 69U
                   ) ||
                   !write_shared_generic_span(
                       self, task_id, func_id, FdwicSwimlanePhase::WinnerBuild, submit_begin + 69U, submit_begin + 72U
                   )) {
            return false;
        }
        if (detailed && !write_shared_business_dcci(self, task_id, submit_begin, task_fault)) return false;
    }

    const uint64_t replay_begin = base + 90U;
    const uint64_t replay_end =
        base + 100U + static_cast<uint64_t>(kFdwicSharedTracePhase1TaskCount - 1U) * 100U + 100U;
    if (!write_shared_generic_span(self, -1, -1, FdwicSwimlanePhase::OrchestrationReplay, replay_begin, replay_end) ||
        !write_shared_generic_span(self, -1, -1, FdwicSwimlanePhase::FinalDrain, replay_end, replay_end + 20U)) {
        return false;
    }

    while (fdwic_swimlane_detail_now() < replay_end + 20U) {}
    fdwic_swimlane_flush_core(self);
    const FdwicSwimlaneCoreState *state = g_fdwic_swimlane_core;
    const bool complete = g_fdwic_swimlane_shared_submit_count == kFdwicSharedTracePhase1TaskCount &&
                          g_fdwic_swimlane_dropped_records == 0 && !g_fdwic_atomic_counter_overflow &&
                          !g_fdwic_dcci_counter_overflow && state->count == g_fdwic_swimlane_record_count &&
                          state->atomic_calls == g_fdwic_atomic_calls && state->poll_calls == g_fdwic_poll_calls &&
                          state->poll_batch_records == g_fdwic_poll_batch_records &&
                          state->dcci_calls == g_fdwic_dcci_calls && state->dcci_lines == g_fdwic_dcci_lines &&
                          state->dcci_records == g_fdwic_dcci_records;
    g_self = nullptr;
    return complete;
}

void configure_shared_host_runtime(Runtime &runtime) {
    runtime.host_api.device_malloc = host_test_device_malloc;
    runtime.host_api.device_free = host_test_device_free;
    runtime.host_api.copy_to_device = host_test_copy_to_device;
    runtime.host_api.copy_from_device = host_test_copy_from_device;
    runtime.worker_count = 96;
    runtime.dist.num_workers = 96;
    for (uint32_t core = 0; core < 96; ++core) {
        runtime.workers[core].core_type = core < 32 ? CoreType::AIC : CoreType::AIV;
    }
}

bool populate_shared_trace(Runtime &runtime, SharedTraceFault fault = SharedTraceFault::None) {
    for (uint32_t core = 0; core < 96; ++core) {
        if (!populate_shared_trace_core(&runtime, core, fault)) return false;
    }
    fdwic_swimlane_attach(nullptr);
    g_self = nullptr;
    return true;
}

void expect_shared_trace_fault_rejected(SharedTraceFault fault) {
    ScopedTemporaryDirectory output;
    ASSERT_FALSE(output.path().empty());

    Runtime runtime;
    configure_shared_host_runtime(runtime);
    ASSERT_EQ(fdwic_swimlane_host_init(&runtime, 96, kFdwicAtomicSwimlaneLevel, output.path().c_str()), 1);
    ASSERT_TRUE(populate_shared_trace(runtime, fault));
    EXPECT_NE(fdwic_swimlane_host_export(&runtime), 0);
    fdwic_swimlane_host_finalize(&runtime);
}

TEST_F(FdwicSharedSwimlaneV5Test, SubmitClaimUseFixedFullWidthAreaAndGenericRowsStayCompact) {
    fdwic_swimlane_detail_record(
        &self_, 17, 2, FdwicSwimlanePhase::Claim, 100, 120, kFdwicClaimWon | kFdwicClaimAttempted
    );
    fdwic_swimlane_detail_record(&self_, 17, 2, FdwicSwimlanePhase::Submit, 90, 200, kFdwicClaimWon);

    EXPECT_EQ(g_fdwic_swimlane_record_count, 0U);
    EXPECT_EQ(g_fdwic_swimlane_shared_submit_count, 18U);
    EXPECT_EQ(endpoints()[17].claim_begin, 100U);
    EXPECT_EQ(endpoints()[17].claim_end_and_winner, 120U | kFdwicSharedClaimWinnerBit);
    EXPECT_EQ(endpoints()[17].submit_begin, 90U);
    EXPECT_EQ(endpoints()[17].submit_end, 200U);

    fdwic_swimlane_detail_record(&self_, 17, 2, FdwicSwimlanePhase::Materialize, 121, 140, 0, 3);
    ASSERT_EQ(g_fdwic_swimlane_record_count, 1U);
    EXPECT_EQ(generic()[0].start_cycle_low, 121U);
    EXPECT_EQ(generic()[0].end_cycle_low, 140U);
    EXPECT_EQ(generic()[0].flags, 0U);
    EXPECT_EQ(generic()[0].packed & kFdwicCompactTraceTaskMask, 17U);
    EXPECT_EQ((generic()[0].packed >> kFdwicCompactTraceFunctionShift) & kFdwicCompactTraceFunctionMask, 2U);
    EXPECT_EQ(
        (generic()[0].packed >> kFdwicCompactTracePhaseShift) & kFdwicCompactTracePhaseMask,
        static_cast<uint32_t>(FdwicSwimlanePhase::Materialize)
    );
    EXPECT_EQ((generic()[0].packed >> kFdwicCompactTraceAuxShift) & kFdwicCompactTraceAuxMask, 3U);
}

TEST_F(FdwicSharedSwimlaneV5Test, ProductionSharedBeginLeavesEfDrainOutOfGenericRows) {
    g_dist_ptr = &g_dist_fallback;
    g_dist.fatal = 0;
    g_dist.error_code = PTO2_ERROR_NONE;
    // Pretend this local group and the root already selected task 0 so Begin
    // also exercises the production nonwinner close and fixed endpoints.
    g_dist.shared_pa.claim_tournament[0].local[0].owner.v = 0;
    g_dist.shared_pa.claim_tournament[0].root.owner.v = 0;
    self_.role = CoreType::AIC;
    self_.local_index = 0;
    self_.occupied_count = 0;

    const DistSharedPaReplayContext replay = dist_shared_pa_replay_context();
    ASSERT_TRUE(replay.ready());
    const DistCompeteFirstTicket ticket = dist_shared_pa_alloc_begin(nullptr, replay);

    ASSERT_EQ(ticket.ready, 1);
    ASSERT_EQ(ticket.won, 0);
    EXPECT_EQ(ticket.task_id, 0);
    EXPECT_EQ(g_fdwic_swimlane_shared_submit_count, 1U);
    for (uint32_t index = 0; index < g_fdwic_swimlane_record_count; ++index) {
        const uint32_t phase =
            (generic()[index].packed >> kFdwicCompactTracePhaseShift) &
            kFdwicCompactTracePhaseMask;
        EXPECT_NE(
            phase, static_cast<uint32_t>(FdwicSwimlanePhase::EfDrain)
        ) << "generic_index=" << index;
    }
    EXPECT_NE(endpoints()[0].claim_begin, 0U);
    EXPECT_NE(endpoints()[0].claim_end_and_winner & ~kFdwicSharedClaimWinnerBit, 0U);
    EXPECT_NE(endpoints()[0].submit_begin, 0U);
    EXPECT_GE(endpoints()[0].submit_end, endpoints()[0].submit_begin);
}

TEST_F(FdwicSharedSwimlaneV5Test, DelayedAtomicPollAndDcciApisClosePhysicalCounters) {
    EXPECT_TRUE(fdwic_swimlane_record_captured_atomic(
        17, FdwicAtomicSite::SharedInsertTurnHandoff, FdwicAtomicOp::CompareExchange, 300, 310, /*result_used=*/true,
        /*return_ready=*/false
    ));
    EXPECT_TRUE(fdwic_swimlane_record_aggregate_atomic_poll(
        FdwicAtomicSite::SharedInsertTurnPoll, 250, 299, 7, /*return_ready_end=*/false
    ));
    EXPECT_TRUE(fdwic_swimlane_record_dcci(
        &self_, 17, 2, FdwicDcciSite::SharedOutputDescriptorFlush,
        /*trailing_dsb=*/true, /*line_count=*/2, 320, 340
    ));

    ASSERT_EQ(g_fdwic_swimlane_record_count, 3U);
    EXPECT_EQ(g_fdwic_atomic_calls, 8U);
    EXPECT_EQ(g_fdwic_poll_calls, 7U);
    EXPECT_EQ(g_fdwic_poll_batch_records, 1U);
    EXPECT_EQ(g_fdwic_dcci_records, 1U);
    EXPECT_EQ(g_fdwic_dcci_calls, 1U);
    EXPECT_EQ(g_fdwic_dcci_lines, 2U);

    EXPECT_EQ(
        (generic()[0].packed >> kFdwicCompactTracePhaseShift) & kFdwicCompactTracePhaseMask,
        static_cast<uint32_t>(FdwicSwimlanePhase::Atomic)
    );
    EXPECT_EQ(
        (generic()[0].packed >> kFdwicCompactTraceAuxShift) & kFdwicCompactTraceAuxMask,
        static_cast<uint32_t>(FdwicAtomicSite::SharedInsertTurnHandoff)
    );
    EXPECT_EQ(generic()[0].flags & kFdwicAtomicOpMask, static_cast<uint32_t>(FdwicAtomicOp::CompareExchange));
    EXPECT_NE(generic()[1].flags & kFdwicAtomicPollBatch, 0U);
    EXPECT_EQ(generic()[1].flags >> kFdwicAtomicPollCountShift, 7U);
    EXPECT_EQ(
        (generic()[2].packed >> kFdwicCompactTracePhaseShift) & kFdwicCompactTracePhaseMask,
        static_cast<uint32_t>(FdwicSwimlanePhase::Dcci)
    );
    EXPECT_NE(generic()[2].flags & kFdwicDcciTrailingDsb, 0U);
}

TEST_F(FdwicSharedSwimlaneV5Test, DelayedApisRejectFlagsThatWouldFailHostClosure) {
    EXPECT_FALSE(fdwic_swimlane_record_captured_atomic(
        17, FdwicAtomicSite::SharedInsertTurnHandoff, FdwicAtomicOp::CompareExchange, 300, 310, /*result_used=*/true,
        /*return_ready=*/false,
        /*value_zero=*/false, /*retries=*/1
    ));
    EXPECT_FALSE(fdwic_swimlane_record_aggregate_atomic_poll(
        FdwicAtomicSite::SharedInsertTurnPoll, 250, 299, 7,
        /*return_ready_end=*/true
    ));
    EXPECT_EQ(g_fdwic_swimlane_record_count, 0U);
    EXPECT_EQ(g_fdwic_atomic_calls, 0U);
    EXPECT_TRUE(g_fdwic_atomic_counter_overflow);
}

TEST(FdwicSharedSwimlaneV5StartupTest, RealAttachResetPublishesCapturedConfigInvalidate) {
    constexpr size_t kTraceBytes = sizeof(FdwicSwimlaneHeader) + kFdwicSwimlaneWorkerBytes;
    void *trace_storage = std::aligned_alloc(64, kTraceBytes);
    ASSERT_NE(trace_storage, nullptr);
    std::memset(trace_storage, 0, kTraceBytes);

    Runtime runtime;
    std::memset(&g_dist_fallback, 0, sizeof(g_dist_fallback));
    runtime.dist.shared_addr = reinterpret_cast<uint64_t>(&g_dist_fallback);
    runtime.dist.num_workers = 1;
    runtime.dist.swimlane_base = reinterpret_cast<uint64_t>(trace_storage);
    runtime.dist.swimlane_level = kFdwicAtomicSwimlaneLevel;
    runtime.dist.swimlane_records_per_core = kFdwicSwimlaneDefaultRecordsPerCore;
    g_dist_fallback.layout[0] = CoreLayout{0, LANE_AIC};

    auto *header = static_cast<FdwicSwimlaneHeader *>(trace_storage);
    header->magic = kFdwicSwimlaneMagic;
    header->version = kFdwicSwimlaneVersion;
    header->num_cores = 1;
    header->records_per_core = kFdwicSwimlaneDefaultRecordsPerCore;
    header->record_size_bytes = kFdwicSwimlaneRecordSizeBytes;

    uint64_t captured_begin = 0;
    uint64_t captured_end = 0;
    DistCore *self =
        dist_aicore_attach_worker(&runtime, 0, static_cast<int>(CoreType::AIC), captured_begin, captured_end);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(captured_begin, 0U);
    ASSERT_GE(captured_end, captured_begin);

    fdwic_swimlane_attach(&runtime);
    fdwic_swimlane_reset_core(self);
    const uint32_t line_count = fdwic_dcci_region_cache_line_count(&runtime.dist.shared_addr, 64);
    ASSERT_EQ(line_count, 1U);
    ASSERT_TRUE(fdwic_swimlane_record_dcci(
        self, -1, -1, FdwicDcciSite::StartupConfigInvalidate, FdwicDcciOp::Invalidate, /*trailing_dsb=*/true,
        /*call_count=*/1, line_count, captured_begin, captured_end
    ));

    ASSERT_EQ(g_fdwic_swimlane_record_count, 1U);
    EXPECT_EQ(g_fdwic_dcci_records, 1U);
    EXPECT_EQ(g_fdwic_dcci_calls, 1U);
    EXPECT_EQ(g_fdwic_dcci_lines, 1U);
    auto *generic = reinterpret_cast<FdwicCompactSwimlaneRecord *>(
        static_cast<uint8_t *>(trace_storage) + sizeof(FdwicSwimlaneHeader) + kFdwicSharedSubmitClaimBytesPerCore
    );
    EXPECT_EQ(generic[0].start_cycle_low, static_cast<uint32_t>(captured_begin));
    EXPECT_EQ(generic[0].end_cycle_low, static_cast<uint32_t>(captured_end));
    EXPECT_EQ(
        (generic[0].packed >> kFdwicCompactTracePhaseShift) & kFdwicCompactTracePhaseMask,
        static_cast<uint32_t>(FdwicSwimlanePhase::Dcci)
    );
    EXPECT_EQ(
        (generic[0].packed >> kFdwicCompactTraceAuxShift) & kFdwicCompactTraceAuxMask,
        static_cast<uint32_t>(FdwicDcciSite::StartupConfigInvalidate)
    );
    EXPECT_EQ(generic[0].flags & kFdwicDcciOpMask, static_cast<uint32_t>(FdwicDcciOp::Invalidate));
    EXPECT_NE(generic[0].flags & kFdwicDcciTrailingDsb, 0U);
    EXPECT_EQ((generic[0].flags >> kFdwicDcciCallCountShift) & kFdwicDcciCallCountMask, 1U);
    EXPECT_EQ(generic[0].flags >> kFdwicDcciLineCountShift, 1U);

    g_self = nullptr;
    g_dist_ptr = nullptr;
    g_fdwic_swimlane_level = 0;
    g_fdwic_swimlane_header = nullptr;
    g_fdwic_swimlane_core = nullptr;
    g_fdwic_swimlane_records = nullptr;
    g_fdwic_swimlane_records_per_core = 0;
    std::free(trace_storage);
}

TEST(FdwicSharedSwimlaneV5HostTest, PhysicalWritersCloseProductionLevel4Exporter) {
    ScopedTemporaryDirectory output;
    ASSERT_FALSE(output.path().empty());

    Runtime runtime;
    configure_shared_host_runtime(runtime);

    ASSERT_EQ(fdwic_swimlane_host_init(&runtime, 96, kFdwicAtomicSwimlaneLevel, output.path().c_str()), 1);
    ASSERT_TRUE(populate_shared_trace(runtime));

    EXPECT_EQ(fdwic_swimlane_host_export(&runtime), 0);
    const std::filesystem::path json = std::filesystem::path(output.path()) / "l2_swimlane_records.json";
    ASSERT_TRUE(std::filesystem::is_regular_file(json));
    EXPECT_GT(std::filesystem::file_size(json), 1024U);

    std::ifstream input(json);
    ASSERT_TRUE(input.is_open());
    std::string prefix(16384, '\0');
    input.read(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    prefix.resize(static_cast<size_t>(input.gcount()));
    EXPECT_NE(prefix.find("\"tensormap_mode\": \"shared\""), std::string::npos);
    EXPECT_NE(prefix.find("\"trace_schema_version\": 5"), std::string::npos);
    EXPECT_NE(prefix.find("\"StartupConfigInvalidate\""), std::string::npos);

    fdwic_swimlane_host_finalize(&runtime);
}

TEST(FdwicSharedSwimlaneV5HostTest, Level1ClosesWithoutAtomicOrDcciRecords) {
    ScopedTemporaryDirectory output;
    ASSERT_FALSE(output.path().empty());

    Runtime runtime;
    configure_shared_host_runtime(runtime);
    ASSERT_EQ(fdwic_swimlane_host_init(&runtime, 96, 1, output.path().c_str()), 1);
    ASSERT_TRUE(populate_shared_trace(runtime));
    EXPECT_EQ(fdwic_swimlane_host_export(&runtime), 0);
    fdwic_swimlane_host_finalize(&runtime);
}

TEST(FdwicSharedSwimlaneV5HostTest, RejectsMissingTournamentLocalForAttemptedClaim) {
    expect_shared_trace_fault_rejected(SharedTraceFault::MissingTournamentLocal);
}

TEST(FdwicSharedSwimlaneV5HostTest, RejectsTournamentLocalForNonattemptedClaim) {
    expect_shared_trace_fault_rejected(SharedTraceFault::ExtraTournamentLocal);
}

TEST(FdwicSharedSwimlaneV5HostTest, RejectsMissingTournamentRootGroup) {
    expect_shared_trace_fault_rejected(SharedTraceFault::MissingTournamentRoot);
}

TEST(FdwicSharedSwimlaneV5HostTest, RejectsDuplicateTournamentRootGroup) {
    expect_shared_trace_fault_rejected(SharedTraceFault::ExtraTournamentRoot);
}

TEST(FdwicSharedSwimlaneV5HostTest, RejectsMissingWinnerBusinessDcci) {
    expect_shared_trace_fault_rejected(SharedTraceFault::MissingDcci);
}

TEST(FdwicSharedSwimlaneV5HostTest, RejectsWrongWinnerBusinessDcciLineCount) {
    expect_shared_trace_fault_rejected(SharedTraceFault::WrongDcciLines);
}

}  // namespace
