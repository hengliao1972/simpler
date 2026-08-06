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

#include "../common/g0_full_pa.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core::g0;

constexpr uint32_t kNoTask = UINT32_MAX;
constexpr auto kCpuWaitLimit = std::chrono::seconds(10);
constexpr uint64_t kFaninWitnessPresent = uint64_t{1} << 63U;

struct CpuTask {
    std::atomic<uint64_t> exec_state{0U};
    std::atomic<int64_t> completion_flag{0};
    std::atomic<uint64_t> completion_vend{0U};
    std::atomic<uint32_t> build_attempt_count{0U};
    std::atomic<uint32_t> build_win_count{0U};
    std::atomic<int64_t> insert_completion{0};
    std::atomic<uint64_t> task_base_plus_one{0U};
    std::atomic<uint64_t> completion_vend_plus_one{0U};
    std::array<std::atomic<int64_t>, kOutputsPerTask> published{};
    std::array<std::atomic<int64_t>, kOutputsPerTask> last_writer{};
    std::array<TensorDesc, kOutputsPerTask> outputs{};
    WriterHistoryCell history{};
    std::array<uint64_t, kMaxPayloadWords> payload{};
    FullPaTaskPlan plan{};
    FullPaBuildReport build_report{};
    std::atomic<uint64_t> witness_state{0U};
    std::atomic<uint64_t> fanin_timing_witness{0U};
    uint64_t witness_nonce = 0U;
    uint64_t witness_magic = 0U;
    uint32_t witness_task = 0U;
    TaskKind witness_kind = TaskKind::Alloc;
    uint32_t witness_owner = 0U;
    uint32_t execution_count = 0U;
    uint64_t completion_sequence = 0U;
    uint64_t output_checksum = 0U;
    uint64_t witness_fanin_ready_prefix = 0U;

    CpuTask() {
        for (uint32_t slot = 0U; slot < kOutputsPerTask; ++slot) {
            published[slot].store(-1, std::memory_order_relaxed);
            last_writer[slot].store(-1, std::memory_order_relaxed);
        }
    }
};

struct CpuOwner {
    FullPaRoleResult result{};
    std::array<ExecutionToken, kTokensPerOwner> tokens{};
    std::array<uint32_t, kTokensPerOwner> last_ticket_tasks{};
    uint32_t busy_tokens = 0U;
    bool exhausted = false;
};

enum class BuildAttemptResult : uint32_t {
    Won = 0,
    Lost = 1,
    Error = 2,
};

struct HeapInterval {
    uint64_t begin;
    uint64_t end;
    uint32_t task_id;
};

class PausePoint {
public:
    void Stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        reached_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this] {
            return released_;
        });
    }

    bool WaitUntilReached() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, kCpuWaitLimit, [this] {
            return reached_;
        });
    }

    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool reached_ = false;
    bool released_ = false;
};

class StartGate {
public:
    explicit StartGate(uint32_t participants) :
        participants_(participants) {}

    void ArriveAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrived_;
        condition_.notify_all();
        condition_.wait(lock, [this] {
            return open_;
        });
    }

    bool OpenWhenReady() {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool ready = condition_.wait_for(lock, kCpuWaitLimit, [this] {
            return arrived_ == participants_;
        });
        open_ = true;
        condition_.notify_all();
        return ready;
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    uint32_t participants_;
    uint32_t arrived_ = 0U;
    bool open_ = false;
};

bool TensorEqual(const TensorDesc &lhs, const TensorDesc &rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(TensorDesc)) == 0;
}

uint64_t PackOutputPair(const std::array<float, 2> &values) {
    std::array<uint32_t, 2> bits{};
    std::memcpy(bits.data(), values.data(), sizeof(bits));
    return static_cast<uint64_t>(bits[0]) | (static_cast<uint64_t>(bits[1]) << 32U);
}

ExecPayloadHeader DecodeCpuHeader(const std::array<uint64_t, kMaxPayloadWords> &payload) {
    const uint64_t word0 = payload[0];
    const uint64_t word3 = payload[3];
    const uint64_t word4 = payload[4];
    const uint64_t word5 = payload[5];
    return ExecPayloadHeader{
        static_cast<uint32_t>(word0),
        payload[1],
        payload[2],
        static_cast<uint32_t>(word3),
        static_cast<uint32_t>(word3 >> 32U),
        static_cast<uint16_t>(word4),
        static_cast<uint16_t>(word4 >> 16U),
        static_cast<uint16_t>(word4 >> 32U),
        static_cast<ExecEngineClass>(static_cast<uint8_t>(word4 >> 48U)),
        static_cast<uint8_t>(word4 >> 56U),
        static_cast<uint32_t>(word5),
        static_cast<uint16_t>(word5 >> 32U),
        static_cast<uint16_t>(word5 >> 48U),
        static_cast<uint32_t>(payload[6]),
    };
}

bool CompetingKernelClaimValid(
    uint32_t task_id, uint32_t current_owner, uint32_t builder_count, uint64_t raw_state
) {
    const DecodedExecState decoded = DecodeExecState(static_cast<int64_t>(raw_state));
    if (!decoded.valid || decoded.task_id != task_id || !TaskExecutable(TaskKindAt(task_id)) ||
        !IsBuilderOwner(decoded.build_owner, builder_count) || decoded.build_owner == current_owner) {
        return false;
    }
    if (decoded.phase == ExecPhase::Building) {
        return decoded.execute_owner == kUnboundOwner && decoded.engine_class == ExecEngineClass::None &&
               decoded.payload_lines == 0U;
    }
    const TaskKind kind = TaskKindAt(task_id);
    const TaskExecShape shape = TaskShape(kind);
    ExecPayloadLayout layout{};
    if (!ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout) ||
        decoded.engine_class != TaskEngine(kind) || decoded.payload_lines != layout.payload_lines) {
        return false;
    }
    if (decoded.phase == ExecPhase::Built) {
        return decoded.execute_owner == kUnboundOwner;
    }
    if (decoded.phase != ExecPhase::Claimed && decoded.phase != ExecPhase::Done) {
        return false;
    }
    return OwnerCanExecute(decoded.execute_owner, decoded.engine_class, builder_count);
}

bool CompetingAllocClaimValid(
    uint64_t nonce, uint32_t task_id, uint32_t current_owner, uint32_t builder_count, int64_t raw_state
) {
    if (raw_state == 1) {
        return true;
    }
    for (uint32_t builder = 0U; builder < builder_count; ++builder) {
        const uint32_t owner = BuilderOwnerForInstance(builder);
        if (owner != current_owner &&
            static_cast<uint64_t>(raw_state) == AllocBuildingState(nonce, task_id, owner)) {
            return true;
        }
    }
    return false;
}

void ResetExecutionTokenControl(ExecutionToken *token) {
    token->control.phase = ExecTokenPhase::Idle;
    token->control.task_id = UINT32_MAX;
    token->control.build_owner = UINT32_MAX;
    token->control.execute_owner = UINT32_MAX;
    token->control.engine_class = ExecEngineClass::None;
    token->control.payload_lines = 0U;
    token->control.payload_bytes = 0U;
    token->control.fanin_ready_prefix = 0U;
    token->control.payload_address = 0U;
    token->control.completion_vend = 0U;
    token->control.function_and_reference = 0U;
    token->control.shape_and_scalar_offset = 0U;
}

TensorDesc LoadPayloadTensor(
    const std::array<uint64_t, kMaxPayloadWords> &payload, const ExecPayloadLayout &layout, uint32_t tensor
) {
    TensorDesc descriptor{};
    std::array<uint64_t, kTensorDescWords> words{};
    const uint32_t begin = layout.tensor_word_offset + tensor * kTensorDescWords;
    for (uint32_t word = 0U; word < kTensorDescWords; ++word) {
        words[word] = payload[begin + word];
    }
    std::memcpy(&descriptor, words.data(), sizeof(descriptor));
    return descriptor;
}

class CpuFullPaModel {
public:
    CpuFullPaModel(uint32_t batches, uint64_t nonce, uint32_t builder_count) :
        batches_(batches),
        task_count_(TaskCount(batches)),
        kernel_task_count_(KernelTaskCount(batches)),
        builder_count_(builder_count),
        nonce_(nonce),
        tasks_(std::make_unique<CpuTask[]>(task_count_)) {
        for (std::atomic<uint64_t> &cursor : heap_cursors_) {
            cursor.store(0U, std::memory_order_relaxed);
        }
        heap_vend_.store(0U, std::memory_order_relaxed);
        fatal_.store(0U, std::memory_order_relaxed);
        builder_started_.store(0U, std::memory_order_relaxed);
        builder_finished_.store(0U, std::memory_order_relaxed);
        done_count_.store(0U, std::memory_order_relaxed);
        alloc_done_.store(0U, std::memory_order_relaxed);
        aic_done_.store(0U, std::memory_order_relaxed);
        aiv_done_.store(0U, std::memory_order_relaxed);
        aic_cursor_.store(0U, std::memory_order_relaxed);
        aiv_cursor_.store(0U, std::memory_order_relaxed);
        committed_prefix_.store(0U, std::memory_order_relaxed);
        for (std::atomic<int64_t> &arrival : drain_arrivals_) {
            arrival.store(0, std::memory_order_relaxed);
        }
        root_finished_.store(0U, std::memory_order_relaxed);
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            InitializeTask(task_id);
            const TaskKind kind = TaskKindAt(task_id);
            if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
                aic_task_ids_.push_back(task_id);
            } else if (kind == TaskKind::Sf || kind == TaskKind::Up) {
                aiv_task_ids_.push_back(task_id);
            }
        }
        for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
            InitializeOwner(owner);
        }
        std::memset(builder_threads_.data(), 0xD3, sizeof(builder_threads_));
        for (uint32_t thread = 0U; thread < BuilderThreadCount(builder_count_); ++thread) {
            if (BuilderThreadActive(thread, builder_count_)) {
                InitializeBuilderThreadReport(thread);
            }
        }
        for (std::array<std::array<float, 2>, 2> &owner : owner_workload_) {
            for (std::array<float, 2> &values : owner) {
                values.fill(kWorkloadOutputSentinel);
            }
        }
        workload_input_a_.fill(kWorkloadInputA);
        workload_input_b_.fill(kWorkloadInputB);
    }

    bool RunConcurrent() {
        if (!BuilderCountValid(builder_count_)) {
            return false;
        }
        const uint32_t participant_count = BuilderLeaderCount(builder_count_) + ExecutorCount(builder_count_);
        StartGate start_gate(participant_count);
        std::array<std::atomic<bool>, kMaxBuilderLeaderCount> builder_ok{};
        std::array<std::atomic<bool>, kOwnerCount> executor_ok{};
        for (std::atomic<bool> &ok : builder_ok) {
            ok.store(false, std::memory_order_relaxed);
        }
        for (std::atomic<bool> &ok : executor_ok) {
            ok.store(false, std::memory_order_relaxed);
        }

        std::vector<std::thread> leaders;
        leaders.reserve(BuilderLeaderCount(builder_count_));
        for (uint32_t builder = 0U; builder < builder_count_; ++builder) {
            for (uint32_t warp = 0U; warp < kBuilderWarpCount; ++warp) {
                const uint32_t leader = builder * kBuilderLeaderCount + warp;
                const uint32_t thread_id = BuilderThreadForInstanceWarp(builder, warp);
                leaders.emplace_back([this, leader, thread_id, &start_gate, &builder_ok] {
                    start_gate.ArriveAndWait();
                    builder_ok[leader].store(
                        EnterBuilderGate(thread_id) && RunBuilderWarp(thread_id), std::memory_order_release
                    );
                });
            }
        }

        std::vector<std::thread> executors;
        executors.reserve(ExecutorCount(builder_count_));
        for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
            if (IsBuilderOwner(owner, builder_count_)) {
                continue;
            }
            executors.emplace_back([this, owner, &start_gate, &executor_ok] {
                start_gate.ArriveAndWait();
                executor_ok[owner].store(RunExecutorOwner(owner), std::memory_order_release);
            });
        }

        if (!start_gate.OpenWhenReady()) {
            PublishFatal(ExecFatalReason::Timeout, kNoTask, kBuilderOwner);
        }

        for (std::thread &leader : leaders) {
            leader.join();
        }
        bool all_builders_ok = true;
        for (uint32_t leader = 0U; leader < BuilderLeaderCount(builder_count_); ++leader) {
            all_builders_ok &= builder_ok[leader].load(std::memory_order_acquire);
        }
        all_builders_ok &= FinalizePerTaskBuildEvidence();
        bool builders_arrived = false;
        bool drain_ok = false;
        if (all_builders_ok && fatal_.load(std::memory_order_acquire) == 0U) {
            builders_arrived = builder_finished_.load(std::memory_order_acquire) == 1U;
            for (uint32_t builder = 0U; builder < builder_count_ && builders_arrived; ++builder) {
                builders_arrived &= ArriveDrain(BuilderOwnerForInstance(builder));
            }
            if (builders_arrived && fatal_.load(std::memory_order_acquire) == 0U) {
                drain_ok = WaitForDrainAndPublishRoot();
            }
        }

        for (std::thread &executor : executors) {
            executor.join();
        }
        bool all_executors_ok = true;
        for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
            if (!IsBuilderOwner(owner, builder_count_)) {
                all_executors_ok &= executor_ok[owner].load(std::memory_order_acquire);
            }
        }
        return all_builders_ok && all_executors_ok && builders_arrived && drain_ok &&
               fatal_.load(std::memory_order_acquire) == 0U;
    }

    bool Validate() const {
        return ValidateCounts() && ValidateBuilderMapping() && ValidateHeapAndOutputs() && ValidatePayloads() &&
               ValidateCompletionAndWitnesses() && ValidateRolesAndDrain();
    }

    bool MaterializeAndValidate(FullPaState *state) const {
        InitializeAbiState(state);
        MaterializeActiveTasks(state);
        MaterializeGlobalState(state);
        return ValidateAbiState(*state);
    }

    uint32_t Batches() const { return batches_; }
    uint32_t TaskCountValue() const { return task_count_; }
    uint64_t Nonce() const { return nonce_; }

private:
    void InitializeTask(uint32_t task_id) {
        CpuTask &task = tasks_[task_id];
        task.insert_completion.store(InsertCompletionInitialValue(task_id), std::memory_order_relaxed);
        const TaskKind kind = TaskKindAt(task_id);
        for (uint32_t slot = 0U; slot < TaskOutputCount(kind); ++slot) {
            std::array<uint64_t, kTensorDescWords> poison{};
            for (uint32_t word = 0U; word < kTensorDescWords; ++word) {
                poison[word] = ExpectedDescriptorPoisonWord(nonce_, task_id, slot, word);
            }
            std::memcpy(&task.outputs[slot], poison.data(), sizeof(TensorDesc));
        }
        for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
            task.payload[word] = ExpectedPayloadPoisonWord(nonce_, task_id, word);
        }
    }

    void InitializeOwner(uint32_t owner) {
        CpuOwner &worker = owners_[owner];
        std::memset(&worker.result, 0, sizeof(worker.result));
        worker.result.owner = owner;
        worker.result.role = OwnerRoleAt(owner, builder_count_);
        worker.result.physical_block = OwnerPhysicalBlock(owner);
        worker.result.drain_group = OwnerDrainGroup(owner);
        worker.result.launch_nonce = nonce_;
        worker.last_ticket_tasks.fill(kNoTask);
        for (ExecutionToken &token : worker.tokens) {
            std::memset(&token, 0, sizeof(token));
            ResetExecutionTokenControl(&token);
        }
    }

    void InitializeBuilderThreadReport(uint32_t thread_id) {
        FullPaBuilderThreadReport &report = builder_threads_[thread_id];
        std::memset(&report, 0, sizeof(report));
        report.thread_id = thread_id;
        report.warp_id = thread_id / kWarpSize;
        report.lane_id = thread_id % kWarpSize;
        report.active_leader = BuilderThreadActive(thread_id, builder_count_) ? 1U : 0U;
        report.task_count = 0U;
        report.first_task = kNoTask;
        report.last_task = kNoTask;
        report.task_state_access_count = 0U;
        report.prepare_count = 0U;
        report.commit_count = 0U;
        report.insert_wait_count = 0U;
        report.claim_lost_count = 0U;
        report.launch_nonce = nonce_;
        report.checksum =
            BuilderReportChecksum(nonce_, thread_id, task_count_, 0U, kNoTask, kNoTask, 0U, 0U, 0U, 0U, 0U);
    }

    template <typename Predicate>
    bool WaitFor(Predicate predicate, uint32_t *poll_count) {
        const auto deadline = std::chrono::steady_clock::now() + kCpuWaitLimit;
        std::unique_lock<std::mutex> lock(progress_mutex_);
        while (!predicate()) {
            ++*poll_count;
            if (fatal_.load(std::memory_order_acquire) != 0U ||
                progress_condition_.wait_until(lock, deadline) == std::cv_status::timeout) {
                return false;
            }
        }
        ++*poll_count;
        return true;
    }

    void SignalProgress() {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        ++progress_epoch_;
        progress_condition_.notify_all();
    }

    uint64_t SnapshotProgressEpoch() {
        std::lock_guard<std::mutex> lock(progress_mutex_);
        return progress_epoch_;
    }

    bool WaitForProgress(uint64_t observed_epoch, std::chrono::steady_clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(progress_mutex_);
        return progress_condition_.wait_until(lock, deadline, [this, observed_epoch] {
            return progress_epoch_ != observed_epoch || fatal_.load(std::memory_order_acquire) != 0U;
        });
    }

    void PublishFatal(ExecFatalReason reason, uint32_t task_id, uint32_t owner) {
        uint64_t expected = 0U;
        fatal_.compare_exchange_strong(
            expected, EncodeExecFatal(reason, owner, task_id), std::memory_order_acq_rel, std::memory_order_acquire
        );
        SignalProgress();
    }

    bool EnterBuilderGate(uint32_t thread_id) {
        if (BuilderLocalThread(thread_id) == 0U) {
            const uint64_t observed = builder_started_.fetch_add(1U, std::memory_order_acq_rel);
            if (observed >= builder_count_) {
                PublishFatal(ExecFatalReason::ControlPublishConflict, kNoTask, BuilderOwnerForThread(thread_id));
                return false;
            }
            SignalProgress();
        }
        uint32_t polls = 0U;
        return WaitFor(
            [this] {
                return builder_started_.load(std::memory_order_acquire) == builder_count_;
            },
            &polls
        );
    }

    bool ExistingKernelClaimValid(uint32_t task_id, uint32_t current_owner, uint64_t raw_state) const {
        return CompetingKernelClaimValid(task_id, current_owner, builder_count_, raw_state);
    }

    bool ExistingAllocClaimValid(uint32_t task_id, uint32_t current_owner, int64_t raw_state) const {
        return CompetingAllocClaimValid(nonce_, task_id, current_owner, builder_count_, raw_state);
    }

    bool RunBuilderWarp(uint32_t thread_id) {
        FullPaBuilderThreadReport &report = builder_threads_[thread_id];
        const uint32_t owner = BuilderOwnerForThread(thread_id);
        const uint32_t first_attempt = BuilderFirstTask(thread_id, builder_count_);
        for (uint32_t task_id = first_attempt; task_id < task_count_;
             task_id += BuilderLeaderCount(builder_count_)) {
            ++report.task_state_access_count;
            const BuildAttemptResult result = BuildTask(task_id, thread_id);
            if (result == BuildAttemptResult::Error) {
                PublishFatal(ExecFatalReason::BuildPackFailed, task_id, owner);
                return false;
            }
            if (result == BuildAttemptResult::Lost) {
                ++report.claim_lost_count;
                continue;
            }
            if (report.task_count == 0U) {
                report.first_task = task_id;
            }
            ++report.task_count;
            report.last_task = task_id;
            ++report.prepare_count;
            ++report.commit_count;
            report.insert_wait_count +=
                TaskKindAt(task_id) == TaskKind::Up && TaskBatch(task_id) != 0U ? 1U : 0U;
        }
        report.checksum = BuilderReportChecksum(
            nonce_, thread_id, task_count_, report.task_count, report.first_task, report.last_task,
            report.task_state_access_count, report.prepare_count, report.commit_count, report.insert_wait_count,
            report.claim_lost_count
        );
        return true;
    }

    bool FinalizePerTaskBuildEvidence() {
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            CpuTask &task = tasks_[task_id];
            const uint32_t attempts = task.build_attempt_count.load(std::memory_order_acquire);
            const uint32_t wins = task.build_win_count.load(std::memory_order_acquire);
            if (attempts != 1U || wins != 1U) {
                PublishFatal(ExecFatalReason::ControlPublishConflict, task_id, kBuilderOwner);
                return false;
            }
            task.build_report.build_attempt_count = attempts;
            task.build_report.build_win_count = wins;
        }
        return true;
    }

    bool RunExecutorOwner(uint32_t owner) {
        CpuOwner &worker = owners_[owner];
        auto deadline = std::chrono::steady_clock::now() + kCpuWaitLimit;
        while (true) {
            const uint64_t observed_epoch = SnapshotProgressEpoch();
            bool progress = FillTokens(worker);
            for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
                progress |= AdvanceToken(worker, slot);
            }
            if (fatal_.load(std::memory_order_acquire) != 0U) {
                return false;
            }
            if (worker.exhausted && worker.busy_tokens == 0U) {
                return ArriveDrain(owner);
            }
            if (progress) {
                deadline = std::chrono::steady_clock::now() + kCpuWaitLimit;
            } else if (!WaitForProgress(observed_epoch, deadline)) {
                uint32_t blocked_task = kNoTask;
                for (const ExecutionToken &token : worker.tokens) {
                    if (token.control.phase != ExecTokenPhase::Idle) {
                        blocked_task = token.control.task_id;
                        break;
                    }
                }
                PublishFatal(ExecFatalReason::Timeout, blocked_task, owner);
                return false;
            }
        }
    }

    bool ReserveHeap(uint32_t task_id, uint64_t *task_base, uint64_t *vend) {
        const uint64_t reserve = AlignOutputBytes(TaskOutputBytes(TaskKindAt(task_id)));
        if (reserve == 0U) {
            *task_base = 0U;
            *vend = heap_vend_.load(std::memory_order_acquire);
        } else {
            const uint32_t shard = TaskHeapShard(task_id);
            const uint64_t cursor = heap_cursors_[shard].fetch_add(reserve, std::memory_order_acq_rel);
            if (cursor > kHeapShardSpan - reserve) {
                return false;
            }
            *task_base = static_cast<uint64_t>(shard) * kHeapShardSpan + cursor;
            *vend = heap_vend_.fetch_add(reserve, std::memory_order_acq_rel) + reserve;
        }
        tasks_[task_id].completion_vend_plus_one.store(*vend + 1U, std::memory_order_release);
        tasks_[task_id].task_base_plus_one.store(*task_base + 1U, std::memory_order_release);
        SignalProgress();
        return true;
    }

    bool LoadPublishedTaskBase(uint32_t producer, uint64_t *task_base) {
        uint32_t polls = 0U;
        if (!WaitFor(
                [this, producer] {
                    return tasks_[producer].task_base_plus_one.load(std::memory_order_acquire) != 0U;
                },
                &polls
            )) {
            return false;
        }
        *task_base = tasks_[producer].task_base_plus_one.load(std::memory_order_acquire) - 1U;
        return true;
    }

    bool PackPayload(uint32_t task_id, uint64_t vend, ExecPayloadLayout *layout) {
        CpuTask &task = tasks_[task_id];
        const TaskKind kind = TaskKindAt(task_id);
        const TaskExecShape shape = TaskShape(kind);
        if (!ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, *layout)) {
            return false;
        }
        task.payload[0] = task_id;
        task.payload[1] = 0U;
        task.payload[2] = vend;
        task.payload[3] = PackHeaderWord3(TaskFunctionId(kind), layout->payload_bytes);
        task.payload[4] =
            PackHeaderWord4(shape.tensor_count, shape.scalar_count, shape.fanin_count, shape.engine_class);
        task.payload[5] = uint64_t{1} << 48U;
        task.payload[6] = 0U;
        task.payload[7] = 0U;
        uint32_t destination = layout->tensor_word_offset;
        for (uint32_t tensor_index = 0U; tensor_index < shape.tensor_count; ++tensor_index) {
            TensorDesc tensor{};
            uint32_t producer = 0U;
            uint32_t output_slot = 0U;
            if (PayloadTensorOutputSource(task_id, tensor_index, producer, output_slot)) {
                uint64_t producer_base = 0U;
                if (!LoadPublishedTaskBase(producer, &producer_base)) {
                    return false;
                }
                tensor = MakeTaskOutputDescriptor(producer, output_slot, producer_base);
            } else if (!ResolveExternalPayloadTensor(batches_, task_id, tensor_index, tensor)) {
                return false;
            }
            std::array<uint64_t, kTensorDescWords> words{};
            std::memcpy(words.data(), &tensor, sizeof(tensor));
            for (uint64_t word : words) {
                task.payload[destination++] = word;
            }
        }
        for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
            task.payload[destination++] = TaskScalar(task_id, scalar);
        }
        for (uint32_t edge = 0U; edge < shape.fanin_count; edge += 2U) {
            const uint64_t low = static_cast<uint32_t>(TaskFanin(task_id, edge));
            const uint64_t high =
                edge + 1U < shape.fanin_count ? static_cast<uint32_t>(TaskFanin(task_id, edge + 1U)) : 0U;
            task.payload[destination++] = low | (high << 32U);
        }
        return destination == layout->written_words;
    }

    BuildAttemptResult BuildTask(uint32_t task_id, uint32_t thread_id) {
        CpuTask &task = tasks_[task_id];
        const TaskKind kind = TaskKindAt(task_id);
        const uint32_t build_owner = BuilderOwnerForThread(thread_id);
        task.build_attempt_count.fetch_add(1U, std::memory_order_acq_rel);
        uint64_t expected_state = 0U;
        int64_t alloc_claim = 0;
        if (kind == TaskKind::Alloc) {
            const int64_t desired = static_cast<int64_t>(AllocBuildingState(nonce_, task_id, build_owner));
            if (!task.completion_flag.compare_exchange_strong(
                    alloc_claim, desired, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                return ExistingAllocClaimValid(task_id, build_owner, alloc_claim) ? BuildAttemptResult::Lost :
                                                                                    BuildAttemptResult::Error;
            }
            alloc_claim = desired;
        } else if (!task.exec_state.compare_exchange_strong(
                       expected_state, BuildingState(task_id, build_owner), std::memory_order_acq_rel,
                       std::memory_order_acquire
                   )) {
            return ExistingKernelClaimValid(task_id, build_owner, expected_state) ? BuildAttemptResult::Lost :
                                                                                    BuildAttemptResult::Error;
        }
        if (task.build_win_count.fetch_add(1U, std::memory_order_acq_rel) != 0U) {
            return BuildAttemptResult::Error;
        }
        uint64_t task_base = 0U;
        uint64_t vend = 0U;
        if (!ReserveHeap(task_id, &task_base, &vend)) {
            return BuildAttemptResult::Error;
        }
        const uint32_t output_count = TaskOutputCount(kind);
        for (uint32_t slot = 0U; slot < output_count; ++slot) {
            task.outputs[slot] = MakeTaskOutputDescriptor(task_id, slot, task_base);
            task.published[slot].store(task_id, std::memory_order_release);
            task.last_writer[slot].store(task_id, std::memory_order_release);
        }
        ExecPayloadLayout layout{};
        if (TaskExecutable(kind) && !PackPayload(task_id, vend, &layout)) {
            return BuildAttemptResult::Error;
        }
        uint32_t insert_polls = 0U;
        int64_t predecessor_observed = -1;
        if (kind == TaskKind::Up) {
            const uint32_t alloc_task = BatchTaskId(TaskBatch(task_id), TaskKind::Alloc);
            uint32_t output_polls = 0U;
            if (!WaitFor(
                    [this, alloc_task] {
                        return tasks_[alloc_task].published[0].load(std::memory_order_acquire) ==
                               static_cast<int64_t>(alloc_task);
                    },
                    &output_polls
                )) {
                return BuildAttemptResult::Error;
            }
            if (TaskBatch(task_id) != 0U) {
                const uint32_t predecessor_task = task_id - kTasksPerBatch;
                if (!WaitFor(
                        [this, predecessor_task] {
                            return tasks_[predecessor_task].insert_completion.load(std::memory_order_acquire) ==
                                   static_cast<int64_t>(predecessor_task);
                        },
                        &insert_polls
                    )) {
                    return BuildAttemptResult::Error;
                }
                predecessor_observed =
                    tasks_[predecessor_task].insert_completion.load(std::memory_order_acquire);
            }
        }
        if (kind == TaskKind::Up) {
            task.history.magic = kWriterHistoryMagic;
            task.history.writer_task = static_cast<int32_t>(task_id);
            task.history.count = 3U;
            task.history.reserved = 0U;
            const uint32_t batch = TaskBatch(task_id);
            const int32_t previous_writer = static_cast<int32_t>(BatchTaskId(batch, TaskKind::Alloc));
            for (uint32_t index = 0U; index < 3U; ++index) {
                task.history.entries[index] = WriterHistoryRecord{
                    WriterHistorySymbolKey(batch, index),
                    previous_writer,
                };
            }
            tasks_[BatchTaskId(batch, TaskKind::Alloc)].last_writer[0].store(task_id, std::memory_order_release);
        }
        if (kind == TaskKind::Up) {
            uint32_t expected_prefix = TaskBatch(task_id);
            if (!committed_prefix_.compare_exchange_strong(
                    expected_prefix, TaskBatch(task_id) + 1U, std::memory_order_acq_rel,
                    std::memory_order_acquire
                )) {
                return BuildAttemptResult::Error;
            }
        }
        if (kind == TaskKind::Up) {
            const int64_t previous_insert = task.insert_completion.fetch_add(1, std::memory_order_acq_rel);
            if (previous_insert != InsertCompletionInitialValue(task_id)) {
                return BuildAttemptResult::Error;
            }
        }
        if (kind == TaskKind::Alloc) {
            task.completion_vend.store(vend, std::memory_order_relaxed);
            if (!task.completion_flag.compare_exchange_strong(
                    alloc_claim, 1, std::memory_order_release, std::memory_order_acquire
                )) {
                return BuildAttemptResult::Error;
            }
            alloc_done_.fetch_add(1U, std::memory_order_acq_rel);
        } else {
            expected_state = BuildingState(task_id, build_owner);
            if (!task.exec_state.compare_exchange_strong(
                    expected_state, BuiltState(task_id, build_owner), std::memory_order_release,
                    std::memory_order_acquire
                )) {
                return BuildAttemptResult::Error;
            }
        }
        task.plan = FullPaTaskPlan{
            task_id,
            TaskBatch(task_id),
            kind,
            TaskEngine(kind),
            output_count,
            layout.payload_lines,
            thread_id,
            BuilderWarp(thread_id),
            EncodeTaskMeta(task_id, task_count_),
            EncodeTaskExecRoute(kind),
            static_cast<uint16_t>(build_owner),
            AlignOutputBytes(TaskOutputBytes(kind)),
            nonce_,
            0U,
        };
        const uint32_t phase_bits = kBuildPreparedBit | kBuildOutputsPublishedBit | kBuildInsertCommittedBit |
                                    (TaskExecutable(kind) ? kBuildExecPublishedBit : kBuildAllocCompletedBit);
        task.build_report = FullPaBuildReport{
            task_id,
            thread_id,
            BuilderWarp(thread_id),
            0U,
            phase_bits,
            output_count,
            layout.written_words,
            insert_polls,
            predecessor_observed,
            1U,
            1U,
            0U,
            0U,
            nonce_,
        };
        if (task_id + 1U == task_count_) {
            uint64_t expected_finished = 0U;
            if (!builder_finished_.compare_exchange_strong(
                    expected_finished, 1U, std::memory_order_release, std::memory_order_acquire
                )) {
                return BuildAttemptResult::Error;
            }
        }
        SignalProgress();
        return BuildAttemptResult::Won;
    }

    bool FillTokens(CpuOwner &worker) {
        if (worker.exhausted) {
            return false;
        }
        bool progress = false;
        const ExecEngineClass engine = OwnerEngine(worker.result.owner, builder_count_);
        const std::vector<uint32_t> &task_ids = engine == ExecEngineClass::Aic ? aic_task_ids_ : aiv_task_ids_;
        std::atomic<uint32_t> &cursor = engine == ExecEngineClass::Aic ? aic_cursor_ : aiv_cursor_;
        for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
            ExecutionToken &token = worker.tokens[slot];
            if (token.control.phase != ExecTokenPhase::Idle) {
                continue;
            }
            const uint32_t ticket = cursor.fetch_add(1U, std::memory_order_acq_rel);
            progress = true;
            if (ticket >= task_ids.size()) {
                worker.exhausted = true;
                ++worker.result.exhausted_ticket_count;
                break;
            }
            const uint32_t task_id = task_ids[ticket];
            token.control.phase = ExecTokenPhase::WaitingBuilt;
            token.control.task_id = task_id;
            token.control.build_owner = UINT32_MAX;
            token.control.execute_owner = worker.result.owner;
            token.control.engine_class = engine;
            token.control.payload_address = reinterpret_cast<uint64_t>(tasks_[task_id].payload.data());
            worker.last_ticket_tasks[slot] = task_id;
            ++worker.result.ticket_count;
            ++worker.busy_tokens;
            worker.result.max_busy_tokens = std::max(worker.result.max_busy_tokens, worker.busy_tokens);
        }
        return progress;
    }

    bool PayloadValid(uint32_t task_id) const {
        const CpuTask &task = tasks_[task_id];
        const TaskKind kind = TaskKindAt(task_id);
        const TaskExecShape shape = TaskShape(kind);
        ExecPayloadLayout layout{};
        if (!ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
            return false;
        }
        const ExecPayloadHeader header = DecodeCpuHeader(task.payload);
        if ((task.payload[0] >> 32U) != 0U || task.payload[6] != 0U || task.payload[7] != 0U ||
            header.task_id != task_id || header.function_address != 0U || header.function_id != TaskFunctionId(kind) ||
            header.payload_bytes != layout.payload_bytes || header.tensor_count != shape.tensor_count ||
            header.scalar_count != shape.scalar_count || header.fanin_count != shape.fanin_count ||
            header.engine_class != shape.engine_class || header.flags != 0U || header.multicore_group_id != 0U ||
            header.multicore_rank != 0U || header.multicore_size != 1U || header.tensor_reference_mask != 0U) {
            return false;
        }
        for (uint32_t tensor_index = 0U; tensor_index < shape.tensor_count; ++tensor_index) {
            TensorDesc expected{};
            uint32_t producer = 0U;
            uint32_t output_slot = 0U;
            if (PayloadTensorOutputSource(task_id, tensor_index, producer, output_slot)) {
                const uint64_t base_plus_one = tasks_[producer].task_base_plus_one.load(std::memory_order_acquire);
                if (base_plus_one == 0U) {
                    return false;
                }
                expected = MakeTaskOutputDescriptor(producer, output_slot, base_plus_one - 1U);
            } else if (!ResolveExternalPayloadTensor(batches_, task_id, tensor_index, expected)) {
                return false;
            }
            if (!TensorEqual(LoadPayloadTensor(task.payload, layout, tensor_index), expected)) {
                return false;
            }
        }
        for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
            if (task.payload[layout.scalar_word_offset + scalar] != TaskScalar(task_id, scalar)) {
                return false;
            }
        }
        for (uint32_t edge = 0U; edge < shape.fanin_count; ++edge) {
            const uint64_t packed = task.payload[layout.fanin_word_offset + edge / 2U];
            const int32_t actual = static_cast<int32_t>(
                edge % 2U == 0U ? static_cast<uint32_t>(packed) : static_cast<uint32_t>(packed >> 32U)
            );
            if (actual != TaskFanin(task_id, edge)) {
                return false;
            }
        }
        if ((shape.fanin_count & 1U) != 0U &&
            (task.payload[layout.fanin_word_offset + shape.fanin_count / 2U] >> 32U) != 0U) {
            return false;
        }
        for (uint32_t word = layout.written_words; word < kMaxPayloadWords; ++word) {
            if (task.payload[word] != ExpectedPayloadPoisonWord(nonce_, task_id, word)) {
                return false;
            }
        }
        return true;
    }

    bool BindExecutionToken(ExecutionToken *token, uint32_t task_id, uint32_t owner) const {
        const CpuTask &task = tasks_[task_id];
        const TaskExecShape shape = TaskShape(TaskKindAt(task_id));
        ExecPayloadLayout layout{};
        if (!ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
            return false;
        }
        const ExecPayloadHeader header = DecodeCpuHeader(task.payload);
        for (uint32_t tensor = 0U; tensor < shape.tensor_count; ++tensor) {
            const uint32_t offset = layout.tensor_word_offset + tensor * kTensorDescWords;
            token->dispatch.args[tensor] = reinterpret_cast<uint64_t>(&task.payload[offset]);
        }
        for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
            token->dispatch.args[shape.tensor_count + scalar] = task.payload[layout.scalar_word_offset + scalar];
        }
        token->control.payload_lines = layout.payload_lines;
        token->control.payload_bytes = layout.payload_bytes;
        token->control.payload_address = reinterpret_cast<uint64_t>(task.payload.data());
        token->control.completion_vend = header.completion_vend;
        token->control.function_and_reference = static_cast<uint64_t>(TaskFunctionId(TaskKindAt(task_id)));
        token->control.shape_and_scalar_offset = PackExecutionTokenShapeAndScalarOffset(
            shape.tensor_count, shape.scalar_count, shape.fanin_count, static_cast<uint16_t>(layout.scalar_word_offset)
        );
        const std::array<uint64_t, kLocalContextBytes / sizeof(uint64_t)> local_context = {
            task_id, owner, layout.payload_bytes, layout.payload_lines, shape.fanin_count, header.completion_vend,
        };
        std::memcpy(token->dispatch.local_context, local_context.data(), sizeof(token->dispatch.local_context));
        std::memcpy(token->dispatch.global_context, &batches_, sizeof(batches_));
        token->dispatch.args[kDispatchLocalContextIndex] =
            reinterpret_cast<uint64_t>(&token->dispatch.local_context[0]);
        token->dispatch.args[kDispatchGlobalContextIndex] =
            reinterpret_cast<uint64_t>(&token->dispatch.global_context[0]);
        return true;
    }

    bool AdvanceToken(CpuOwner &worker, uint32_t slot) {
        ExecutionToken &token = worker.tokens[slot];
        if (token.control.phase == ExecTokenPhase::Idle) {
            return false;
        }
        const uint32_t task_id = token.control.task_id;
        CpuTask &task = tasks_[task_id];
        if (token.control.phase == ExecTokenPhase::WaitingBuilt) {
            uint64_t expected = task.exec_state.load(std::memory_order_acquire);
            if (expected == 0U) {
                return false;
            }
            const DecodedExecState decoded = DecodeExecState(static_cast<int64_t>(expected));
            if (decoded.valid && decoded.phase == ExecPhase::Building && decoded.task_id == task_id &&
                IsBuilderOwner(decoded.build_owner, builder_count_)) {
                return false;
            }
            if (!decoded.valid || decoded.phase != ExecPhase::Built || decoded.task_id != task_id ||
                !IsBuilderOwner(decoded.build_owner, builder_count_) ||
                decoded.engine_class != TaskEngine(TaskKindAt(task_id))) {
                PublishFatal(ExecFatalReason::ControlPublishConflict, task_id, worker.result.owner);
                return false;
            }
            const uint32_t build_owner = decoded.build_owner;
            if (expected != BuiltState(task_id, build_owner)) {
                PublishFatal(ExecFatalReason::ControlPublishConflict, task_id, worker.result.owner);
                return false;
            }
            if (!task.exec_state.compare_exchange_strong(
                    expected, ClaimedState(task_id, build_owner, worker.result.owner), std::memory_order_acq_rel,
                    std::memory_order_acquire
                )) {
                PublishFatal(ExecFatalReason::ControlPublishConflict, task_id, worker.result.owner);
                return false;
            }
            token.control.build_owner = build_owner;
            if (!PayloadValid(task_id)) {
                PublishFatal(ExecFatalReason::ClaimedPayloadInvalid, task_id, worker.result.owner);
                return false;
            }
            ++worker.result.claim_count;
            token.control.phase = ExecTokenPhase::Binding;
            if (!BindExecutionToken(&token, task_id, worker.result.owner)) {
                PublishFatal(ExecFatalReason::InvalidTokenPayload, task_id, worker.result.owner);
                return false;
            }
            token.control.phase = ExecTokenPhase::WaitingFanin;
        }
        if (token.control.phase == ExecTokenPhase::WaitingFanin) {
            const uint32_t fanin_count = TaskFaninCount(task_id);
            bool prefix_progress = false;
            while (token.control.fanin_ready_prefix < fanin_count) {
                const int32_t producer = TaskFanin(task_id, token.control.fanin_ready_prefix);
                if (producer < 0) {
                    PublishFatal(ExecFatalReason::InvalidTokenPayload, task_id, worker.result.owner);
                    return false;
                }
                if (tasks_[static_cast<uint32_t>(producer)].completion_flag.load(std::memory_order_acquire) != 1) {
                    return prefix_progress;
                }
                ++token.control.fanin_ready_prefix;
                prefix_progress = true;
            }
            token.control.phase = ExecTokenPhase::EngineInflight;
        }
        const TaskKind kind = TaskKindAt(task_id);
        const uint32_t fanin_count = TaskFaninCount(task_id);
        uint32_t ready_mask = 0U;
        if (token.control.fanin_ready_prefix != fanin_count) {
            PublishFatal(ExecFatalReason::InvalidTokenPayload, task_id, worker.result.owner);
            return false;
        }
        for (uint32_t edge = 0U; edge < fanin_count; ++edge) {
            const int32_t producer = TaskFanin(task_id, edge);
            if (producer < 0 ||
                tasks_[static_cast<uint32_t>(producer)].completion_flag.load(std::memory_order_acquire) != 1) {
                PublishFatal(ExecFatalReason::InvalidTokenPayload, task_id, worker.result.owner);
                return false;
            }
            ready_mask |= uint32_t{1} << edge;
        }
        const uint64_t fanin_witness = kFaninWitnessPresent | (static_cast<uint64_t>(fanin_count) << 32U) |
                                       (static_cast<uint64_t>(token.control.fanin_ready_prefix) << 16U) | ready_mask;
        uint64_t empty_fanin_witness = 0U;
        if (!task.fanin_timing_witness.compare_exchange_strong(
                empty_fanin_witness, fanin_witness, std::memory_order_release, std::memory_order_relaxed
            )) {
            PublishFatal(ExecFatalReason::ControlPublishConflict, task_id, worker.result.owner);
            return false;
        }
        const uint32_t kind_slot = kind == TaskKind::Pv || kind == TaskKind::Up ? 1U : 0U;
        owner_workload_[worker.result.owner][kind_slot] = RunCpuWorkload(kind);
        token.control.phase = ExecTokenPhase::Completing;
        const ExecPayloadHeader header = DecodeCpuHeader(task.payload);
        task.witness_nonce = nonce_;
        task.witness_magic = kExecutionWitnessMagic;
        task.witness_task = task_id;
        task.witness_kind = kind;
        task.witness_owner = worker.result.owner;
        task.execution_count = 1U;
        // This value is terminal evidence of the intended publication path, not an independent temporal trace.
        task.completion_sequence = kCompletionSequenceWorkloadWitnessVendFlagDone;
        task.output_checksum = PackOutputPair(owner_workload_[worker.result.owner][kind_slot]);
        task.witness_fanin_ready_prefix = token.control.fanin_ready_prefix;
        task.witness_state.store(
            ExecutionWitnessState(nonce_, task_id, kind, worker.result.owner), std::memory_order_release
        );
        task.completion_vend.store(header.completion_vend, std::memory_order_relaxed);
        if (task.completion_flag.exchange(1, std::memory_order_release) != 0) {
            PublishFatal(ExecFatalReason::CompletionPublishFailed, task_id, worker.result.owner);
            return false;
        }
        uint64_t expected_state = ClaimedState(task_id, token.control.build_owner, worker.result.owner);
        if (!task.exec_state.compare_exchange_strong(
                expected_state, DoneState(task_id, token.control.build_owner, worker.result.owner),
                std::memory_order_release, std::memory_order_acquire
            )) {
            PublishFatal(ExecFatalReason::CompletionStateConflict, task_id, worker.result.owner);
            return false;
        }
        ++worker.result.execute_count;
        ++worker.result.completed_by_kind[static_cast<uint32_t>(kind)];
        done_count_.fetch_add(1U, std::memory_order_acq_rel);
        if (TaskEngine(kind) == ExecEngineClass::Aic) {
            aic_done_.fetch_add(1U, std::memory_order_acq_rel);
        } else {
            aiv_done_.fetch_add(1U, std::memory_order_acq_rel);
        }
        ResetExecutionTokenControl(&token);
        --worker.busy_tokens;
        SignalProgress();
        return true;
    }

    std::array<float, 2> RunCpuWorkload(TaskKind kind) const {
        std::array<float, 2> outputs{};
        for (uint32_t output = 0U; output < outputs.size(); ++output) {
            if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
                float value = 0.0F;
                for (uint32_t column = 0U; column < kWorkloadTileColumns; ++column) {
                    value += workload_input_a_[column] * workload_input_b_[column];
                }
                outputs[output] = value;
            } else if (kind == TaskKind::Sf) {
                outputs[output] = workload_input_a_[output] + workload_input_b_[output];
            } else {
                outputs[output] = workload_input_a_[output] * workload_input_b_[output];
            }
        }
        return outputs;
    }

    bool ArriveDrain(uint32_t owner) {
        CpuOwner &worker = owners_[owner];
        worker.result.final_busy_tokens = worker.busy_tokens;
        if (worker.busy_tokens != 0U || worker.result.drain_arrival_count != 0U) {
            PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, owner);
            return false;
        }
        worker.result.drain_arrival_count = 1U;
        const int64_t contribution = EncodeDrainContribution(worker.result.execute_count);
        if (contribution < 0) {
            PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, owner);
            return false;
        }
        const uint32_t group = OwnerDrainGroup(owner);
        const int64_t previous = drain_arrivals_[group].fetch_add(contribution, std::memory_order_acq_rel);
        if (previous < 0 || DecodeDrainArrivals(previous) >= 6U) {
            PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, owner);
            return false;
        }
        SignalProgress();
        return true;
    }

    bool WaitForDrainAndPublishRoot() {
        const auto deadline = std::chrono::steady_clock::now() + kCpuWaitLimit;
        while (true) {
            const uint64_t observed_epoch = SnapshotProgressEpoch();
            if (fatal_.load(std::memory_order_acquire) != 0U) {
                return false;
            }
            bool ready = true;
            for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
                const int64_t raw = drain_arrivals_[group].load(std::memory_order_acquire);
                const uint32_t arrivals = DecodeDrainArrivals(raw);
                if (raw < 0 || arrivals > 6U) {
                    PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, kBuilderOwner);
                    return false;
                }
                ready &= arrivals == 6U;
            }
            if (ready) {
                std::array<int64_t, kDrainGroupCount> expected_group_raw{};
                uint64_t completed = 0U;
                for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
                    const int64_t contribution = EncodeDrainContribution(owners_[owner].result.execute_count);
                    if (contribution < 0) {
                        PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, kBuilderOwner);
                        return false;
                    }
                    expected_group_raw[OwnerDrainGroup(owner)] += contribution;
                }
                for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
                    const int64_t raw = drain_arrivals_[group].load(std::memory_order_acquire);
                    if (raw != expected_group_raw[group]) {
                        PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, kBuilderOwner);
                        return false;
                    }
                    completed += DecodeDrainCompletions(raw);
                }
                if (completed != kernel_task_count_) {
                    PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, kBuilderOwner);
                    return false;
                }
                uint32_t expected_root = 0U;
                if (!root_finished_.compare_exchange_strong(
                        expected_root, 1U, std::memory_order_release, std::memory_order_acquire
                    )) {
                    PublishFatal(ExecFatalReason::DrainMismatch, kNoTask, kBuilderOwner);
                    return false;
                }
                SignalProgress();
                return true;
            }
            if (!WaitForProgress(observed_epoch, deadline)) {
                PublishFatal(ExecFatalReason::Timeout, kNoTask, kBuilderOwner);
                return false;
            }
        }
    }

    bool ValidateCounts() const {
        if (fatal_.load(std::memory_order_acquire) != 0U ||
            builder_started_.load(std::memory_order_acquire) != builder_count_ ||
            builder_finished_.load(std::memory_order_acquire) != 1U ||
            done_count_.load(std::memory_order_acquire) != kernel_task_count_ ||
            alloc_done_.load(std::memory_order_acquire) != batches_ ||
            aic_done_.load(std::memory_order_acquire) != 2U * batches_ ||
            aiv_done_.load(std::memory_order_acquire) != 2U * batches_ ||
            committed_prefix_.load(std::memory_order_acquire) != batches_ ||
            aic_cursor_.load(std::memory_order_acquire) != 2U * batches_ + kAicOwnerCount ||
            aiv_cursor_.load(std::memory_order_acquire) != 2U * batches_ + AivExecutorCount(builder_count_)) {
            return false;
        }
        return aic_task_ids_.size() == 2U * batches_ && aiv_task_ids_.size() == 2U * batches_;
    }

    bool ValidateBuilderMapping() const {
        std::array<uint32_t, kMaxBuilderThreadCount> expected_wins{};
        std::array<uint32_t, kMaxBuilderThreadCount> expected_first{};
        std::array<uint32_t, kMaxBuilderThreadCount> expected_last{};
        std::array<uint32_t, kMaxBuilderThreadCount> expected_waits{};
        expected_first.fill(kNoTask);
        expected_last.fill(kNoTask);
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            const CpuTask &task = tasks_[task_id];
            const TaskKind kind = TaskKindAt(task_id);
            ExecPayloadLayout layout{};
            const TaskExecShape shape = TaskShape(kind);
            if (TaskExecutable(kind) &&
                !ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
                return false;
            }
            const uint32_t expected_phase_bits =
                kBuildPreparedBit | kBuildOutputsPublishedBit | kBuildInsertCommittedBit |
                (TaskExecutable(kind) ? kBuildExecPublishedBit : kBuildAllocCompletedBit);
            const uint32_t builder_thread = task.plan.builder_thread;
            if (builder_thread >= BuilderThreadCount(builder_count_) ||
                !BuilderThreadActive(builder_thread, builder_count_)) {
                return false;
            }
            const uint32_t build_owner = BuilderOwnerForThread(builder_thread);
            if (builder_thread != BuilderThreadForTask(task_id, builder_count_) ||
                !IsBuilderOwner(build_owner, builder_count_)) {
                return false;
            }
            if (TaskExecutable(kind)) {
                const DecodedExecState decoded =
                    DecodeExecState(static_cast<int64_t>(task.exec_state.load(std::memory_order_acquire)));
                if (!decoded.valid || decoded.build_owner != build_owner || decoded.task_id != task_id) {
                    return false;
                }
            } else if (task.completion_flag.load(std::memory_order_acquire) != 1) {
                return false;
            }
            if (!ConsecutiveTasksHaveSafeBuilderMapping(task_id, builder_count_) || task.plan.task_id != task_id ||
                task.plan.batch != TaskBatch(task_id) || task.plan.kind != TaskKindAt(task_id) ||
                task.plan.engine_class != TaskEngine(TaskKindAt(task_id)) ||
                task.plan.output_count != TaskOutputCount(kind) || task.plan.payload_lines != layout.payload_lines ||
                task.plan.builder_warp != BuilderWarp(builder_thread) ||
                task.plan.encoded_meta != EncodeTaskMeta(task_id, task_count_) ||
                task.plan.exec_route != EncodeTaskExecRoute(TaskKindAt(task_id)) ||
                task.plan.builder_owner != build_owner ||
                task.plan.reserved_bytes != AlignOutputBytes(TaskOutputBytes(kind)) ||
                task.plan.launch_nonce != nonce_ || task.plan.reserved != 0U || task.build_report.task_id != task_id ||
                task.build_report.builder_thread != builder_thread ||
                task.build_report.builder_warp != BuilderWarp(builder_thread) || task.build_report.builder_lane != 0U ||
                task.build_report.phase_bits != expected_phase_bits ||
                task.build_report.output_count != TaskOutputCount(kind) ||
                task.build_report.payload_words != layout.written_words || task.build_report.launch_nonce != nonce_ ||
                task.build_report.build_attempt_count != 1U || task.build_report.build_win_count != 1U ||
                task.build_report.prepare_count != 1U || task.build_report.commit_count != 1U ||
                task.insert_completion.load(std::memory_order_acquire) !=
                    (kind == TaskKind::Up ? static_cast<int64_t>(task_id) : InsertCompletionInitialValue(task_id))) {
                return false;
            }
            if (kind == TaskKind::Up && TaskBatch(task_id) != 0U) {
                if (task.build_report.predecessor_observed != static_cast<int64_t>(task_id - kTasksPerBatch) ||
                    task.build_report.insert_poll_count == 0U) {
                    return false;
                }
            } else if (task.build_report.predecessor_observed != -1 || task.build_report.insert_poll_count != 0U) {
                return false;
            }
            ++expected_wins[builder_thread];
            if (expected_first[builder_thread] == kNoTask) {
                expected_first[builder_thread] = task_id;
            }
            expected_last[builder_thread] = task_id;
            expected_waits[builder_thread] += kind == TaskKind::Up && TaskBatch(task_id) != 0U ? 1U : 0U;
        }

        uint64_t total_attempts = 0U;
        uint64_t total_wins = 0U;
        uint64_t total_losses = 0U;
        for (uint32_t thread = 0U; thread < BuilderThreadCount(builder_count_); ++thread) {
            const FullPaBuilderThreadReport &report = builder_threads_[thread];
            if (!BuilderThreadActive(thread, builder_count_)) {
                FullPaBuilderThreadReport poison{};
                std::memset(&poison, 0xD3, sizeof(poison));
                if (std::memcmp(&report, &poison, sizeof(report)) != 0) {
                    return false;
                }
                continue;
            }
            const uint32_t attempts = BuilderExpectedTaskCount(thread, task_count_, builder_count_);
            const uint32_t wins = expected_wins[thread];
            if (wins > attempts) {
                return false;
            }
            const uint32_t losses = 0U;
            const uint64_t checksum = BuilderReportChecksum(
                nonce_, thread, task_count_, wins, expected_first[thread], expected_last[thread], attempts, wins, wins,
                expected_waits[thread], losses
            );
            if (report.thread_id != thread || report.warp_id != BuilderWarp(thread) ||
                report.lane_id != thread % kWarpSize ||
                report.active_leader != (BuilderThreadActive(thread, builder_count_) ? 1U : 0U) ||
                report.task_count != wins || report.first_task != expected_first[thread] ||
                report.last_task != expected_last[thread] || report.task_state_access_count != attempts ||
                report.prepare_count != wins || report.commit_count != wins ||
                report.insert_wait_count != expected_waits[thread] || report.claim_lost_count != losses ||
                report.launch_nonce != nonce_ || report.checksum != checksum) {
                return false;
            }
            total_attempts += attempts;
            total_wins += wins;
            total_losses += losses;
        }
        return total_attempts == task_count_ && total_wins == task_count_ && total_losses == 0U;
    }

    bool ValidateHeapAndOutputs() const {
        std::array<std::vector<HeapInterval>, kSharedHeapShards> intervals;
        std::vector<HeapInterval> aggregate_intervals;
        std::vector<uint64_t> zero_reserve_vends;
        std::array<uint64_t, kSharedHeapShards> expected_bytes{};
        uint64_t expected_total = 0U;
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            const CpuTask &task = tasks_[task_id];
            const TaskKind kind = TaskKindAt(task_id);
            const uint64_t base_plus_one = task.task_base_plus_one.load(std::memory_order_acquire);
            const uint64_t vend_plus_one = task.completion_vend_plus_one.load(std::memory_order_acquire);
            if (base_plus_one == 0U || vend_plus_one == 0U) {
                return false;
            }
            const uint64_t task_base = base_plus_one - 1U;
            const uint64_t bytes = AlignOutputBytes(TaskOutputBytes(kind));
            if (bytes != 0U) {
                const uint32_t shard = TaskHeapShard(task_id);
                const uint64_t shard_begin = static_cast<uint64_t>(shard) * kHeapShardSpan;
                const uint64_t vend = vend_plus_one - 1U;
                if (task_base < shard_begin || task_base > shard_begin + kHeapShardSpan - bytes || vend < bytes) {
                    return false;
                }
                intervals[shard].push_back(HeapInterval{task_base, task_base + bytes, task_id});
                aggregate_intervals.push_back(HeapInterval{vend - bytes, vend, task_id});
                expected_bytes[shard] += bytes;
                expected_total += bytes;
            } else {
                const uint64_t vend = vend_plus_one - 1U;
                if (kind != TaskKind::Up || task_base != 0U || (vend % kCacheLineBytes) != 0U) {
                    return false;
                }
                zero_reserve_vends.push_back(vend);
            }
            const uint32_t output_count = TaskOutputCount(kind);
            for (uint32_t slot = 0U; slot < kOutputsPerTask; ++slot) {
                if (slot < output_count) {
                    int64_t expected_writer = task_id;
                    if (kind == TaskKind::Alloc && slot == 0U) {
                        expected_writer = BatchTaskId(TaskBatch(task_id), TaskKind::Up);
                    }
                    if (task.published[slot].load(std::memory_order_acquire) != static_cast<int64_t>(task_id) ||
                        task.last_writer[slot].load(std::memory_order_acquire) != expected_writer ||
                        !TensorEqual(task.outputs[slot], MakeTaskOutputDescriptor(task_id, slot, task_base))) {
                        return false;
                    }
                } else {
                    const TensorDesc zero{};
                    if (task.published[slot].load(std::memory_order_acquire) != -1 ||
                        task.last_writer[slot].load(std::memory_order_acquire) != -1 ||
                        !TensorEqual(task.outputs[slot], zero)) {
                        return false;
                    }
                }
            }
            if (kind == TaskKind::Up) {
                WriterHistoryCell expected{};
                expected.magic = kWriterHistoryMagic;
                expected.writer_task = static_cast<int32_t>(task_id);
                expected.count = 3U;
                for (uint32_t index = 0U; index < 3U; ++index) {
                    expected.entries[index] = WriterHistoryRecord{
                        WriterHistorySymbolKey(TaskBatch(task_id), index),
                        static_cast<int32_t>(BatchTaskId(TaskBatch(task_id), TaskKind::Alloc)),
                    };
                }
                if (std::memcmp(&task.history, &expected, sizeof(expected)) != 0) {
                    return false;
                }
            } else {
                const WriterHistoryCell zero{};
                if (std::memcmp(&task.history, &zero, sizeof(zero)) != 0) {
                    return false;
                }
            }
        }
        for (uint32_t shard = 0U; shard < kSharedHeapShards; ++shard) {
            std::sort(
                intervals[shard].begin(), intervals[shard].end(),
                [](const HeapInterval &lhs, const HeapInterval &rhs) {
                    return lhs.begin < rhs.begin;
                }
            );
            uint64_t cursor = static_cast<uint64_t>(shard) * kHeapShardSpan;
            for (const HeapInterval &interval : intervals[shard]) {
                if (interval.begin != cursor || interval.end <= interval.begin) {
                    return false;
                }
                cursor = interval.end;
            }
            if (heap_cursors_[shard].load(std::memory_order_acquire) != expected_bytes[shard] ||
                cursor != static_cast<uint64_t>(shard) * kHeapShardSpan + expected_bytes[shard]) {
                return false;
            }
        }
        std::sort(
            aggregate_intervals.begin(), aggregate_intervals.end(),
            [](const HeapInterval &lhs, const HeapInterval &rhs) {
                return lhs.begin < rhs.begin;
            }
        );
        uint64_t aggregate_cursor = 0U;
        for (const HeapInterval &interval : aggregate_intervals) {
            if (interval.begin != aggregate_cursor || interval.end <= interval.begin) {
                return false;
            }
            aggregate_cursor = interval.end;
        }
        for (uint64_t vend : zero_reserve_vends) {
            if (vend > expected_total) {
                return false;
            }
        }
        if (aggregate_cursor != expected_total || heap_vend_.load(std::memory_order_acquire) != expected_total) {
            return false;
        }
        return batches_ != kDefaultBatches ||
               (expected_total == kMainReservedHeapBytes &&
                std::all_of(expected_bytes.begin(), expected_bytes.end(), [](uint64_t bytes) {
                    return bytes == kMainReservedHeapBytesPerShard;
                }));
    }

    bool ValidatePayloads() const {
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            const TaskKind kind = TaskKindAt(task_id);
            if (TaskExecutable(kind)) {
                const uint64_t raw_state = tasks_[task_id].exec_state.load(std::memory_order_acquire);
                const DecodedExecState decoded = DecodeExecState(static_cast<int64_t>(raw_state));
                const TaskExecShape shape = TaskShape(kind);
                ExecPayloadLayout layout{};
                if (!ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
                    return false;
                }
                if (!decoded.valid || decoded.phase != ExecPhase::Done || decoded.task_id != task_id ||
                    !IsBuilderOwner(decoded.build_owner, builder_count_) ||
                    !OwnerCanExecute(decoded.execute_owner, TaskEngine(kind), builder_count_) ||
                    decoded.engine_class != TaskEngine(kind) || decoded.payload_lines != layout.payload_lines ||
                    raw_state != DoneState(task_id, decoded.build_owner, decoded.execute_owner) ||
                    !PayloadValid(task_id)) {
                    return false;
                }
            } else if (tasks_[task_id].exec_state.load(std::memory_order_acquire) != 0U) {
                return false;
            } else {
                for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
                    if (tasks_[task_id].payload[word] != ExpectedPayloadPoisonWord(nonce_, task_id, word)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool ValidateCompletionAndWitnesses() const {
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            const CpuTask &task = tasks_[task_id];
            const TaskKind kind = TaskKindAt(task_id);
            if (task.completion_flag.load(std::memory_order_acquire) != 1 ||
                task.completion_vend.load(std::memory_order_acquire) + 1U !=
                    task.completion_vend_plus_one.load(std::memory_order_acquire)) {
                return false;
            }
            if (!TaskExecutable(kind)) {
                if (task.witness_state.load(std::memory_order_acquire) != 0U || task.witness_nonce != 0U ||
                    task.witness_magic != 0U || task.witness_task != 0U || task.witness_kind != TaskKind::Alloc ||
                    task.witness_owner != 0U || task.execution_count != 0U || task.completion_sequence != 0U ||
                    task.output_checksum != 0U || task.witness_fanin_ready_prefix != 0U ||
                    task.fanin_timing_witness.load(std::memory_order_acquire) != 0U) {
                    return false;
                }
                continue;
            }
            const DecodedExecState decoded =
                DecodeExecState(static_cast<int64_t>(task.exec_state.load(std::memory_order_acquire)));
            const uint64_t expected_witness = ExecutionWitnessState(nonce_, task_id, kind, decoded.execute_owner);
            const uint32_t fanin_count = TaskFaninCount(task_id);
            const uint32_t ready_mask = fanin_count == 0U ? 0U : (uint32_t{1} << fanin_count) - 1U;
            const uint64_t expected_fanin_witness = kFaninWitnessPresent | (static_cast<uint64_t>(fanin_count) << 32U) |
                                                    (static_cast<uint64_t>(fanin_count) << 16U) | ready_mask;
            if (task.witness_state.load(std::memory_order_acquire) != expected_witness ||
                task.witness_nonce != nonce_ || task.witness_magic != kExecutionWitnessMagic ||
                task.witness_task != task_id || task.witness_kind != kind ||
                task.witness_owner != decoded.execute_owner || task.execution_count != 1U ||
                task.completion_sequence != kCompletionSequenceWorkloadWitnessVendFlagDone ||
                task.output_checksum != ExpectedWorkloadOutputPair(kind) ||
                task.witness_fanin_ready_prefix != fanin_count ||
                task.fanin_timing_witness.load(std::memory_order_acquire) != expected_fanin_witness) {
                return false;
            }
        }
        return true;
    }

    bool ValidateRetainedTokenBinding(const CpuOwner &worker, uint32_t slot, uint32_t expected_task) const {
        const ExecutionToken &token = worker.tokens[slot];
        if (token.control.phase != ExecTokenPhase::Idle || token.control.task_id != UINT32_MAX ||
            token.control.build_owner != UINT32_MAX || token.control.execute_owner != UINT32_MAX ||
            token.control.engine_class != ExecEngineClass::None || token.control.payload_lines != 0U ||
            token.control.payload_bytes != 0U || token.control.fanin_ready_prefix != 0U ||
            token.control.payload_address != 0U || token.control.completion_vend != 0U ||
            token.control.function_and_reference != 0U || token.control.shape_and_scalar_offset != 0U) {
            return false;
        }
        const ExecutionDispatchBinding zero_dispatch{};
        if (expected_task == kNoTask) {
            return std::memcmp(&token.dispatch, &zero_dispatch, sizeof(zero_dispatch)) == 0;
        }
        if (expected_task >= task_count_) {
            return false;
        }

        std::array<uint64_t, kLocalContextBytes / sizeof(uint64_t)> local_context{};
        std::memcpy(local_context.data(), token.dispatch.local_context, sizeof(token.dispatch.local_context));
        const uint32_t task_id = expected_task;
        const CpuTask &task = tasks_[task_id];
        const TaskKind kind = TaskKindAt(task_id);
        const TaskExecShape shape = TaskShape(kind);
        ExecPayloadLayout layout{};
        if (!TaskExecutable(kind) || TaskEngine(kind) != OwnerEngine(worker.result.owner, builder_count_) ||
            task.witness_owner != worker.result.owner ||
            !ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
            return false;
        }
        const ExecPayloadHeader header = DecodeCpuHeader(task.payload);
        const std::array<uint64_t, kLocalContextBytes / sizeof(uint64_t)> expected_local = {
            task_id,           worker.result.owner,    layout.payload_bytes, layout.payload_lines,
            shape.fanin_count, header.completion_vend,
        };
        uint32_t global_batches = 0U;
        std::memcpy(&global_batches, token.dispatch.global_context, sizeof(global_batches));
        if (local_context != expected_local || global_batches != batches_ ||
            token.dispatch.args[kDispatchLocalContextIndex] !=
                reinterpret_cast<uint64_t>(&token.dispatch.local_context[0]) ||
            token.dispatch.args[kDispatchGlobalContextIndex] !=
                reinterpret_cast<uint64_t>(&token.dispatch.global_context[0])) {
            return false;
        }
        for (uint32_t tensor = 0U; tensor < shape.tensor_count; ++tensor) {
            const uint32_t offset = layout.tensor_word_offset + tensor * kTensorDescWords;
            if (token.dispatch.args[tensor] != reinterpret_cast<uint64_t>(&task.payload[offset])) {
                return false;
            }
        }
        for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
            if (token.dispatch.args[shape.tensor_count + scalar] != task.payload[layout.scalar_word_offset + scalar]) {
                return false;
            }
        }
        for (uint32_t arg = 9U; arg < kDispatchLocalContextIndex; ++arg) {
            if (token.dispatch.args[arg] != 0U) {
                return false;
            }
        }
        return std::memcmp(token.dispatch.padding, zero_dispatch.padding, sizeof(token.dispatch.padding)) == 0;
    }

    bool ValidateRolesAndDrain() const {
        uint64_t total_execute = 0U;
        std::array<uint64_t, static_cast<uint32_t>(TaskKind::Count)> completed_by_kind{};
        std::array<std::array<uint32_t, static_cast<uint32_t>(TaskKind::Count)>, kOwnerCount> witness_by_owner{};
        std::array<int64_t, kDrainGroupCount> expected_group_raw{};
        std::vector<bool> retained_tasks(task_count_, false);
        const std::array<uint64_t, 5> zero_reserved{};
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            const CpuTask &task = tasks_[task_id];
            const TaskKind kind = TaskKindAt(task_id);
            if (!TaskExecutable(kind)) {
                continue;
            }
            if (task.witness_owner >= kOwnerCount || IsBuilderOwner(task.witness_owner, builder_count_)) {
                return false;
            }
            ++witness_by_owner[task.witness_owner][static_cast<uint32_t>(kind)];
        }
        for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
            const CpuOwner &worker = owners_[owner];
            const FullPaRoleResult &result = worker.result;
            if (result.owner != owner || result.role != OwnerRoleAt(owner, builder_count_) ||
                result.physical_block != OwnerPhysicalBlock(owner) || result.drain_group != OwnerDrainGroup(owner) ||
                result.final_busy_tokens != 0U || result.drain_arrival_count != 1U || result.claim_lost_count != 0U ||
                result.fatal_count != 0U || result.launch_nonce != nonce_ ||
                std::memcmp(result.reserved, zero_reserved.data(), sizeof(result.reserved)) != 0) {
                return false;
            }
            uint32_t used_tokens = 0U;
            for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
                const uint32_t expected_task = worker.last_ticket_tasks[slot];
                if (expected_task != kNoTask) {
                    if (expected_task >= task_count_ || retained_tasks[expected_task]) {
                        return false;
                    }
                    retained_tasks[expected_task] = true;
                    ++used_tokens;
                }
                if (!ValidateRetainedTokenBinding(worker, slot, expected_task)) {
                    return false;
                }
            }
            if (used_tokens != std::min(result.ticket_count, kTokensPerOwner)) {
                return false;
            }
            if (IsBuilderOwner(owner, builder_count_)) {
                if (result.build_count != 0U || result.commit_count != 0U || result.execute_count != 0U ||
                    result.ticket_count != 0U || result.exhausted_ticket_count != 0U || result.claim_count != 0U ||
                    result.max_busy_tokens != 0U) {
                    return false;
                }
            } else if (result.build_count != 0U || result.commit_count != 0U || result.exhausted_ticket_count != 1U ||
                       result.claim_count != result.ticket_count || result.execute_count != result.ticket_count ||
                       result.max_busy_tokens > kTokensPerOwner ||
                       result.max_busy_tokens != std::min(result.ticket_count, kTokensPerOwner)) {
                return false;
            }
            uint32_t owner_kind_total = 0U;
            for (uint32_t kind_index = 0U; kind_index < static_cast<uint32_t>(TaskKind::Count); ++kind_index) {
                const TaskKind kind = static_cast<TaskKind>(kind_index);
                const uint32_t count = result.completed_by_kind[kind_index];
                const bool legal = !IsBuilderOwner(owner, builder_count_) && TaskExecutable(kind) &&
                                   TaskEngine(kind) == OwnerEngine(owner, builder_count_);
                if ((!legal && count != 0U) || count != witness_by_owner[owner][kind_index]) {
                    return false;
                }
                owner_kind_total += count;
                completed_by_kind[kind_index] += count;
            }
            if (owner_kind_total != result.execute_count) {
                return false;
            }
            total_execute += result.execute_count;
            const int64_t contribution = EncodeDrainContribution(result.execute_count);
            if (contribution < 0) {
                return false;
            }
            expected_group_raw[OwnerDrainGroup(owner)] += contribution;
            if (!IsBuilderOwner(owner, builder_count_)) {
                const TaskKind kinds[2] = {
                    owner < kAicOwnerCount ? TaskKind::Qk : TaskKind::Sf,
                    owner < kAicOwnerCount ? TaskKind::Pv : TaskKind::Up,
                };
                for (uint32_t slot = 0U; slot < 2U; ++slot) {
                    const bool active = result.completed_by_kind[static_cast<uint32_t>(kinds[slot])] != 0U;
                    const uint64_t expected = active ? ExpectedWorkloadOutputPair(kinds[slot]) :
                                                       PackOutputPair(std::array<float, 2>{
                                                           kWorkloadOutputSentinel,
                                                           kWorkloadOutputSentinel,
                                                       });
                    if (PackOutputPair(owner_workload_[owner][slot]) != expected) {
                        return false;
                    }
                }
            }
        }
        if (total_execute != kernel_task_count_ || root_finished_.load(std::memory_order_acquire) != 1U ||
            completed_by_kind[static_cast<uint32_t>(TaskKind::Alloc)] != 0U ||
            completed_by_kind[static_cast<uint32_t>(TaskKind::Qk)] != batches_ ||
            completed_by_kind[static_cast<uint32_t>(TaskKind::Sf)] != batches_ ||
            completed_by_kind[static_cast<uint32_t>(TaskKind::Pv)] != batches_ ||
            completed_by_kind[static_cast<uint32_t>(TaskKind::Up)] != batches_) {
            return false;
        }
        uint64_t drain_completions = 0U;
        for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
            const int64_t raw = drain_arrivals_[group].load(std::memory_order_acquire);
            if (raw != expected_group_raw[group] || DecodeDrainArrivals(raw) != 6U) {
                return false;
            }
            drain_completions += DecodeDrainCompletions(raw);
        }
        return drain_completions == kernel_task_count_;
    }

    void InitializeTaskAbi(FullPaTask *task, uint32_t task_id) const {
        std::memset(task, 0, sizeof(*task));
        task->completion.deps_prepared = InsertCompletionInitialValue(task_id);
        task->insert_completion.value = InsertCompletionInitialValue(task_id);
        for (uint32_t slot = 0U; slot < kOutputsPerTask; ++slot) {
            task->outputs.published[slot].value = -1;
            task->outputs.last_writer[slot].value = -1;
        }
        for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
            task->exec.payload.words[word] = ExpectedPayloadPoisonWord(nonce_, task_id, word);
        }
    }

    void InitializeAbiState(FullPaState *state) const {
        std::memset(state, 0, sizeof(*state));
        std::memset(state->builder_threads, 0xD3, sizeof(state->builder_threads));
        state->control.magic = kProbeMagic;
        state->control.version = kProbeVersion;
        state->control.launch_nonce = nonce_;
        state->control.timeout_ticks = 2000000000ULL;
        state->control.batch_count = batches_;
        state->control.task_count = task_count_;
        state->control.kernel_task_count = kernel_task_count_;
        state->control.builder_thread_count = kBuilderThreadCount;
        state->control.heap_base = kSyntheticHeapBase;
        state->control.heap_bytes = kHeapBytes;
        state->control.workspace_base = 0x800000000ULL;
        state->control.workspace_bytes = kWorkloadBytes;
        state->control.qk_repeats = 1U;
        state->control.sf_repeats = 1U;
        state->control.pv_repeats = 1U;
        state->control.up_repeats = 1U;
        state->control.builder_count = builder_count_;
        for (uint32_t task_id = 0U; task_id < kMaxTasks; ++task_id) {
            InitializeTaskAbi(&state->tasks[task_id], task_id);
        }
        std::fill_n(state->exec_dispatch.aic_task_ids, kAicTaskCapacity, UINT32_MAX);
        std::fill_n(state->exec_dispatch.aiv_task_ids, kAivTaskCapacity, UINT32_MAX);
        for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
            for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
                ExecutionToken &token = state->tokens[owner][slot];
                token.control.phase = ExecTokenPhase::Idle;
                token.control.task_id = UINT32_MAX;
                token.control.build_owner = UINT32_MAX;
                token.control.execute_owner = UINT32_MAX;
                token.control.engine_class = ExecEngineClass::None;
            }
        }
        FullPaGuard *guards[] = {
            &state->guard_before_tasks,           &state->guard_after_tasks,           &state->guard_before_tokens,
            &state->guard_after_tokens,           &state->guard_before_roles,          &state->guard_after_roles,
            &state->guard_before_builder_threads, &state->guard_after_builder_threads,
        };
        for (uint32_t guard = 0U; guard < sizeof(guards) / sizeof(guards[0]); ++guard) {
            for (uint32_t word = 0U; word < kCacheLineBytes / sizeof(uint64_t); ++word) {
                guards[guard]->words[word] = ExpectedGuardWord(nonce_, guard, word);
            }
        }
    }

    void MaterializeActiveTasks(FullPaState *state) const {
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            const CpuTask &source = tasks_[task_id];
            FullPaTask &destination = state->tasks[task_id];
            destination.plan = source.plan;
            destination.completion.flag = source.completion_flag.load(std::memory_order_acquire);
            destination.completion.vend = source.completion_vend.load(std::memory_order_acquire);
            destination.insert_completion.value = source.insert_completion.load(std::memory_order_acquire);
            destination.allocation.task_base_plus_one.value =
                static_cast<int64_t>(source.task_base_plus_one.load(std::memory_order_acquire));
            destination.allocation.completion_vend_plus_one.value =
                static_cast<int64_t>(source.completion_vend_plus_one.load(std::memory_order_acquire));
            for (uint32_t slot = 0U; slot < kOutputsPerTask; ++slot) {
                destination.outputs.published[slot].value = source.published[slot].load(std::memory_order_acquire);
                destination.outputs.last_writer[slot].value = source.last_writer[slot].load(std::memory_order_acquire);
                destination.outputs.tensors[slot] = source.outputs[slot];
            }
            destination.writer_history = source.history;
            destination.exec.control.state = static_cast<int64_t>(source.exec_state.load(std::memory_order_acquire));
            for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
                destination.exec.payload.words[word] = source.payload[word];
            }
            destination.build_report = source.build_report;
            destination.execution_witness.state =
                static_cast<int64_t>(source.witness_state.load(std::memory_order_acquire));
            destination.execution_witness.launch_nonce = source.witness_nonce;
            destination.execution_witness.witness_magic = source.witness_magic;
            destination.execution_witness.task_id = source.witness_task;
            destination.execution_witness.kind = source.witness_kind;
            destination.execution_witness.execute_owner = source.witness_owner;
            destination.execution_witness.execution_count = source.execution_count;
            destination.execution_witness.completion_sequence = source.completion_sequence;
            destination.execution_witness.output_checksum = source.output_checksum;
            destination.execution_witness.fanin_ready_prefix = source.witness_fanin_ready_prefix;
        }
    }

    void MaterializeGlobalState(FullPaState *state) const {
        state->fatal.state = static_cast<int64_t>(fatal_.load(std::memory_order_acquire));
        state->drain.builder_started.value = static_cast<int64_t>(builder_started_.load(std::memory_order_acquire));
        state->drain.builder_finished.value = static_cast<int64_t>(builder_finished_.load(std::memory_order_acquire));
        state->drain.done_count.value = static_cast<int64_t>(done_count_.load(std::memory_order_acquire));
        state->drain.alloc_done.value = static_cast<int64_t>(alloc_done_.load(std::memory_order_acquire));
        state->drain.aic_done.value = static_cast<int64_t>(aic_done_.load(std::memory_order_acquire));
        state->drain.aiv_done.value = static_cast<int64_t>(aiv_done_.load(std::memory_order_acquire));
        state->drain.root_finished.value = root_finished_.load(std::memory_order_acquire);
        for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
            state->drain.arrivals[group].value = drain_arrivals_[group].load(std::memory_order_acquire);
        }
        state->exec_dispatch.aic_next.value = aic_cursor_.load(std::memory_order_acquire);
        state->exec_dispatch.aiv_next.value = aiv_cursor_.load(std::memory_order_acquire);
        state->exec_dispatch.aic_task_count = static_cast<uint32_t>(aic_task_ids_.size());
        state->exec_dispatch.aiv_task_count = static_cast<uint32_t>(aiv_task_ids_.size());
        std::copy(aic_task_ids_.begin(), aic_task_ids_.end(), state->exec_dispatch.aic_task_ids);
        std::copy(aiv_task_ids_.begin(), aiv_task_ids_.end(), state->exec_dispatch.aiv_task_ids);
        for (uint32_t shard = 0U; shard < kSharedHeapShards; ++shard) {
            state->heap.shard_cursors[shard].value = heap_cursors_[shard].load(std::memory_order_acquire);
        }
        state->heap.aggregate_vend.value = heap_vend_.load(std::memory_order_acquire);
        for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
            state->roles[owner] = owners_[owner].result;
            for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
                ExecutionToken &destination = state->tokens[owner][slot];
                destination = owners_[owner].tokens[slot];
                const uint32_t task_id = owners_[owner].last_ticket_tasks[slot];
                if (task_id == kNoTask) {
                    continue;
                }
                std::memset(&destination.dispatch, 0, sizeof(destination.dispatch));
                const TaskExecShape shape = TaskShape(TaskKindAt(task_id));
                ExecPayloadLayout layout{};
                (void)ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout);
                for (uint32_t tensor = 0U; tensor < shape.tensor_count; ++tensor) {
                    const uint32_t offset = layout.tensor_word_offset + tensor * kTensorDescWords;
                    destination.dispatch.args[tensor] =
                        reinterpret_cast<uint64_t>(&state->tasks[task_id].exec.payload.words[offset]);
                }
                for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
                    destination.dispatch.args[shape.tensor_count + scalar] =
                        state->tasks[task_id].exec.payload.words[layout.scalar_word_offset + scalar];
                }
                const std::array<uint64_t, kLocalContextBytes / sizeof(uint64_t)> local_context = {
                    task_id,
                    owner,
                    layout.payload_bytes,
                    layout.payload_lines,
                    shape.fanin_count,
                    state->tasks[task_id].exec.payload.words[2],
                };
                std::memcpy(
                    destination.dispatch.local_context, local_context.data(), sizeof(destination.dispatch.local_context)
                );
                std::memcpy(destination.dispatch.global_context, &batches_, sizeof(batches_));
                destination.dispatch.args[kDispatchLocalContextIndex] =
                    reinterpret_cast<uint64_t>(&destination.dispatch.local_context[0]);
                destination.dispatch.args[kDispatchGlobalContextIndex] =
                    reinterpret_cast<uint64_t>(&destination.dispatch.global_context[0]);
            }
        }
        for (uint32_t thread = 0U; thread < BuilderThreadCount(builder_count_); ++thread) {
            if (BuilderThreadActive(thread, builder_count_)) {
                state->builder_threads[thread] = builder_threads_[thread];
            }
        }
    }

    bool GuardsValid(const FullPaState &state) const {
        const FullPaGuard *guards[] = {
            &state.guard_before_tasks,           &state.guard_after_tasks,           &state.guard_before_tokens,
            &state.guard_after_tokens,           &state.guard_before_roles,          &state.guard_after_roles,
            &state.guard_before_builder_threads, &state.guard_after_builder_threads,
        };
        for (uint32_t guard = 0U; guard < sizeof(guards) / sizeof(guards[0]); ++guard) {
            for (uint32_t word = 0U; word < kCacheLineBytes / sizeof(uint64_t); ++word) {
                if (guards[guard]->words[word] != ExpectedGuardWord(nonce_, guard, word)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool ValidateTailTask(const FullPaTask &task, uint32_t task_id) const {
        const TensorDesc zero_tensor{};
        const FullPaTaskPlan zero_plan{};
        const FullPaBuildReport zero_report{};
        const WriterHistoryCell zero_history{};
        const FullPaExecutionWitness &witness = task.execution_witness;
        if (std::memcmp(&task.plan, &zero_plan, sizeof(zero_plan)) != 0 || task.completion.flag != 0 ||
            task.completion.vend != 0U || task.completion.deps_prepared != InsertCompletionInitialValue(task_id) ||
            task.insert_completion.value != InsertCompletionInitialValue(task_id) ||
            task.allocation.task_base_plus_one.value != 0 || task.allocation.completion_vend_plus_one.value != 0 ||
            task.exec.control.state != 0 ||
            std::memcmp(&task.writer_history, &zero_history, sizeof(zero_history)) != 0 || witness.state != 0 ||
            witness.launch_nonce != 0U || witness.witness_magic != 0U || witness.task_id != 0U ||
            witness.kind != TaskKind::Alloc || witness.execute_owner != 0U || witness.execution_count != 0U ||
            witness.completion_sequence != 0U || witness.output_checksum != 0U || witness.fanin_ready_prefix != 0U ||
            std::memcmp(&task.build_report, &zero_report, sizeof(zero_report)) != 0) {
            return false;
        }
        for (uint32_t slot = 0U; slot < kOutputsPerTask; ++slot) {
            if (task.outputs.published[slot].value != -1 || task.outputs.last_writer[slot].value != -1 ||
                !TensorEqual(task.outputs.tensors[slot], zero_tensor)) {
                return false;
            }
        }
        for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
            if (task.exec.payload.words[word] != ExpectedPayloadPoisonWord(nonce_, task_id, word)) {
                return false;
            }
        }
        return true;
    }

    bool ValidateMaterializedActiveTask(const FullPaTask &task, uint32_t task_id) const {
        const CpuTask &source = tasks_[task_id];
        const FullPaExecutionWitness &witness = task.execution_witness;
        if (std::memcmp(&task.plan, &source.plan, sizeof(task.plan)) != 0 ||
            task.completion.flag != source.completion_flag.load(std::memory_order_acquire) ||
            task.completion.vend != source.completion_vend.load(std::memory_order_acquire) ||
            task.completion.deps_prepared != InsertCompletionInitialValue(task_id) ||
            task.insert_completion.value != source.insert_completion.load(std::memory_order_acquire) ||
            task.allocation.task_base_plus_one.value !=
                static_cast<int64_t>(source.task_base_plus_one.load(std::memory_order_acquire)) ||
            task.allocation.completion_vend_plus_one.value !=
                static_cast<int64_t>(source.completion_vend_plus_one.load(std::memory_order_acquire)) ||
            std::memcmp(&task.writer_history, &source.history, sizeof(task.writer_history)) != 0 ||
            task.exec.control.state != static_cast<int64_t>(source.exec_state.load(std::memory_order_acquire)) ||
            std::memcmp(&task.build_report, &source.build_report, sizeof(task.build_report)) != 0 ||
            witness.state != static_cast<int64_t>(source.witness_state.load(std::memory_order_acquire)) ||
            witness.launch_nonce != source.witness_nonce || witness.witness_magic != source.witness_magic ||
            witness.task_id != source.witness_task || witness.kind != source.witness_kind ||
            witness.execute_owner != source.witness_owner || witness.execution_count != source.execution_count ||
            witness.completion_sequence != source.completion_sequence ||
            witness.output_checksum != source.output_checksum ||
            witness.fanin_ready_prefix != source.witness_fanin_ready_prefix) {
            return false;
        }
        for (uint32_t slot = 0U; slot < kOutputsPerTask; ++slot) {
            if (task.outputs.published[slot].value != source.published[slot].load(std::memory_order_acquire) ||
                task.outputs.last_writer[slot].value != source.last_writer[slot].load(std::memory_order_acquire) ||
                !TensorEqual(task.outputs.tensors[slot], source.outputs[slot])) {
                return false;
            }
        }
        for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
            if (task.exec.payload.words[word] != source.payload[word]) {
                return false;
            }
        }
        return true;
    }

    bool ValidateMaterializedToken(const FullPaState &state, uint32_t owner, uint32_t slot) const {
        const ExecutionToken &token = state.tokens[owner][slot];
        if (token.control.phase != ExecTokenPhase::Idle || token.control.task_id != UINT32_MAX ||
            token.control.build_owner != UINT32_MAX || token.control.execute_owner != UINT32_MAX ||
            token.control.engine_class != ExecEngineClass::None || token.control.payload_lines != 0U ||
            token.control.payload_bytes != 0U || token.control.fanin_ready_prefix != 0U ||
            token.control.payload_address != 0U || token.control.completion_vend != 0U ||
            token.control.function_and_reference != 0U || token.control.shape_and_scalar_offset != 0U) {
            return false;
        }
        const ExecutionDispatchBinding zero_dispatch{};
        const uint32_t task_id = owners_[owner].last_ticket_tasks[slot];
        if (task_id == kNoTask) {
            return std::memcmp(&token.dispatch, &zero_dispatch, sizeof(zero_dispatch)) == 0;
        }
        if (task_id >= task_count_) {
            return false;
        }
        const FullPaTask &task = state.tasks[task_id];
        const TaskKind kind = TaskKindAt(task_id);
        const TaskExecShape shape = TaskShape(kind);
        ExecPayloadLayout layout{};
        if (!TaskExecutable(kind) || TaskEngine(kind) != OwnerEngine(owner, builder_count_) ||
            task.execution_witness.execute_owner != owner ||
            !ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
            return false;
        }
        std::array<uint64_t, kLocalContextBytes / sizeof(uint64_t)> local_context{};
        std::memcpy(local_context.data(), token.dispatch.local_context, sizeof(token.dispatch.local_context));
        const std::array<uint64_t, kLocalContextBytes / sizeof(uint64_t)> expected_local = {
            task_id, owner, layout.payload_bytes, layout.payload_lines, shape.fanin_count, task.exec.payload.words[2],
        };
        uint32_t global_batches = 0U;
        std::memcpy(&global_batches, token.dispatch.global_context, sizeof(global_batches));
        if (local_context != expected_local || global_batches != batches_ ||
            token.dispatch.args[kDispatchLocalContextIndex] !=
                reinterpret_cast<uint64_t>(&token.dispatch.local_context[0]) ||
            token.dispatch.args[kDispatchGlobalContextIndex] !=
                reinterpret_cast<uint64_t>(&token.dispatch.global_context[0])) {
            return false;
        }
        for (uint32_t tensor = 0U; tensor < shape.tensor_count; ++tensor) {
            const uint32_t offset = layout.tensor_word_offset + tensor * kTensorDescWords;
            if (token.dispatch.args[tensor] !=
                reinterpret_cast<uint64_t>(&state.tasks[task_id].exec.payload.words[offset])) {
                return false;
            }
        }
        for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
            if (token.dispatch.args[shape.tensor_count + scalar] !=
                state.tasks[task_id].exec.payload.words[layout.scalar_word_offset + scalar]) {
                return false;
            }
        }
        for (uint32_t arg = shape.tensor_count + shape.scalar_count; arg < kDispatchLocalContextIndex; ++arg) {
            if (token.dispatch.args[arg] != 0U) {
                return false;
            }
        }
        return std::memcmp(token.dispatch.padding, zero_dispatch.padding, sizeof(token.dispatch.padding)) == 0;
    }

    bool ValidateMaterializedGlobals(const FullPaState &state) const {
        if (state.fatal.state != static_cast<int64_t>(fatal_.load(std::memory_order_acquire)) ||
            state.drain.builder_started.value !=
                static_cast<int64_t>(builder_started_.load(std::memory_order_acquire)) ||
            state.drain.builder_finished.value !=
                static_cast<int64_t>(builder_finished_.load(std::memory_order_acquire)) ||
            state.drain.done_count.value != static_cast<int64_t>(done_count_.load(std::memory_order_acquire)) ||
            state.drain.alloc_done.value != static_cast<int64_t>(alloc_done_.load(std::memory_order_acquire)) ||
            state.drain.aic_done.value != static_cast<int64_t>(aic_done_.load(std::memory_order_acquire)) ||
            state.drain.aiv_done.value != static_cast<int64_t>(aiv_done_.load(std::memory_order_acquire)) ||
            state.drain.root_finished.value != static_cast<int64_t>(root_finished_.load(std::memory_order_acquire)) ||
            state.exec_dispatch.aic_next.value != static_cast<int64_t>(aic_cursor_.load(std::memory_order_acquire)) ||
            state.exec_dispatch.aiv_next.value != static_cast<int64_t>(aiv_cursor_.load(std::memory_order_acquire)) ||
            state.exec_dispatch.aic_task_count != aic_task_ids_.size() ||
            state.exec_dispatch.aiv_task_count != aiv_task_ids_.size() ||
            state.heap.aggregate_vend.value != static_cast<int64_t>(heap_vend_.load(std::memory_order_acquire))) {
            return false;
        }
        for (uint8_t byte : state.exec_dispatch.header_padding) {
            if (byte != 0U) {
                return false;
            }
        }
        for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
            if (state.drain.arrivals[group].value != drain_arrivals_[group].load(std::memory_order_acquire)) {
                return false;
            }
        }
        for (uint32_t shard = 0U; shard < kSharedHeapShards; ++shard) {
            if (state.heap.shard_cursors[shard].value !=
                static_cast<int64_t>(heap_cursors_[shard].load(std::memory_order_acquire))) {
                return false;
            }
        }
        for (uint32_t owner = 0U; owner < kOwnerCount; ++owner) {
            if (std::memcmp(&state.roles[owner], &owners_[owner].result, sizeof(state.roles[owner])) != 0) {
                return false;
            }
            for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
                if (!ValidateMaterializedToken(state, owner, slot)) {
                    return false;
                }
            }
        }
        FullPaBuilderThreadReport poison{};
        std::memset(&poison, 0xD3, sizeof(poison));
        for (uint32_t thread = 0U; thread < kMaxBuilderThreadCount; ++thread) {
            const bool active =
                thread < BuilderThreadCount(builder_count_) && BuilderThreadActive(thread, builder_count_);
            const FullPaBuilderThreadReport &expected = active ? builder_threads_[thread] : poison;
            if (std::memcmp(&state.builder_threads[thread], &expected, sizeof(expected)) != 0) {
                return false;
            }
        }
        return true;
    }

    bool ValidateAbiState(const FullPaState &state) const {
        if (state.control.magic != kProbeMagic || state.control.version != kProbeVersion ||
            state.control.launch_nonce != nonce_ || state.control.batch_count != batches_ ||
            state.control.task_count != task_count_ || state.control.kernel_task_count != kernel_task_count_ ||
            state.control.timeout_ticks != 2000000000ULL || state.control.builder_thread_count != kBuilderThreadCount ||
            state.control.heap_base != kSyntheticHeapBase || state.control.heap_bytes != kHeapBytes ||
            state.control.workspace_base != 0x800000000ULL || state.control.workspace_bytes != kWorkloadBytes ||
            state.control.qk_repeats != 1U || state.control.sf_repeats != 1U || state.control.pv_repeats != 1U ||
            state.control.up_repeats != 1U || state.control.builder_count != builder_count_ ||
            state.control.reserved32 != 0U || state.fatal.state != 0 || !GuardsValid(state) ||
            state.ordinary_map.head.value != 0 || state.ordinary_map.tail.value != 0 ||
            state.ordinary_map.lookup_count.value != 0 || state.ordinary_map.append_count.value != 0) {
            return false;
        }
        for (uint64_t reserved : state.control.reserved) {
            if (reserved != 0U) {
                return false;
            }
        }
        for (uint32_t task_id = 0U; task_id < task_count_; ++task_id) {
            if (!ValidateMaterializedActiveTask(state.tasks[task_id], task_id)) {
                return false;
            }
        }
        for (uint32_t task_id = task_count_; task_id < kMaxTasks; ++task_id) {
            if (!ValidateTailTask(state.tasks[task_id], task_id)) {
                return false;
            }
        }
        for (uint32_t index = 0U; index < aic_task_ids_.size(); ++index) {
            if (state.exec_dispatch.aic_task_ids[index] != AicDispatchTaskId(index)) {
                return false;
            }
        }
        for (uint32_t index = static_cast<uint32_t>(aic_task_ids_.size()); index < kAicTaskCapacity; ++index) {
            if (state.exec_dispatch.aic_task_ids[index] != UINT32_MAX) {
                return false;
            }
        }
        for (uint32_t index = 0U; index < aiv_task_ids_.size(); ++index) {
            if (state.exec_dispatch.aiv_task_ids[index] != AivDispatchTaskId(index)) {
                return false;
            }
        }
        for (uint32_t index = static_cast<uint32_t>(aiv_task_ids_.size()); index < kAivTaskCapacity; ++index) {
            if (state.exec_dispatch.aiv_task_ids[index] != UINT32_MAX) {
                return false;
            }
        }
        return ValidateMaterializedGlobals(state);
    }

    uint32_t batches_;
    uint32_t task_count_;
    uint32_t kernel_task_count_;
    uint32_t builder_count_;
    uint64_t nonce_;
    std::unique_ptr<CpuTask[]> tasks_;
    std::array<std::atomic<uint64_t>, kSharedHeapShards> heap_cursors_{};
    std::atomic<uint64_t> heap_vend_{0U};
    std::atomic<uint64_t> fatal_{0U};
    std::atomic<uint64_t> builder_started_{0U};
    std::atomic<uint64_t> builder_finished_{0U};
    std::atomic<uint64_t> done_count_{0U};
    std::atomic<uint64_t> alloc_done_{0U};
    std::atomic<uint64_t> aic_done_{0U};
    std::atomic<uint64_t> aiv_done_{0U};
    std::atomic<uint32_t> aic_cursor_{0U};
    std::atomic<uint32_t> aiv_cursor_{0U};
    std::atomic<uint32_t> committed_prefix_{0U};
    std::vector<uint32_t> aic_task_ids_;
    std::vector<uint32_t> aiv_task_ids_;
    std::array<CpuOwner, kOwnerCount> owners_{};
    std::array<FullPaBuilderThreadReport, kMaxBuilderThreadCount> builder_threads_{};
    std::array<float, kWorkloadTileColumns> workload_input_a_{};
    std::array<float, kWorkloadTileColumns> workload_input_b_{};
    std::array<std::array<std::array<float, 2>, 2>, kOwnerCount> owner_workload_{};
    std::array<std::atomic<int64_t>, kDrainGroupCount> drain_arrivals_{};
    std::atomic<uint32_t> root_finished_{0U};
    std::mutex progress_mutex_;
    std::condition_variable progress_condition_;
    uint64_t progress_epoch_ = 0U;
};

bool TestResetPreservesDispatch() {
    ExecutionToken token{};
    token.control.phase = ExecTokenPhase::Completing;
    token.control.task_id = 17U;
    token.control.build_owner = kBuilderOwner;
    token.control.execute_owner = 3U;
    token.control.engine_class = ExecEngineClass::Aic;
    token.control.payload_lines = 9U;
    token.control.payload_bytes = 513U;
    token.control.fanin_ready_prefix = 2U;
    token.control.payload_address = UINT64_C(0x12340000);
    token.control.completion_vend = UINT64_C(0x56780000);
    token.control.function_and_reference = UINT64_C(0x9ABC0000);
    token.control.shape_and_scalar_offset = UINT64_C(0xDEF00000);
    std::memset(&token.dispatch, 0x5A, sizeof(token.dispatch));
    const ExecutionDispatchBinding retained = token.dispatch;

    ResetExecutionTokenControl(&token);
    return token.control.phase == ExecTokenPhase::Idle && token.control.task_id == UINT32_MAX &&
           token.control.build_owner == UINT32_MAX && token.control.execute_owner == UINT32_MAX &&
           token.control.engine_class == ExecEngineClass::None && token.control.payload_lines == 0U &&
           token.control.payload_bytes == 0U && token.control.fanin_ready_prefix == 0U &&
           token.control.payload_address == 0U && token.control.completion_vend == 0U &&
           token.control.function_and_reference == 0U && token.control.shape_and_scalar_offset == 0U &&
           std::memcmp(&token.dispatch, &retained, sizeof(retained)) == 0;
}

bool TestCompetingBuilderClaimValidation() {
    constexpr uint32_t kBuilderCount = 2U;
    constexpr uint32_t kCurrentBuilder = kBuilderOwner;
    constexpr uint32_t kOtherBuilder = kBuilderOwner + 1U;
    constexpr uint32_t kQkTask = static_cast<uint32_t>(TaskKind::Qk);
    constexpr uint32_t kUpTask = static_cast<uint32_t>(TaskKind::Up);
    constexpr uint64_t kNonce = UINT64_C(0xA5000000B17D0001);

    const TaskExecShape qk_shape = TaskShape(TaskKind::Qk);
    const TaskExecShape up_shape = TaskShape(TaskKind::Up);
    ExecPayloadLayout qk_layout{};
    ExecPayloadLayout up_layout{};
    if (!ComputeExecPayloadLayout(
            qk_shape.tensor_count, qk_shape.scalar_count, qk_shape.fanin_count, qk_layout
        ) ||
        !ComputeExecPayloadLayout(up_shape.tensor_count, up_shape.scalar_count, up_shape.fanin_count, up_layout)) {
        return false;
    }

    const bool legal_kernel_states =
        CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, BuildingState(kQkTask, kOtherBuilder)) &&
        CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, BuiltState(kQkTask, kOtherBuilder)) &&
        CompetingKernelClaimValid(
            kQkTask, kCurrentBuilder, kBuilderCount, ClaimedState(kQkTask, kOtherBuilder, 0U)
        ) &&
        CompetingKernelClaimValid(
            kQkTask, kCurrentBuilder, kBuilderCount, DoneState(kQkTask, kOtherBuilder, 0U)
        ) &&
        CompetingKernelClaimValid(kUpTask, kCurrentBuilder, kBuilderCount, BuiltState(kUpTask, kOtherBuilder)) &&
        CompetingKernelClaimValid(
            kUpTask, kCurrentBuilder, kBuilderCount,
            ClaimedState(kUpTask, kOtherBuilder, kBuilderOwner + kBuilderCount)
        );

    const uint64_t wrong_qk_engine = EncodeExecState(
        ExecPhase::Built, kOtherBuilder, kUnboundOwner, ExecEngineClass::Aiv, qk_layout.payload_lines, kQkTask
    );
    const uint64_t wrong_qk_lines = EncodeExecState(
        ExecPhase::Built, kOtherBuilder, kUnboundOwner, ExecEngineClass::Aic, qk_layout.payload_lines + 1U, kQkTask
    );
    const uint64_t bound_built = EncodeExecState(
        ExecPhase::Built, kOtherBuilder, 0U, ExecEngineClass::Aic, qk_layout.payload_lines, kQkTask
    );
    const uint64_t wrong_qk_route = EncodeExecState(
        ExecPhase::Claimed, kOtherBuilder, kBuilderOwner + kBuilderCount, ExecEngineClass::Aic,
        qk_layout.payload_lines, kQkTask
    );
    const uint64_t wrong_up_route = EncodeExecState(
        ExecPhase::Done, kOtherBuilder, 0U, ExecEngineClass::Aiv, up_layout.payload_lines, kUpTask
    );
    const uint64_t builder_as_executor = EncodeExecState(
        ExecPhase::Claimed, kOtherBuilder, kOtherBuilder, ExecEngineClass::Aiv, up_layout.payload_lines, kUpTask
    );
    const bool illegal_kernel_states =
        !CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, 0U) &&
        !CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, BuildingState(kQkTask, kCurrentBuilder)) &&
        !CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, BuildingState(kQkTask + 5U, kOtherBuilder)) &&
        !CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, wrong_qk_engine) &&
        !CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, wrong_qk_lines) &&
        !CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, bound_built) &&
        !CompetingKernelClaimValid(kQkTask, kCurrentBuilder, kBuilderCount, wrong_qk_route) &&
        !CompetingKernelClaimValid(kUpTask, kCurrentBuilder, kBuilderCount, wrong_up_route) &&
        !CompetingKernelClaimValid(kUpTask, kCurrentBuilder, kBuilderCount, builder_as_executor) &&
        !CompetingKernelClaimValid(
            kQkTask, kCurrentBuilder, kBuilderCount, BuiltState(kQkTask, kOtherBuilder) | (uint64_t{1} << 63U)
        );

    const int64_t other_alloc = static_cast<int64_t>(AllocBuildingState(kNonce, 0U, kOtherBuilder));
    const bool legal_alloc_states =
        CompetingAllocClaimValid(kNonce, 0U, kCurrentBuilder, kBuilderCount, 1) &&
        CompetingAllocClaimValid(kNonce, 0U, kCurrentBuilder, kBuilderCount, other_alloc);
    const bool illegal_alloc_states =
        !CompetingAllocClaimValid(
            kNonce, 0U, kCurrentBuilder, kBuilderCount,
            static_cast<int64_t>(AllocBuildingState(kNonce, 0U, kCurrentBuilder))
        ) &&
        !CompetingAllocClaimValid(kNonce + 1U, 0U, kCurrentBuilder, kBuilderCount, other_alloc) &&
        !CompetingAllocClaimValid(kNonce, 5U, kCurrentBuilder, kBuilderCount, other_alloc) &&
        !CompetingAllocClaimValid(
            kNonce, 0U, kCurrentBuilder, kBuilderCount,
            static_cast<int64_t>(AllocBuildingState(kNonce, 0U, kBuilderOwner + kBuilderCount))
        ) &&
        !CompetingAllocClaimValid(kNonce, 0U, kCurrentBuilder, kBuilderCount, 2);

    return legal_kernel_states && illegal_kernel_states && legal_alloc_states && illegal_alloc_states;
}

bool TestHalfPacketAndUniqueClaim() {
    std::atomic<uint64_t> state{BuildingState(1U)};
    std::array<uint64_t, 32> payload{};
    std::atomic<uint32_t> payload_reads{0U};
    PausePoint pause;
    std::thread builder([&] {
        for (uint32_t word = 0U; word < payload.size(); ++word) {
            payload[word] = 0xA500000000000000ULL | word;
            if (word + 1U == payload.size() / 2U) {
                pause.Stop();
            }
        }
        state.store(BuiltState(1U), std::memory_order_release);
    });
    if (!pause.WaitUntilReached()) {
        pause.Release();
        builder.join();
        return false;
    }
    uint64_t expected = BuiltState(1U);
    const bool premature_claim = state.compare_exchange_strong(
        expected, ClaimedState(1U, 0U), std::memory_order_acq_rel, std::memory_order_acquire
    );
    if (premature_claim) {
        payload_reads.fetch_add(1U, std::memory_order_relaxed);
    }
    pause.Release();
    builder.join();
    if (premature_claim || payload_reads.load(std::memory_order_relaxed) != 0U ||
        state.load(std::memory_order_acquire) != BuiltState(1U)) {
        return false;
    }
    std::atomic<uint32_t> winners{0U};
    std::vector<std::thread> claimants;
    for (uint32_t owner = 0U; owner < 8U; ++owner) {
        claimants.emplace_back([&, owner] {
            uint64_t built = BuiltState(1U);
            if (state.compare_exchange_strong(
                    built, ClaimedState(1U, owner), std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                winners.fetch_add(1U, std::memory_order_relaxed);
                for (uint32_t index = 0U; index < payload.size(); ++index) {
                    const uint64_t expected_word = 0xA500000000000000ULL | index;
                    if (payload[index] != expected_word) {
                        payload_reads.fetch_add(1000U, std::memory_order_relaxed);
                        return;
                    }
                    payload_reads.fetch_add(1U, std::memory_order_relaxed);
                }
            }
        });
    }
    for (std::thread &claimant : claimants) {
        claimant.join();
    }
    return winners.load(std::memory_order_relaxed) == 1U &&
           payload_reads.load(std::memory_order_relaxed) == payload.size();
}

bool TestFourTokenWaitingBuilt() {
    std::array<std::atomic<uint64_t>, 5> states{};
    std::array<ExecutionToken, kTokensPerOwner> tokens{};
    for (uint32_t task = 0U; task < states.size(); ++task) {
        states[task].store(BuildingState(task + 1U), std::memory_order_relaxed);
    }
    states[3].store(0U, std::memory_order_relaxed);
    uint32_t next_ticket = 0U;
    uint32_t busy = 0U;
    uint32_t payload_reads = 0U;
    for (ExecutionToken &token : tokens) {
        token.control.phase = ExecTokenPhase::WaitingBuilt;
        token.control.task_id = next_ticket++ + 1U;
        ++busy;
    }
    if (busy != kTokensPerOwner || next_ticket != 4U) {
        return false;
    }
    for (const ExecutionToken &token : tokens) {
        if (states[token.control.task_id - 1U].load(std::memory_order_acquire) == BuiltState(token.control.task_id)) {
            ++payload_reads;
        }
    }
    if (payload_reads != 0U || next_ticket != 4U || states[3].load(std::memory_order_acquire) != 0U ||
        tokens[3].control.phase != ExecTokenPhase::WaitingBuilt) {
        return false;
    }
    uint64_t empty_expected = BuiltState(4U);
    if (states[3].compare_exchange_strong(
            empty_expected, ClaimedState(4U, 0U), std::memory_order_acq_rel, std::memory_order_acquire
        ) ||
        empty_expected != 0U || tokens[3].control.phase != ExecTokenPhase::WaitingBuilt) {
        return false;
    }
    states[0].store(BuiltState(1U), std::memory_order_release);
    uint64_t expected = BuiltState(1U);
    if (!states[0].compare_exchange_strong(
            expected, ClaimedState(1U, 0U), std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    tokens[0].control.phase = ExecTokenPhase::Idle;
    --busy;
    tokens[0].control.phase = ExecTokenPhase::WaitingBuilt;
    tokens[0].control.task_id = next_ticket++ + 1U;
    ++busy;
    return busy == kTokensPerOwner && next_ticket == 5U;
}

bool TestBuildExecuteOverlap() {
    std::atomic<uint64_t> task1_state{BuildingState(1U)};
    std::atomic<uint64_t> task2_state{BuildingState(2U)};
    std::atomic<int64_t> task1_insert{InsertCompletionInitialValue(1U)};
    std::atomic<int64_t> task2_insert{InsertCompletionInitialValue(2U)};
    std::atomic<int64_t> task1_flag{0};
    PausePoint pause;
    bool builder_ok = false;
    std::thread builder([&] {
        const int64_t task1_before = task1_insert.fetch_add(1, std::memory_order_acq_rel);
        task1_state.store(BuiltState(1U), std::memory_order_release);
        pause.Stop();
        const int64_t task2_before = task2_insert.fetch_add(1, std::memory_order_acq_rel);
        task2_state.store(BuiltState(2U), std::memory_order_release);
        builder_ok = task1_before == 0 && task2_before == 1;
    });
    if (!pause.WaitUntilReached()) {
        pause.Release();
        builder.join();
        return false;
    }
    uint64_t expected = BuiltState(1U);
    const bool claimed = task1_state.compare_exchange_strong(
        expected, ClaimedState(1U, 0U), std::memory_order_acq_rel, std::memory_order_acquire
    );
    const bool later_still_unpublished =
        task2_state.load(std::memory_order_acquire) == BuildingState(2U) &&
        task2_insert.load(std::memory_order_acquire) == InsertCompletionInitialValue(2U);
    task1_flag.store(1, std::memory_order_release);
    expected = ClaimedState(1U, 0U);
    const bool completed = task1_state.compare_exchange_strong(
        expected, DoneState(1U, 0U), std::memory_order_release, std::memory_order_acquire
    );
    const bool completed_before_release = task1_state.load(std::memory_order_acquire) == DoneState(1U, 0U) &&
                                          task1_flag.load(std::memory_order_acquire) == 1 && later_still_unpublished;
    pause.Release();
    builder.join();
    return claimed && completed && completed_before_release && builder_ok &&
           task2_insert.load(std::memory_order_acquire) == 2 &&
           task2_state.load(std::memory_order_acquire) == BuiltState(2U);
}

bool RunCase(uint32_t batches, uint64_t nonce, uint32_t builder_count, FullPaState *state) {
    CpuFullPaModel model(batches, nonce, builder_count);
    if (!model.RunConcurrent()) {
        std::fprintf(
            stderr, "[FAIL] G0 CPU concurrent build/execute builders=%u batches=%u nonce=%llu\n", builder_count,
            batches, static_cast<unsigned long long>(nonce)
        );
        return false;
    }
    if (!model.Validate() || !model.MaterializeAndValidate(state)) {
        std::fprintf(
            stderr, "[FAIL] G0 CPU oracle builders=%u batches=%u nonce=%llu\n", builder_count, batches,
            static_cast<unsigned long long>(nonce)
        );
        return false;
    }
    std::printf(
        "[PASS] G0 CPU builders=%u batches=%u tasks=%u kernels=%u nonce=%llu cursor=%u/%u\n", builder_count, batches,
        model.TaskCountValue(), KernelTaskCount(batches), static_cast<unsigned long long>(nonce),
        2U * batches + kAicOwnerCount, 2U * batches + AivExecutorCount(builder_count)
    );
    return true;
}

bool ParseRounds(int argc, char **argv, uint32_t *rounds) {
    *rounds = 1U;
    if (argc == 1) {
        return true;
    }
    if (argc != 3 || std::strcmp(argv[1], "--rounds") != 0) {
        std::fprintf(stderr, "usage: %s [--rounds N]\n", argv[0]);
        return false;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value == 0U || value > 64U) {
        return false;
    }
    *rounds = static_cast<uint32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    uint32_t rounds = 0U;
    if (!ParseRounds(argc, argv, &rounds) || !TestResetPreservesDispatch() ||
        !TestCompetingBuilderClaimValidation() || !TestHalfPacketAndUniqueClaim() ||
        !TestFourTokenWaitingBuilt() || !TestBuildExecuteOverlap()) {
        std::fprintf(stderr, "[FAIL] G0 controlled interleaving\n");
        return EXIT_FAILURE;
    }
    auto state = std::make_unique<FullPaState>();
    for (uint32_t builder_count = kDefaultBuilderCount; builder_count <= kMaxBuilderCount; ++builder_count) {
        if (!RunCase(
                1U, 0xA500000000000000ULL | (static_cast<uint64_t>(builder_count) << 8U) | 1U, builder_count,
                state.get()
            )) {
            return EXIT_FAILURE;
        }
        for (uint32_t round = 0U; round < rounds; ++round) {
            const uint64_t nonce = 0xA500000000001000ULL + (static_cast<uint64_t>(builder_count) << 32U) + round;
            if (!RunCase(kDefaultBatches, nonce, builder_count, state.get()) ||
                !RunCase(kDefaultBatches, nonce ^ 0x55AAULL, builder_count, state.get())) {
                return EXIT_FAILURE;
            }
        }
    }
    std::printf(
        "[PASS] GM CPU complete: builders=1..8, B1/B256, leaders/builder=%u, unique build claim, "
        "8-shard heap, exact DAG/payload, 4-token tickets, fanin/completion/drain/tail, "
        "same-address reuse rounds=%u\n",
        kBuilderLeaderCount, rounds
    );
    return EXIT_SUCCESS;
}
