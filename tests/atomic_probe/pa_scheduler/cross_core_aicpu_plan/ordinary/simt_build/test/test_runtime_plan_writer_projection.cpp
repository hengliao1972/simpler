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
#define PTO_FDWIC_TENSORMAP_RING_CAP 128
#define PTO_FDWIC_SHARED_INSERT_TURN_GROUPS 1
#define PA_RUNTIME_PLAN_BUILD_BACKEND 1
#define PA_RUNTIME_PLAN_BUILD_WORKERS 4
#define PA_BUILD_SWIMLANE 1

#include "../../scalar_build/common/host_support.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

namespace plan = pa_scheduler::aicpu_plan;
using namespace pa_scheduler;
using namespace pa_scheduler::host;

constexpr uint32_t kTaskCount = 2U;
constexpr uint64_t kOrdinaryAddress = 0x5A7100000ULL;
constexpr uint32_t kOutputShapes[3] = {8U, 16U, 24U};

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(1);
}

void Expect(bool condition, const char *message)
{
    if (!condition) Fail(message);
}

struct CpuPlanOps {
    static int64_t LoadControl(const volatile int64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static void StorePayloadWord(
        volatile uint64_t *address, uint64_t value
    )
    {
        *address = value;
    }

    template <typename Pointer>
    static void FlushRegion(Pointer address, uint64_t bytes)
    {
        (void)address;
        (void)bytes;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void PublishControl(
        volatile int64_t *address, int64_t value
    )
    {
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }
};

struct CanonicalSource {
    std::array<plan::TensorTag, plan::kMaxTaskTensors> tags{};
    std::array<uint8_t, plan::kMaxTaskTensors> references{};
    std::array<
        std::array<uint64_t, plan::kTensorCanonicalWords>,
        plan::kMaxTaskTensors
    > words{};

    plan::TensorTag TensorTagAt(uint32_t tensor) const
    {
        return tags[tensor];
    }

    bool TensorIsReference(uint32_t tensor) const
    {
        return references[tensor] != 0U;
    }

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const
    {
        return words[tensor][word];
    }

    uint64_t Scalar(uint32_t) const
    {
        return 0U;
    }

    uint64_t ExplicitDependency(uint32_t) const
    {
        return 0U;
    }
};

uint64_t PackU32(uint32_t low, uint32_t high)
{
    return static_cast<uint64_t>(low) |
           (static_cast<uint64_t>(high) << 32U);
}

void SetOutput(
    CanonicalSource &source, uint32_t tensor,
    uint32_t shape, int32_t version
)
{
    source.tags[tensor] = plan::TensorTag::Output;
    source.words[tensor][4] = PackU32(
        static_cast<uint32_t>(version), 1U
    );
    source.words[tensor][5] =
        static_cast<uint64_t>(DataType::Float32) |
        (uint64_t{1} << 16U) |
        (static_cast<uint64_t>(shape) << 32U);
}

void SetOutputReference(
    CanonicalSource &source, uint32_t tensor, int16_t output_slot
)
{
    source.tags[tensor] = plan::TensorTag::Inout;
    source.references[tensor] = 1U;
    const plan::RuntimeOutputReferenceWire reference{
        /*producer_task_id=*/0,
        output_slot,
        /*flags=*/0U,
        /*view_ndims=*/0U,
        /*view_shape0=*/0U,
        /*view_offset0=*/0U,
    };
    source.words[tensor][0] =
        plan::RuntimeOutputReferenceWireWord(reference, 0U);
    source.words[tensor][1] =
        plan::RuntimeOutputReferenceWireWord(reference, 1U);
}

void SetInlineOrdinaryInout(CanonicalSource &source, uint32_t tensor)
{
    source.tags[tensor] = plan::TensorTag::Inout;
    source.words[tensor][0] = kOrdinaryAddress;
    source.words[tensor][1] = 4096U;
    source.words[tensor][2] = kHostInvalidTaskId;
    source.words[tensor][3] = 2U;
    source.words[tensor][4] = PackU32(7U, 1U);
    source.words[tensor][5] =
        static_cast<uint64_t>(DataType::Float32) |
        (uint64_t{1} << 16U) |
        (uint64_t{64} << 32U);
    source.words[tensor][8] = 64U;
    source.words[tensor][9] = 1U;
}

TensorDesc ExpectedOutputDescriptor(uint32_t slot, uint64_t address)
{
    TensorDesc tensor{};
    tensor.buffer_addr = address;
    tensor.buffer_size =
        static_cast<uint64_t>(kOutputShapes[slot]) * sizeof(float);
    tensor.owner_task_id = 0U;
    tensor.version = static_cast<int32_t>(10U + slot);
    tensor.ndims = 1U;
    tensor.dtype = DataType::Float32;
    tensor.is_contiguous = true;
    tensor.shapes[0] = kOutputShapes[slot];
    tensor.extent_elem_cache = kOutputShapes[slot];
    tensor.strides[0] = 1U;
    return tensor;
}

bool BuildCanonicalSnapshot(
    plan::RuntimePlanControl &control,
    std::array<plan::RuntimeTaskPlanCell, kTaskCount> &cells
)
{
    control.closed_task_count.value = plan::kPlanOpenTaskCount;
    control.build_release.value = plan::kBuildReleasePending;
    const plan::RuntimePlanView view{
        &control, cells.data(), kTaskCount,
    };

    CanonicalSource producer{};
    for (uint32_t slot = 0U; slot < 3U; ++slot) {
        SetOutput(
            producer, slot, kOutputShapes[slot],
            static_cast<int32_t>(10U + slot)
        );
    }
    const plan::RuntimeTaskPlanSpec producer_spec{
        /*task_id=*/0U,
        /*function_id=*/plan::kInvalidFunctionId,
        /*tensor_count=*/3U,
        /*scalar_count=*/0U,
        /*explicit_dep_count=*/0U,
        /*output_count=*/3U,
        plan::EngineClass::MetadataOnly,
        /*adapter_flags=*/0U,
        /*core_num=*/1,
        /*require_sync_start=*/0U,
        /*reserved=*/0U,
        /*adapter_data=*/0U,
        /*tensor_reference_mask=*/0U,
    };
    if (plan::PublishRuntimeTaskPlan<CpuPlanOps>(
            view, producer_spec, producer
        ) != plan::PlanPublishResult::Published ||
        !plan::AdvancePlannedFrontier<CpuPlanOps>(view, 0U)) {
        return false;
    }

    CanonicalSource writer{};
    SetOutputReference(writer, 0U, 2);
    SetOutputReference(writer, 1U, 1);
    SetOutputReference(writer, 2U, 0);
    SetInlineOrdinaryInout(writer, 3U);
    const plan::RuntimeTaskPlanSpec writer_spec{
        /*task_id=*/1U,
        /*function_id=*/plan::kInvalidFunctionId,
        /*tensor_count=*/4U,
        /*scalar_count=*/0U,
        /*explicit_dep_count=*/0U,
        /*output_count=*/0U,
        plan::EngineClass::MetadataOnly,
        /*adapter_flags=*/0U,
        /*core_num=*/1,
        /*require_sync_start=*/0U,
        /*reserved=*/0U,
        /*adapter_data=*/0U,
        /*tensor_reference_mask=*/0x7U,
    };
    return plan::PublishRuntimeTaskPlan<CpuPlanOps>(
               view, writer_spec, writer
           ) == plan::PlanPublishResult::Published &&
           plan::AdvancePlannedFrontier<CpuPlanOps>(view, 1U) &&
           plan::CloseRuntimePlan<CpuPlanOps>(view, kTaskCount);
}

std::unique_ptr<SharedTensorMapSidecar> MakePositiveMap()
{
    auto map = std::make_unique<SharedTensorMapSidecar>();
    std::memset(map.get(), 0, sizeof(*map));
    map->committed_tasks.value = 0;
    for (uint32_t lane = 1U;
         lane < kSharedInsertTurnCapacity; ++lane) {
        map->insert_turn_extra[lane - 1U].value = -1;
    }
    map->reclaim_upto.value = -1;
    for (uint32_t slot = 0U; slot < kMapCapacity; ++slot) {
        map->slots[slot].seq.value = -1;
    }
    for (uint32_t task = 0U; task < kMaxTasks; ++task) {
        for (uint32_t slot = 0U;
             slot < kSharedOutputMaxPerTask; ++slot) {
            map->shared_outputs[task].published[slot].value = -1;
            map->shared_outputs[task].last_writer[slot].value = -1;
        }
    }
    for (uint32_t worker = 0U; worker < kWorkers; ++worker) {
        map->reader_done[worker].value = -1;
    }

    uint64_t output_offset = 0U;
    for (uint32_t slot = 0U; slot < 3U; ++slot) {
        SharedOutputCell &outputs = map->shared_outputs[0];
        outputs.published[slot].value = 0;
        outputs.last_writer[slot].value = 1;
        outputs.tensors[slot] = ExpectedOutputDescriptor(
            slot, kSyntheticHeapBase + output_offset
        );
        output_offset += kOutputAlignment;
    }

    SharedWriterHistoryCell &history = map->writer_history[1];
    history.magic = kSharedWriterHistoryMagic;
    history.writer_task = 1;
    history.count = 3U;
    history.entries[0] = SharedWriterHistoryRecord{3U, 0};
    history.entries[1] = SharedWriterHistoryRecord{2U, 0};
    history.entries[2] = SharedWriterHistoryRecord{1U, 0};

    const uint32_t bucket = SharedTensorMapHashHost(kOrdinaryAddress);
    const uint32_t slot_index = bucket * kMapBucketCapacity;
    map->buckets[bucket].tail.value = 1;
    map->slots[slot_index].seq.value = 0;
    map->slots[slot_index].payload.value = SharedRegionValue{
        kOrdinaryAddress,
        /*lo=*/2U * sizeof(float),
        /*hi=*/66U * sizeof(float),
        /*producer=*/1,
        /*reserved=*/0U,
    };
    return map;
}

void CheckProjection(const RuntimePlanWriterProjection &projection)
{
    Expect(projection.protocol_ok, "Plan projection rejected canonical snapshot");
    Expect(
        projection.published_outputs == 3U &&
            projection.allocation_bytes == 3U * kOutputAlignment,
        "projected output/allocation totals are wrong"
    );
    Expect(
        projection.output_counts.size() == kTaskCount &&
            projection.output_counts[0] == 3U &&
            projection.output_counts[1] == 0U,
        "projected per-task output counts are wrong"
    );
    for (uint32_t slot = 0U; slot < 3U; ++slot) {
        const size_t index = slot;
        Expect(
            projection.final_writers[index] == 1,
            "one of the three output slots did not advance to task 1"
        );
        const TensorDesc expected = ExpectedOutputDescriptor(slot, 0U);
        Expect(
            TensorDescFieldsMatch(
                projection.output_descriptors[index], expected
            ),
            "projected Output descriptor is not canonical"
        );
    }
    const SharedWriterHistoryCell &history =
        projection.writer_histories[1];
    Expect(
        history.magic == kSharedWriterHistoryMagic &&
            history.writer_task == 1 && history.count == 3U,
        "writer history header is wrong"
    );
    const uint32_t expected_keys[3] = {3U, 2U, 1U};
    for (uint32_t entry = 0U; entry < 3U; ++entry) {
        Expect(
            history.entries[entry].symbol_key == expected_keys[entry] &&
                history.entries[entry].symbol_key != 0U &&
                history.entries[entry].previous_writer == 0,
            "writer history is not one-based slot2/slot1/slot0"
        );
    }
    uint64_t ordinary_entries = 0U;
    for (const auto &bucket : projection.ordinary_by_bucket) {
        ordinary_entries += bucket.size();
    }
    const uint32_t ordinary_bucket =
        SharedTensorMapHashHost(kOrdinaryAddress);
    Expect(
        ordinary_entries == 1U &&
            projection.ordinary_by_bucket[ordinary_bucket].size() == 1U,
        "projection did not produce exactly one ordinary-ring writer"
    );
    const SharedRegionValue &ordinary =
        projection.ordinary_by_bucket[ordinary_bucket][0];
    Expect(
        ordinary.buffer_addr == kOrdinaryAddress &&
            ordinary.lo == 2U * sizeof(float) &&
            ordinary.hi == 66U * sizeof(float) &&
            ordinary.producer == 1 && ordinary.reserved == 0U,
        "ordinary-ring projection payload is wrong"
    );
}

void RunGate()
{
    plan::RuntimePlanControl control{};
    std::array<plan::RuntimeTaskPlanCell, kTaskCount> cells{};
    Expect(
        BuildCanonicalSnapshot(control, cells),
        "failed to publish canonical two-task Plan snapshot"
    );
    const RuntimePlanWriterProjection projection =
        BuildRuntimePlanWriterProjection(
            RuntimePlanHostSnapshot{cells.data(), kTaskCount},
            kTaskCount
        );
    CheckProjection(projection);

    std::unique_ptr<SharedTensorMapSidecar> map = MakePositiveMap();
    const SharedTensorMapValidation map_validation =
        ValidateRuntimePlanSharedTensorMap(*map, projection);
    const SharedOutputValidation output_validation =
        ValidateRuntimePlanSharedOutputs(*map, projection, kHeapBytes);
    Expect(
        map_validation.protocol_ok &&
            map_validation.total_appends == 1U &&
            map_validation.physical_entries == 1U &&
            map_validation.logical_entries == 1U,
        "positive ordinary TensorMap state was rejected"
    );
    Expect(
        output_validation.protocol_ok &&
            output_validation.published_outputs == 3U &&
            output_validation.allocated_bytes == 3U * kOutputAlignment,
        "positive shared-output state was rejected"
    );

    map->shared_outputs[0].last_writer[1].value = 0;
    const SharedOutputValidation stale_slot =
        ValidateRuntimePlanSharedOutputs(*map, projection, kHeapBytes);
    Expect(
        !stale_slot.protocol_ok && stale_slot.first_bad_task == 0U &&
            stale_slot.first_bad_slot == 1U &&
            std::strcmp(
                stale_slot.first_bad_reason, "active last_writer"
            ) == 0,
        "slot-1 last_writer regression was not rejected"
    );
    map->shared_outputs[0].last_writer[1].value = 1;

    map->writer_history[1].entries[0].symbol_key = 0U;
    Expect(
        !ValidateRuntimePlanSharedTensorMap(*map, projection).protocol_ok,
        "zero writer-history key was not rejected"
    );
    map->writer_history[1].entries[0].symbol_key = 3U;

    const uint32_t bucket = SharedTensorMapHashHost(kOrdinaryAddress);
    const uint32_t slot_index = bucket * kMapBucketCapacity;
    map->buckets[bucket].tail.value = 0;
    map->slots[slot_index].seq.value = -1;
    map->slots[slot_index].payload = SharedRegionPayload{};
    Expect(
        !ValidateRuntimePlanSharedTensorMap(*map, projection).protocol_ok,
        "missing ordinary-ring slot was not rejected"
    );
}

}  // namespace

int main()
{
    RunGate();
    std::puts(
        "[PASS] runtime Plan writer projection positive/negative gate"
    );
    return 0;
}
