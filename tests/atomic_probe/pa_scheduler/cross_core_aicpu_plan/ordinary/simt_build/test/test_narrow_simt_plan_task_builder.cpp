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

#include "../../scalar_build/common/pa_scheduler_core.h"
#include "../common/simt_plan_task_builder.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <sys/mman.h>
#include <thread>

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace simt = pa_scheduler::aicpu_plan_simt;
namespace exec = pa_scheduler::cross_core;
using namespace pa_scheduler;

constexpr uint32_t kTaskCount = 8U;
constexpr uint32_t kDelayedOrdinaryReader = 4U;
constexpr uint32_t kFutureOrdinaryWriter = 7U;
constexpr uint64_t kOrdinaryAddress = UINT64_C(0x6A5100000);
constexpr uint8_t kAdapterMagic = 0xD3U;
constexpr uint64_t kHeapBytes = 1U << 20U;
constexpr auto kDeadline = std::chrono::seconds(15);

static_assert(sizeof(simt::SimtCanonicalTensorDesc) == sizeof(TensorDesc));
static_assert(alignof(simt::SimtCanonicalTensorDesc) == alignof(TensorDesc));
static_assert(offsetof(simt::SimtCanonicalTensorDesc, buffer_addr) ==
              offsetof(TensorDesc, buffer_addr));
static_assert(offsetof(simt::SimtCanonicalTensorDesc, shapes) ==
              offsetof(TensorDesc, shapes));
static_assert(offsetof(simt::SimtCanonicalTensorDesc, strides) ==
              offsetof(TensorDesc, strides));
static_assert(sizeof(simt::SimtWriterRegion) == sizeof(SharedRegionValue));
static_assert(alignof(simt::SimtWriterRegion) == alignof(SharedRegionValue));
static_assert(offsetof(simt::SimtWriterRegion, buffer_addr) ==
              offsetof(SharedRegionValue, buffer_addr));
static_assert(offsetof(simt::SimtWriterRegion, lo) ==
              offsetof(SharedRegionValue, lo));
static_assert(offsetof(simt::SimtWriterRegion, hi) ==
              offsetof(SharedRegionValue, hi));
static_assert(offsetof(simt::SimtWriterRegion, producer) ==
              offsetof(SharedRegionValue, producer));
static_assert(offsetof(simt::SimtWriterRegion, reserved) ==
              offsetof(SharedRegionValue, reserved));
static_assert(sizeof(simt::SimtWriterHistoryRecord) ==
              sizeof(SharedWriterHistoryRecord));
static_assert(alignof(simt::SimtWriterHistoryRecord) ==
              alignof(SharedWriterHistoryRecord));
static_assert(offsetof(simt::SimtWriterHistoryRecord, symbol_key) ==
              offsetof(SharedWriterHistoryRecord, symbol_key));
static_assert(offsetof(simt::SimtWriterHistoryRecord, previous_writer) ==
              offsetof(SharedWriterHistoryRecord, previous_writer));
static_assert(sizeof(simt::SimtWriterHistoryCell) ==
              sizeof(SharedWriterHistoryCell));
static_assert(alignof(simt::SimtWriterHistoryCell) ==
              alignof(SharedWriterHistoryCell));
static_assert(offsetof(simt::SimtWriterHistoryCell, magic) ==
              offsetof(SharedWriterHistoryCell, magic));
static_assert(offsetof(simt::SimtWriterHistoryCell, writer_task) ==
              offsetof(SharedWriterHistoryCell, writer_task));
static_assert(offsetof(simt::SimtWriterHistoryCell, count) ==
              offsetof(SharedWriterHistoryCell, count));
static_assert(offsetof(simt::SimtWriterHistoryCell, reserved) ==
              offsetof(SharedWriterHistoryCell, reserved));
static_assert(offsetof(simt::SimtWriterHistoryCell, entries) ==
              offsetof(SharedWriterHistoryCell, entries));

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(1);
}

void Expect(bool condition, const char *message)
{
    if (!condition) Fail(message);
}

void TestSharedSymbolHistoryKeyAbi()
{
    uint32_t key = 0U;
    uint32_t producer = UINT32_MAX;
    uint32_t slot = UINT32_MAX;
    Expect(simt::SimtEncodeSharedSymbolHistoryKey(0U, 0U, key) &&
               key == 1U &&
               simt::SimtDecodeSharedSymbolHistoryKey(
                   key, producer, slot
               ) && producer == 0U && slot == 0U,
           "producer 0 slot 0 did not use one-based history key 1");
    Expect(simt::SimtEncodeSharedSymbolHistoryKey(7U, 3U, key) &&
               key == 60U &&
               simt::SimtDecodeSharedSymbolHistoryKey(
                   key, producer, slot
               ) && producer == 7U && slot == 3U,
           "nonzero shared history key did not round-trip");
    Expect(!simt::SimtDecodeSharedSymbolHistoryKey(
               0U, producer, slot
           ) && !simt::SimtEncodeSharedSymbolHistoryKey(
               0U, plan::kMaxRuntimeOutputsPerTask, key
           ),
           "reserved key 0 or out-of-range output slot was accepted");
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

struct CpuOps {
    static inline SchedulerState *state = nullptr;
    static inline uint32_t tracked_tasks = 0U;
    static inline std::array<std::atomic<int32_t>, kTaskCount>
        completion_order{};
    static inline std::atomic<uint32_t> completion_count{0U};
    static inline std::atomic<uint32_t> strict_order_errors{0U};

    static void TrackCompletions(SchedulerState *scheduler, uint32_t tasks)
    {
        state = scheduler;
        tracked_tasks = tasks;
        completion_count.store(0U, std::memory_order_relaxed);
        strict_order_errors.store(0U, std::memory_order_relaxed);
        for (auto &entry : completion_order) {
            entry.store(-1, std::memory_order_relaxed);
        }
    }

    static int32_t Load(volatile int32_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static int64_t Load(volatile int64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static uint64_t Load(volatile uint64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static int64_t LoadControl(const volatile int64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
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
        int64_t observed = expected;
        const bool changed = __atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        if (changed && state != nullptr) {
            for (uint32_t task = 0U; task < tracked_tasks; ++task) {
                if (address != &state->claim_tournament[task]
                                    .root.insert_completion.value) {
                    continue;
                }
                if (expected != SharedInsertCompletionInitialValue(task) ||
                    desired != static_cast<int64_t>(task)) {
                    strict_order_errors.fetch_add(
                        1U, std::memory_order_relaxed
                    );
                }
                const uint32_t ordinal = completion_count.fetch_add(
                    1U, std::memory_order_acq_rel
                );
                if (ordinal >= tracked_tasks) {
                    strict_order_errors.fetch_add(
                        1U, std::memory_order_relaxed
                    );
                } else {
                    completion_order[ordinal].store(
                        static_cast<int32_t>(task),
                        std::memory_order_release
                    );
                }
                break;
            }
        }
        return observed;
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value)
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchAddControl(
        volatile int64_t *address, int64_t value
    )
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static void PublishControl(volatile int64_t *address, int64_t value)
    {
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }

    static void StorePayloadWord(
        volatile uint64_t *address, uint64_t value
    )
    {
        *address = value;
    }

    template <typename Pointer>
    static void FlushRegion(Pointer, uint64_t)
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    template <typename Pointer>
    static void InvalidateRegion(Pointer, uint64_t)
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    template <typename Pointer>
    static void PreloadBuildDestination(Pointer, uint64_t)
    {
    }

    static void BeforeBuiltPublish(uint32_t)
    {
    }

    static uint64_t Now()
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    static void SpinHint()
    {
        std::this_thread::yield();
    }
};

struct TensorWire {
    plan::TensorTag tag = plan::TensorTag::NoDependency;
    bool reference = false;
    simt::SimtCanonicalTensorDesc tensor{};
    simt::SimtCanonicalCreateInfo create{};
    plan::RuntimeOutputReferenceWire output_ref{};
};

struct TaskDefinition {
    plan::EngineClass engine = plan::EngineClass::MetadataOnly;
    uint32_t function_id = plan::kInvalidFunctionId;
    std::array<TensorWire, 3U> tensors{};
    uint16_t tensor_count = 0U;
    std::array<uint64_t, 2U> scalars{};
    uint16_t scalar_count = 0U;
    std::array<uint64_t, 2U> explicit_dependencies{};
    uint16_t explicit_dep_count = 0U;
    uint16_t output_count = 0U;
};

uint64_t PackPair(uint32_t low, uint32_t high)
{
    return static_cast<uint64_t>(low) |
           (static_cast<uint64_t>(high) << 32U);
}

uint64_t TensorWord(
    const simt::SimtCanonicalTensorDesc &tensor, uint32_t word
)
{
    switch (word) {
        case 0U: return tensor.buffer_addr;
        case 1U: return tensor.buffer_size;
        case 2U: return tensor.owner_task_id;
        case 3U: return tensor.start_offset;
        case 4U:
            return PackPair(
                static_cast<uint32_t>(tensor.version), tensor.ndims
            );
        case 5U:
            return static_cast<uint64_t>(tensor.dtype) |
                   (static_cast<uint64_t>(tensor.manual_dep) << 8U) |
                   (static_cast<uint64_t>(tensor.is_contiguous) << 16U) |
                   (static_cast<uint64_t>(tensor.child_memory) << 24U) |
                   (static_cast<uint64_t>(tensor.shapes[0]) << 32U);
        case 6U: return PackPair(tensor.shapes[1], tensor.shapes[2]);
        case 7U: return PackPair(tensor.shapes[3], tensor.shapes[4]);
        case 8U: return tensor.extent_elem_cache;
        case 9U: return PackPair(tensor.strides[0], tensor.strides[1]);
        case 10U: return PackPair(tensor.strides[2], tensor.strides[3]);
        case 11U: return tensor.strides[4];
        default: return 0U;
    }
}

uint64_t CreateInfoWord(
    const simt::SimtCanonicalCreateInfo &info, uint32_t word
)
{
    switch (word) {
        case 0U: return info.initial_value;
        case 1U: return info.has_initial_value;
        case 2U: return info.reserved0;
        case 3U: return info.start_offset;
        case 4U:
            return PackPair(
                static_cast<uint32_t>(info.version), info.ndims
            );
        case 5U:
            return static_cast<uint64_t>(info.dtype) |
                   (static_cast<uint64_t>(info.manual_dep) << 8U) |
                   (static_cast<uint64_t>(info.is_contiguous) << 16U) |
                   (static_cast<uint64_t>(info.child_memory) << 24U) |
                   (static_cast<uint64_t>(info.shapes[0]) << 32U);
        case 6U: return PackPair(info.shapes[1], info.shapes[2]);
        case 7U: return PackPair(info.shapes[3], info.shapes[4]);
        default: return 0U;
    }
}

struct TaskPlanSource {
    const TaskDefinition &definition;

    plan::TensorTag TensorTagAt(uint32_t tensor) const
    {
        return definition.tensors[tensor].tag;
    }

    bool TensorIsReference(uint32_t tensor) const
    {
        return definition.tensors[tensor].reference;
    }

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const
    {
        const TensorWire &wire = definition.tensors[tensor];
        if (wire.reference) {
            return plan::RuntimeOutputReferenceWireWord(
                wire.output_ref, word
            );
        }
        return wire.tag == plan::TensorTag::Output
            ? CreateInfoWord(wire.create, word)
            : ::TensorWord(wire.tensor, word);
    }

    uint64_t Scalar(uint32_t scalar) const
    {
        return definition.scalars[scalar];
    }

    uint64_t ExplicitDependency(uint32_t dependency) const
    {
        return definition.explicit_dependencies[dependency];
    }
};

simt::SimtCanonicalTensorDesc OrdinaryTensor()
{
    simt::SimtCanonicalTensorDesc tensor{};
    tensor.buffer_addr = kOrdinaryAddress;
    tensor.buffer_size = 4096U;
    tensor.owner_task_id = simt::kSimtInvalidTaskOwner;
    tensor.ndims = 1U;
    tensor.dtype = static_cast<uint8_t>(DataType::Float32);
    tensor.is_contiguous = 1U;
    tensor.shapes[0] = 1024U;
    tensor.extent_elem_cache = 1024U;
    tensor.strides[0] = 1U;
    return tensor;
}

simt::SimtCanonicalCreateInfo FreshOutputInfo()
{
    simt::SimtCanonicalCreateInfo info{};
    info.ndims = 1U;
    info.dtype = static_cast<uint8_t>(DataType::Float32);
    info.is_contiguous = 1U;
    info.shapes[0] = 256U;
    return info;
}

TensorWire PlainTensor(
    plan::TensorTag tag,
    const simt::SimtCanonicalTensorDesc &tensor
)
{
    TensorWire wire{};
    wire.tag = tag;
    wire.tensor = tensor;
    return wire;
}

TensorWire OutputTensor()
{
    TensorWire wire{};
    wire.tag = plan::TensorTag::Output;
    wire.create = FreshOutputInfo();
    return wire;
}

TensorWire OutputReference(plan::TensorTag tag)
{
    TensorWire wire{};
    wire.tag = tag;
    wire.reference = true;
    wire.output_ref = plan::RuntimeOutputReferenceWire{
        0, 0, 0U, 0U, 0U, 0U,
    };
    return wire;
}

std::array<TaskDefinition, kTaskCount> MakeWorkload()
{
    std::array<TaskDefinition, kTaskCount> tasks{};
    tasks[0].tensors[0] = OutputTensor();
    tasks[0].tensor_count = 1U;
    tasks[0].output_count = 1U;

    tasks[1].engine = plan::EngineClass::Aiv;
    tasks[1].function_id = 101U;
    tasks[1].tensors[0] = OutputReference(plan::TensorTag::Inout);
    tasks[1].tensor_count = 1U;
    tasks[1].scalars[0] = UINT64_C(0x11110001);
    tasks[1].scalar_count = 1U;

    tasks[2].engine = plan::EngineClass::Aic;
    tasks[2].function_id = 202U;
    tasks[2].tensors[0] = OutputReference(plan::TensorTag::Input);
    tasks[2].tensor_count = 1U;
    tasks[2].scalars[0] = UINT64_C(0x22220002);
    tasks[2].scalar_count = 1U;

    const simt::SimtCanonicalTensorDesc ordinary = OrdinaryTensor();
    tasks[3].engine = plan::EngineClass::Aiv;
    tasks[3].function_id = 303U;
    tasks[3].tensors[0] = PlainTensor(plan::TensorTag::Inout, ordinary);
    tasks[3].tensor_count = 1U;

    tasks[4].engine = plan::EngineClass::Aic;
    tasks[4].function_id = 404U;
    tasks[4].tensors[0] = PlainTensor(plan::TensorTag::Input, ordinary);
    tasks[4].tensor_count = 1U;
    tasks[4].explicit_dependencies[0] = 0U;
    tasks[4].explicit_dep_count = 1U;

    // task 5 是空 metadata task：它仍必须推进 strict insert completion，
    // 然后通过真实 TaskCell 发布 vend/flag，而不是生成 ExecCell。

    tasks[6].engine = plan::EngineClass::Aiv;
    tasks[6].function_id = 606U;
    tasks[6].tensors[0] = OutputReference(plan::TensorTag::Inout);
    tasks[6].tensor_count = 1U;

    tasks[7].engine = plan::EngineClass::Aic;
    tasks[7].function_id = 707U;
    tasks[7].tensors[0] = PlainTensor(plan::TensorTag::Inout, ordinary);
    tasks[7].tensor_count = 1U;
    return tasks;
}

std::array<TaskDefinition, kTaskCount> MakeTwoOutputWorkload()
{
    std::array<TaskDefinition, kTaskCount> tasks{};
    tasks[0].tensors[0] = OutputTensor();
    tasks[0].tensors[1] = OutputTensor();
    tasks[0].tensor_count = 2U;
    tasks[0].output_count = 2U;
    return tasks;
}

uint32_t ReferenceMask(const TaskDefinition &definition)
{
    uint32_t mask = 0U;
    for (uint32_t tensor = 0U;
         tensor < definition.tensor_count; ++tensor) {
        if (definition.tensors[tensor].reference) {
            mask |= uint32_t{1} << tensor;
        }
    }
    return mask;
}

plan::RuntimeTaskPlanSpec MakePlanSpec(
    uint32_t task_id, const TaskDefinition &definition
)
{
    return plan::RuntimeTaskPlanSpec{
        task_id,
        definition.function_id,
        definition.tensor_count,
        definition.scalar_count,
        definition.explicit_dep_count,
        definition.output_count,
        definition.engine,
        kAdapterMagic,
        /*core_num=*/1,
        /*require_sync_start=*/0U,
        /*reserved=*/0U,
        static_cast<uint16_t>(task_id),
        ReferenceMask(definition),
    };
}

struct PlanCellFree {
    void operator()(plan::RuntimeTaskPlanCell *cells) const
    {
        std::free(cells);
    }
};

struct PlanFixture {
    alignas(plan::kAtomicIsolationBytes) plan::RuntimePlanControl control{};
    std::unique_ptr<plan::RuntimeTaskPlanCell, PlanCellFree> cells;
    plan::RuntimePlanView view{};
    uint32_t capacity;

    explicit PlanFixture(uint32_t requested_capacity)
        : capacity(requested_capacity)
    {
        void *raw = nullptr;
        const size_t bytes = static_cast<size_t>(capacity) *
                             sizeof(plan::RuntimeTaskPlanCell);
        if (posix_memalign(
                &raw, plan::kAtomicIsolationBytes, bytes
            ) != 0) {
            return;
        }
        std::memset(raw, 0, bytes);
        cells.reset(static_cast<plan::RuntimeTaskPlanCell *>(raw));
        control.closed_task_count.value = plan::kPlanOpenTaskCount;
        control.build_release.value = plan::kBuildReleasePending;
        view = plan::RuntimePlanView{
            &control, cells.get(), requested_capacity,
        };
    }

    bool Valid() const
    {
        return cells != nullptr;
    }
};

bool PublishPlan(
    PlanFixture &fixture,
    const std::array<TaskDefinition, kTaskCount> &tasks,
    uint32_t task_count
)
{
    if (!fixture.Valid() || task_count > tasks.size()) return false;
    for (uint32_t task = 0U; task < task_count; ++task) {
        const TaskPlanSource source{tasks[task]};
        if (plan::PublishRuntimeTaskPlan<CpuOps>(
                fixture.view, MakePlanSpec(task, tasks[task]), source
            ) != plan::PlanPublishResult::Published ||
            !plan::AdvancePlannedFrontier<CpuOps>(fixture.view, task)) {
            return false;
        }
    }
    return plan::CloseRuntimePlan<CpuOps>(fixture.view, task_count);
}

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
    if (memory == MAP_FAILED) return nullptr;
    return ::new (memory) SchedulerState;
}

void UnmapSchedulerState(SchedulerState *state)
{
    if (state != nullptr) {
        (void)munmap(state, sizeof(SchedulerState));
    }
}

struct HeapFree {
    void operator()(uint8_t *memory) const
    {
        std::free(memory);
    }
};

struct RuntimeFixture {
    std::unique_ptr<SchedulerState, void (*)(SchedulerState *)> state;
    std::unique_ptr<uint8_t, HeapFree> heap;

    RuntimeFixture()
        : state(MapSchedulerState(), UnmapSchedulerState)
    {
        void *raw = nullptr;
        if (state == nullptr ||
            posix_memalign(&raw, kOutputAlignment, kHeapBytes) != 0) {
            state.reset();
            return;
        }
        std::memset(raw, 0, kHeapBytes);
        heap.reset(static_cast<uint8_t *>(raw));
        Reset(kTaskCount);
    }

    bool Valid() const
    {
        return state != nullptr && heap != nullptr;
    }

    void Reset(uint32_t task_count)
    {
        SchedulerState &scheduler = *state;
        scheduler.fatal.value = 0;
        scheduler.heap_window = kHeapWindow;
        scheduler.heap_base = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(heap.get())
        );
        scheduler.heap_size = kHeapBytes;
        scheduler.shared_map.reclaim_upto.value = -1;
        scheduler.shared_map.shared_heap_vend.value = 0;
        scheduler.exec_fatal.state = 0;
        for (uint32_t shard = 0U; shard < kSharedHeapShards; ++shard) {
            scheduler.shared_map.shared_heap_cursor[shard].value = 0;
        }
        for (uint32_t bucket = 0U; bucket < kMapBuckets; ++bucket) {
            scheduler.shared_map.buckets[bucket].head.value = 0;
            scheduler.shared_map.buckets[bucket].tail.value = 0;
        }
        for (uint32_t slot = 0U; slot < kMapCapacity; ++slot) {
            scheduler.shared_map.slots[slot].seq.value =
                kSharedMapEmptySeq;
        }
        for (uint32_t task = 0U; task < task_count; ++task) {
            scheduler.tasks[task].flag = 0;
            scheduler.tasks[task].vend = 0;
            scheduler.tasks[task].deps_prepared = 0;
            scheduler.claim_tournament[task]
                .root.insert_completion.value =
                SharedInsertCompletionInitialValue(task);
            scheduler.exec_cells[task].control.state = 0;
            std::memset(
                &scheduler.exec_cells[task].payload, 0,
                sizeof(scheduler.exec_cells[task].payload)
            );
            SharedOutputCell &outputs =
                scheduler.shared_map.shared_outputs[task];
            for (uint32_t output = 0U;
                 output < kSharedOutputMaxPerTask; ++output) {
                outputs.published[output].value = -1;
                outputs.last_writer[output].value = -1;
                std::memset(
                    &outputs.tensors[output], 0,
                    sizeof(outputs.tensors[output])
                );
            }
            std::memset(
                &scheduler.shared_map.writer_history[task], 0,
                sizeof(SharedWriterHistoryCell)
            );
        }
        CpuOps::TrackCompletions(&scheduler, task_count);
    }
};

struct NarrowRuntime {
    SchedulerState &state;
    plan::RuntimePlanControl &plan_control;
    const TaskDefinition *definitions;
    uint32_t task_count;
    std::atomic<uint32_t> *fatal_task;
    std::atomic<uint32_t> *fatal_status;
    std::atomic<uint32_t> *future_lookup_observed;
    uint32_t fail_descriptor_task;
    uint32_t fail_descriptor_output;

    volatile int64_t *OutputLastWriter(uint32_t task, uint32_t output)
    {
        if (task >= task_count || output >= kSharedOutputMaxPerTask) {
            return nullptr;
        }
        return &state.shared_map.shared_outputs[task]
                    .last_writer[output].value;
    }

    volatile int64_t *OutputPublished(uint32_t task, uint32_t output)
    {
        if (task >= task_count || output >= kSharedOutputMaxPerTask) {
            return nullptr;
        }
        return &state.shared_map.shared_outputs[task]
                    .published[output].value;
    }

    volatile uint64_t *OutputDescriptorWords(
        uint32_t task, uint32_t output
    )
    {
        if (task >= task_count || output >= kSharedOutputMaxPerTask) {
            return nullptr;
        }
        if (task == fail_descriptor_task &&
            output == fail_descriptor_output) {
            return nullptr;
        }
        return reinterpret_cast<volatile uint64_t *>(
            &state.shared_map.shared_outputs[task].tensors[output]
        );
    }

    bool ReserveOutputHeap(
        uint32_t task, uint64_t total,
        uint64_t &first_address, uint64_t &completion_vend
    )
    {
        SharedHeapReservation reservation{};
        if (!ReserveSharedOutputHeap<CpuOps, false>(
                state.shared_map, task, total,
                state.heap_size, reservation
            )) {
            return false;
        }
        first_address = total == 0U
            ? 0U
            : state.heap_base + reservation.task_base;
        completion_vend = reservation.aggregate_vend;
        return true;
    }

    uint32_t OrdinaryBucket(uint64_t address) const
    {
        return TensorMapHash(address);
    }

    bool CheckOrdinaryAppend(
        const simt::SimtWriterRegion *entries,
        const uint16_t *buckets,
        const uint8_t *ordinals,
        uint32_t count, uint32_t task
    )
    {
        std::array<SharedRegionValue, plan::kMaxTaskTensors> converted{};
        if (!ConvertEntries(entries, count, converted)) return false;
        return SharedCheckPreparedTaskAppend<CpuOps, false>(
                   state.shared_map, converted.data(), buckets,
                   ordinals, count, -1, static_cast<int32_t>(task)
               ) == SharedAppendCheck::Ready;
    }

    bool AppendOrdinary(
        const simt::SimtWriterRegion *entries,
        const uint16_t *buckets, uint32_t count, uint32_t task
    )
    {
        std::array<SharedRegionValue, plan::kMaxTaskTensors> converted{};
        if (!ConvertEntries(entries, count, converted)) return false;
        return SharedAppendPreparedTask<CpuOps, false>(
            state.shared_map, converted.data(), buckets, count,
            static_cast<int32_t>(task)
        );
    }

    bool LookupOrdinary(
        const simt::SimtCanonicalTensorDesc &descriptor,
        uint32_t task, int32_t &producer
    )
    {
        if (task == kDelayedOrdinaryReader) {
            const auto deadline =
                std::chrono::steady_clock::now() + kDeadline;
            if (!WaitUntil(deadline, [&] {
                    return CpuOps::Load(
                               InsertCompletion(
                                   kFutureOrdinaryWriter
                               )
                           ) == static_cast<int64_t>(
                                    kFutureOrdinaryWriter
                                ) ||
                           CpuOps::Load(GlobalFatal()) != 0;
                }) || CpuOps::Load(GlobalFatal()) != 0) {
                return false;
            }
            future_lookup_observed->fetch_add(
                1U, std::memory_order_relaxed
            );
        }
        simt::SimtWriterRegion region{};
        if (!simt::SimtMakeWriterRegion(descriptor, task, region)) {
            return false;
        }
        const SharedRegionValue query{
            region.buffer_addr, region.lo, region.hi, -1, 0U,
        };
        bool protocol_ok = false;
        producer = SharedLookupRegion<CpuOps, false, true>(
            state.shared_map, query, static_cast<int32_t>(task),
            state.heap_window, protocol_ok
        );
        return protocol_ok;
    }

    volatile int64_t *InsertCompletion(uint32_t task)
    {
        return task < task_count
            ? &state.claim_tournament[task]
                   .root.insert_completion.value
            : nullptr;
    }

    volatile int32_t *GlobalFatal()
    {
        return &state.fatal.value;
    }

    simt::SimtWriterHistoryCell *WriterHistory(uint32_t task)
    {
        if (task >= task_count) return nullptr;
        return reinterpret_cast<simt::SimtWriterHistoryCell *>(
            &state.shared_map.writer_history[task]
        );
    }

    uint32_t TaskCapacity() const
    {
        return task_count;
    }

    bool FaninLowerBound(
        uint32_t reader_task, int32_t &lower_bound
    ) const
    {
        if (reader_task > static_cast<uint32_t>(INT32_MAX) ||
            state.heap_window < 0) {
            return false;
        }
        const uint32_t window = static_cast<uint32_t>(
            state.heap_window
        );
        lower_bound = static_cast<int32_t>(
            reader_task > window ? reader_task - window : 0U
        );
        return true;
    }

    uint64_t WatchdogTicks() const
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                kDeadline
            ).count()
        );
    }

    bool ValidateAdapterRoute(
        uint8_t flags, uint16_t data, uint32_t function,
        plan::EngineClass engine
    ) const
    {
        if (flags != kAdapterMagic || data >= task_count) return false;
        const TaskDefinition &definition = definitions[data];
        return definition.function_id == function &&
               definition.engine == engine;
    }

    bool PublishMetadataCompletion(uint32_t task, uint64_t vend)
    {
        if (task >= task_count) return false;
        const uint64_t prior_vend = CpuOps::Exchange(
            &state.tasks[task].vend, vend
        );
        CpuOps::StoreBarrier();
        const int64_t prior_flag = CpuOps::CompareExchange(
            &state.tasks[task].flag, 0, 1
        );
        return prior_vend == 0U && prior_flag == 0;
    }

    exec::SharedExecCell &ExecCell(uint32_t task)
    {
        return state.exec_cells[task];
    }

    exec::SharedExecFatalControl &ExecFatal()
    {
        return state.exec_fatal;
    }

    bool PublishBuildFatal(uint32_t task, uint32_t status)
    {
        uint32_t no_task = UINT32_MAX;
        (void)fatal_task->compare_exchange_strong(
            no_task, task, std::memory_order_acq_rel
        );
        uint32_t no_status = UINT32_MAX;
        (void)fatal_status->compare_exchange_strong(
            no_status, status, std::memory_order_acq_rel
        );
        int32_t scheduler_expected = 0;
        (void)__atomic_compare_exchange_n(
            &state.fatal.value, &scheduler_expected, 1, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        int64_t expected = 0;
        (void)__atomic_compare_exchange_n(
            &plan_control.fatal.value, &expected, 1, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return true;
    }

private:
    static bool ConvertEntries(
        const simt::SimtWriterRegion *entries, uint32_t count,
        std::array<SharedRegionValue, plan::kMaxTaskTensors> &converted
    )
    {
        if (count > converted.size() ||
            (count != 0U && entries == nullptr)) {
            return false;
        }
        for (uint32_t index = 0U; index < count; ++index) {
            converted[index] = SharedRegionValue{
                entries[index].buffer_addr,
                entries[index].lo,
                entries[index].hi,
                entries[index].producer,
                entries[index].reserved,
            };
        }
        return true;
    }
};

struct RuntimeObservations {
    std::atomic<uint32_t> fatal_task{UINT32_MAX};
    std::atomic<uint32_t> fatal_status{UINT32_MAX};
    std::atomic<uint32_t> future_lookup_observed{0U};
};

NarrowRuntime MakeRuntime(
    RuntimeFixture &fixture, PlanFixture &plan_fixture,
    const TaskDefinition *definitions, uint32_t task_count,
    RuntimeObservations &observations
)
{
    return NarrowRuntime{
        *fixture.state,
        plan_fixture.control,
        definitions,
        task_count,
        &observations.fatal_task,
        &observations.fatal_status,
        &observations.future_lookup_observed,
        UINT32_MAX,
        UINT32_MAX,
    };
}

int32_t ExecFanin(
    const exec::SharedExecCell &cell,
    const exec::ExecPayloadLayout &layout, uint32_t edge
)
{
    const uint64_t packed = cell.payload.words[
        layout.fanin_word_offset + edge / 2U
    ];
    return static_cast<int32_t>(
        edge % 2U == 0U
            ? static_cast<uint32_t>(packed)
            : static_cast<uint32_t>(packed >> 32U)
    );
}

bool ExpectBuiltTask(
    const SchedulerState &state,
    const std::array<TaskDefinition, kTaskCount> &tasks,
    uint32_t task, const int32_t *expected_fanin,
    uint32_t expected_fanin_count
)
{
    const exec::SharedExecCell &cell = state.exec_cells[task];
    const exec::DecodedExecState decoded =
        exec::DecodeExecState(cell.control.state);
    const exec::ExecPayloadHeader header =
        exec::DecodeExecPayloadHeader(cell.payload);
    exec::ExecPayloadLayout layout{};
    if (!decoded.valid || decoded.phase != exec::ExecPhase::Built ||
        decoded.task_id != task ||
        decoded.build_owner >= simt::kBuilderLeaders ||
        header.task_id != task ||
        header.function_id != tasks[task].function_id ||
        header.tensor_count != tasks[task].tensor_count ||
        header.scalar_count != tasks[task].scalar_count ||
        header.fanin_count != expected_fanin_count ||
        header.tensor_reference_mask != 0U ||
        !exec::ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count, header.tensor_reference_mask, layout
        ) || header.payload_bytes != layout.payload_bytes ||
        decoded.payload_lines != layout.payload_lines) {
        return false;
    }
    for (uint32_t edge = 0U; edge < expected_fanin_count; ++edge) {
        if (ExecFanin(cell, layout, edge) != expected_fanin[edge]) {
            return false;
        }
    }
    if (tasks[task].tensor_count != 0U) {
        const uint64_t address = cell.payload.words[
            layout.tensor_word_offset
        ];
        if (tasks[task].tensors[0].reference) {
            const TensorDesc &resolved =
                state.shared_map.shared_outputs[0].tensors[0];
            if (address != resolved.buffer_addr) return false;
        } else if (address != tasks[task].tensors[0].tensor.buffer_addr) {
            return false;
        }
    }
    return true;
}

void TestFourLeaderFullBuild()
{
    const std::array<TaskDefinition, kTaskCount> tasks = MakeWorkload();
    PlanFixture plan_fixture(kTaskCount);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate real SchedulerState");
    Expect(PublishPlan(plan_fixture, tasks, kTaskCount),
           "failed to publish canonical Plan v2 workload");
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(),
        kTaskCount, observations
    );

    uint32_t attached_count = 0U;
    Expect(simt::AttachClosedPlan<CpuOps>(
               plan_fixture.view, attached_count
           ) && attached_count == kTaskCount,
           "four leaders failed to attach the closed Plan");

    std::array<std::thread, simt::kBuilderLeaders> leaders;
    std::array<simt::BuilderLeaderLocalState, simt::kBuilderLeaders>
        leader_states{};
    std::array<std::atomic<uint32_t>, kTaskCount> visits{};
    std::array<std::atomic<uint32_t>, kTaskCount> entered{};
    std::array<std::atomic<uint32_t>, kTaskCount> statuses{};
    std::atomic<bool> ok{true};
    std::atomic<bool> start{false};
    std::atomic<bool> release_task_zero{false};
    std::atomic<uint32_t> last_arrivals{0U};
    const auto deadline = std::chrono::steady_clock::now() + kDeadline;

    const auto fail = [&] {
        ok.store(false, std::memory_order_release);
        release_task_zero.store(true, std::memory_order_release);
        start.store(true, std::memory_order_release);
    };

    for (uint32_t leader = 0U;
         leader < simt::kBuilderLeaders; ++leader) {
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
                    simt::TakeAttachedBuildTicket<CpuOps>(
                        plan_fixture.view, attached_count
                    );
                if (reservation.status ==
                    plan::BuildReservationStatus::Closed) {
                    const plan::BuildArrivalStatus arrival =
                        simt::ArriveBuilderLeaderOnce<CpuOps>(
                            plan_fixture.view, leader,
                            leader_states[leader]
                        );
                    if (arrival == plan::BuildArrivalStatus::Last) {
                        last_arrivals.fetch_add(
                            1U, std::memory_order_relaxed
                        );
                        if (CpuOps::Load(runtime.InsertCompletion(
                                kTaskCount - 1U
                            )) != static_cast<int64_t>(
                                     kTaskCount - 1U
                                 ) ||
                            !simt::PublishBuildRelease<CpuOps>(
                                plan_fixture.view, kTaskCount
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
                               ) ||
                               !ok.load(std::memory_order_acquire);
                    })) {
                    fail();
                    return;
                }
                simt::SimtTaskBuildScratch scratch{};
                const simt::SimtTaskBuildStatus status =
                    simt::BuildCanonicalPlanTask<CpuOps>(
                        runtime, plan_fixture.view, task,
                        leader, scratch
                    );
                statuses[task].store(
                    static_cast<uint32_t>(status),
                    std::memory_order_release
                );
                if (status != simt::SimtTaskBuildStatus::Published) {
                    fail();
                    return;
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    Expect(WaitUntil(deadline, [&] {
               uint32_t count = 0U;
               for (uint32_t task = 0U; task < 4U; ++task) {
                   count += entered[task].load(std::memory_order_acquire);
               }
               return count == 4U || !ok.load(std::memory_order_acquire);
           }), "initial four leader tickets did not enter");
    for (uint32_t task = 0U; task < 4U; ++task) {
        Expect(CpuOps::Load(runtime.InsertCompletion(task)) ==
                   SharedInsertCompletionInitialValue(task),
               "a successor passed the deliberately held task 0");
    }
    Expect(CpuOps::Load(&runtime_fixture.state->shared_map
                             .buckets[TensorMapHash(kOrdinaryAddress)]
                             .tail.value) == 0,
           "ordinary TensorMap changed before task 0 release");
    release_task_zero.store(true, std::memory_order_release);
    for (auto &leader : leaders) leader.join();

    Expect(ok.load(std::memory_order_acquire),
           "four-leader full Build protocol failed");
    Expect(CpuOps::Load(&plan_fixture.control.fatal.value) == 0 &&
               CpuOps::Load(&runtime_fixture.state->fatal.value) == 0,
           "successful workload published fatal");
    Expect(CpuOps::LoadControl(&plan_fixture.control.build_next.value) ==
               static_cast<int64_t>(kTaskCount + simt::kBuilderLeaders),
           "four-leader terminal ticket was not N+4");
    Expect(CpuOps::LoadControl(
               &plan_fixture.control.build_workers_done.value
           ) == static_cast<int64_t>(simt::kBuilderLeaders) &&
               CpuOps::LoadControl(
                   &plan_fixture.control.build_release.value
               ) == static_cast<int64_t>(kTaskCount) &&
               last_arrivals.load(std::memory_order_acquire) == 1U,
           "four distinct leaders did not publish one release");
    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        Expect(visits[task].load(std::memory_order_acquire) == 1U &&
                   statuses[task].load(std::memory_order_acquire) ==
                       static_cast<uint32_t>(
                           simt::SimtTaskBuildStatus::Published
                       ),
               "a canonical Plan task was not built exactly once");
        Expect(CpuOps::Load(runtime.InsertCompletion(task)) ==
                   static_cast<int64_t>(task),
               "strict completion did not reach its exact task id");
        Expect(CpuOps::completion_order[task].load(
                   std::memory_order_acquire
               ) == static_cast<int32_t>(task),
               "strict completion publication order changed");
    }
    Expect(CpuOps::completion_count.load(std::memory_order_acquire) ==
               kTaskCount &&
               CpuOps::strict_order_errors.load(
                   std::memory_order_acquire
               ) == 0U,
           "strict completion CAS contract was violated");

    const SharedOutputCell &output =
        runtime_fixture.state->shared_map.shared_outputs[0];
    Expect(CpuOps::Load(const_cast<volatile int64_t *>(
               &output.published[0].value)) == 0 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &output.last_writer[0].value)) == 6,
           "fresh output publication or symbol latest-writer is wrong");
    Expect(output.tensors[0].buffer_addr ==
               runtime_fixture.state->heap_base &&
               output.tensors[0].buffer_size == 1024U &&
               output.tensors[0].owner_task_id == 0U &&
               output.tensors[0].ndims == 1U &&
               output.tensors[0].shapes[0] == 256U &&
               output.tensors[0].strides[0] == 1U,
           "fresh output did not materialize into real SharedOutputCell");

    const SharedWriterHistoryCell &history1 =
        runtime_fixture.state->shared_map.writer_history[1];
    const SharedWriterHistoryCell &history6 =
        runtime_fixture.state->shared_map.writer_history[6];
    Expect(history1.magic == simt::kSimtWriterHistoryMagic &&
               history1.writer_task == 1 && history1.count == 1U &&
               history1.entries[0].symbol_key == 1U &&
               history1.entries[0].previous_writer == 0 &&
               history6.magic == simt::kSimtWriterHistoryMagic &&
               history6.writer_task == 6 && history6.count == 1U &&
               history6.entries[0].symbol_key == 1U &&
               history6.entries[0].previous_writer == 1,
           "symbol writer history did not preserve its predecessor chain");

    const uint32_t bucket = TensorMapHash(kOrdinaryAddress);
    Expect(CpuOps::Load(&runtime_fixture.state->shared_map
                             .buckets[bucket].tail.value) == 2,
           "ordinary_count>0 did not append both real map entries");
    SharedRegionValue first{};
    SharedRegionValue second{};
    Expect(SharedReadRegionSlot<CpuOps>(
               runtime_fixture.state->shared_map, bucket, 0U, first
           ) && SharedReadRegionSlot<CpuOps>(
               runtime_fixture.state->shared_map, bucket, 1U, second
           ) && first.producer == 3 && second.producer == 7,
           "ordinary TensorMap writer order is not task-id strict");
    Expect(observations.future_lookup_observed.load(
               std::memory_order_acquire
           ) == 1U,
           "future ordinary writer was not present during reader lookup");

    constexpr int32_t kTask1Fanin[] = {0};
    constexpr int32_t kTask2Fanin[] = {1};
    constexpr int32_t kTask4Fanin[] = {3, 0};
    constexpr int32_t kTask6Fanin[] = {1};
    constexpr int32_t kTask7Fanin[] = {3};
    Expect(ExpectBuiltTask(*runtime_fixture.state, tasks, 1U,
                           kTask1Fanin, 1U),
           "task 1 symbol fanin/BUILT payload is wrong");
    Expect(ExpectBuiltTask(*runtime_fixture.state, tasks, 2U,
                           kTask2Fanin, 1U),
           "task 2 future-symbol filtering is wrong");
    Expect(ExpectBuiltTask(*runtime_fixture.state, tasks, 3U,
                           nullptr, 0U),
           "task 3 self ordinary writer was not filtered");
    Expect(ExpectBuiltTask(*runtime_fixture.state, tasks, 4U,
                           kTask4Fanin, 2U),
           "task 4 ordinary/explicit fanin or future filtering is wrong");
    Expect(ExpectBuiltTask(*runtime_fixture.state, tasks, 6U,
                           kTask6Fanin, 1U),
           "task 6 symbol history fanin is wrong");
    Expect(ExpectBuiltTask(*runtime_fixture.state, tasks, 7U,
                           kTask7Fanin, 1U),
           "task 7 ordinary predecessor is wrong");
    Expect(exec::DecodeExecState(
               runtime_fixture.state->exec_cells[0].control.state
           ).phase == exec::ExecPhase::Empty &&
               exec::DecodeExecState(
                   runtime_fixture.state->exec_cells[5].control.state
               ).phase == exec::ExecPhase::Empty &&
               CpuOps::Load(&runtime_fixture.state->tasks[0].flag) == 1 &&
               CpuOps::Load(&runtime_fixture.state->tasks[5].flag) == 1,
           "metadata tasks did not use TaskCell-only terminal publication");
}

void TestCorruptPlanFailsBeforeSideEffects()
{
    const auto tasks = MakeWorkload();
    PlanFixture plan_fixture(1U);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate corrupt-plan fixture");
    const TaskPlanSource source{tasks[0]};
    Expect(plan::PublishRuntimeTaskPlan<CpuOps>(
               plan_fixture.view, MakePlanSpec(0U, tasks[0]), source
           ) == plan::PlanPublishResult::Published &&
               plan::AdvancePlannedFrontier<CpuOps>(
                   plan_fixture.view, 0U
               ) && plan::CloseRuntimePlan<CpuOps>(
                   plan_fixture.view, 1U
               ),
           "failed to publish corrupt-plan seed");
    plan_fixture.cells.get()[0].payload.words[7] ^=
        UINT64_C(1) << 32U;
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(), 1U, observations
    );
    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, plan_fixture.view, 0U, 0U, scratch
        );
    Expect(status == simt::SimtTaskBuildStatus::InvalidPlanPayload &&
               CpuOps::Load(&runtime_fixture.state->fatal.value) != 0 &&
               CpuOps::LoadControl(&plan_fixture.control.fatal.value) != 0 &&
               CpuOps::Load(runtime.InsertCompletion(0U)) == -1 &&
               CpuOps::Load(&runtime_fixture.state->tasks[0].flag) == 0 &&
               CpuOps::Load(&runtime_fixture.state->shared_map
                                  .shared_outputs[0].published[0].value) ==
                   -1 &&
               exec::DecodeExecState(
                   runtime_fixture.state->exec_cells[0].control.state
               ).phase == exec::ExecPhase::Empty,
           "corrupt canonical Plan did not fail before all side effects");
}

void TestBrokenPredecessorFailsClosed()
{
    auto tasks = MakeWorkload();
    tasks[0] = TaskDefinition{};
    tasks[1] = TaskDefinition{};
    tasks[1].engine = plan::EngineClass::Aic;
    tasks[1].function_id = 911U;
    PlanFixture plan_fixture(2U);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate predecessor fixture");
    Expect(PublishPlan(plan_fixture, tasks, 2U),
           "failed to publish predecessor fixture Plan");
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(), 2U, observations
    );
    runtime_fixture.state->claim_tournament[0]
        .root.insert_completion.value = 7;
    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, plan_fixture.view, 1U, 0U, scratch
        );
    Expect(status == simt::SimtTaskBuildStatus::InsertWaitFailed &&
               CpuOps::Load(&runtime_fixture.state->fatal.value) != 0 &&
               CpuOps::Load(runtime.InsertCompletion(1U)) == 0 &&
               exec::DecodeExecState(
                   runtime_fixture.state->exec_cells[1].control.state
               ).phase == exec::ExecPhase::Empty,
           "out-of-order predecessor state did not fail closed");
}

void TestExecConflictFailsClosed()
{
    auto tasks = MakeWorkload();
    tasks[0] = TaskDefinition{};
    tasks[0].engine = plan::EngineClass::Aiv;
    tasks[0].function_id = 919U;
    PlanFixture plan_fixture(1U);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate exec-conflict fixture");
    Expect(PublishPlan(plan_fixture, tasks, 1U),
           "failed to publish exec-conflict Plan");
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(), 1U, observations
    );
    const int64_t occupied = static_cast<int64_t>(exec::EncodeExecState(
        exec::ExecPhase::Built, 1U, exec::kExecUnboundOwner,
        exec::ExecEngineClass::Aiv, 1U, 0U
    ));
    runtime_fixture.state->exec_cells[0].control.state = occupied;
    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, plan_fixture.view, 0U, 0U, scratch
        );
    Expect(status == simt::SimtTaskBuildStatus::ExecPublishFailed &&
               CpuOps::Load(&runtime_fixture.state->fatal.value) != 0 &&
               CpuOps::Load(runtime.InsertCompletion(0U)) == 0 &&
               runtime_fixture.state->exec_cells[0].control.state ==
                   occupied,
           "occupied SharedExecCell was overwritten or not failed closed");
}

void TestSecondOutputPublishConflictRollsBackOwnedPrefix()
{
    const auto tasks = MakeTwoOutputWorkload();
    PlanFixture plan_fixture(1U);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate output-conflict fixture");
    Expect(PublishPlan(plan_fixture, tasks, 1U),
           "failed to publish two-output conflict Plan");
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(), 1U, observations
    );
    constexpr int64_t kForeignPublication = 73;
    runtime_fixture.state->shared_map.shared_outputs[0]
        .published[1].value = kForeignPublication;

    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, plan_fixture.view, 0U, 0U, scratch
        );
    const SharedOutputCell &outputs =
        runtime_fixture.state->shared_map.shared_outputs[0];
    Expect(status == simt::SimtTaskBuildStatus::MaterializeFailed &&
               CpuOps::Load(&runtime_fixture.state->fatal.value) != 0 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.last_writer[0].value
               )) == -1 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.last_writer[1].value
               )) == -1 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.published[0].value
               )) == -1 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.published[1].value
               )) == kForeignPublication &&
               scratch.reserved_output_count == 0U &&
               scratch.published_output_count == 0U &&
               !scratch.insert_completion_published &&
               CpuOps::Load(runtime.InsertCompletion(0U)) == -1 &&
               CpuOps::LoadControl(
                   &plan_fixture.control.build_release.value
               ) == plan::kBuildReleasePending,
           "second-output conflict did not reverse owned publication/reservation prefixes");
}

void TestSecondOutputDescriptorFailureRollsBackReservations()
{
    const auto tasks = MakeTwoOutputWorkload();
    PlanFixture plan_fixture(1U);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate descriptor-failure fixture");
    Expect(PublishPlan(plan_fixture, tasks, 1U),
           "failed to publish descriptor-failure Plan");
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(), 1U, observations
    );
    runtime.fail_descriptor_task = 0U;
    runtime.fail_descriptor_output = 1U;

    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, plan_fixture.view, 0U, 0U, scratch
        );
    const SharedOutputCell &outputs =
        runtime_fixture.state->shared_map.shared_outputs[0];
    Expect(status == simt::SimtTaskBuildStatus::MaterializeFailed &&
               CpuOps::Load(&runtime_fixture.state->fatal.value) != 0 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.last_writer[0].value
               )) == -1 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.last_writer[1].value
               )) == -1 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.published[0].value
               )) == -1 &&
               CpuOps::Load(const_cast<volatile int64_t *>(
                   &outputs.published[1].value
               )) == -1 &&
               scratch.reserved_output_count == 0U &&
               scratch.published_output_count == 0U &&
               CpuOps::Load(&runtime_fixture.state->shared_map
                                  .shared_heap_vend.value) > 0 &&
               CpuOps::Load(runtime.InsertCompletion(0U)) == -1 &&
               CpuOps::LoadControl(
                   &plan_fixture.control.build_release.value
               ) == plan::kBuildReleasePending,
           "descriptor failure did not roll back controls or preserve terminal heap advance");
}

void TestNoDependencySharedOutputReferenceFailsBeforeSideEffects()
{
    auto tasks = MakeWorkload();
    tasks[0] = TaskDefinition{};
    tasks[1] = TaskDefinition{};
    tasks[1].engine = plan::EngineClass::Aiv;
    tasks[1].function_id = 929U;
    tasks[1].tensors[0] = OutputReference(
        plan::TensorTag::NoDependency
    );
    tasks[1].tensor_count = 1U;
    PlanFixture plan_fixture(2U);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate invalid-reference fixture");
    Expect(PublishPlan(plan_fixture, tasks, 2U),
           "failed to publish invalid-reference Plan wire");
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(), 2U, observations
    );

    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, plan_fixture.view, 1U, 0U, scratch
        );
    Expect(status == simt::SimtTaskBuildStatus::InvalidPlanPayload &&
               CpuOps::Load(&runtime_fixture.state->fatal.value) != 0 &&
               CpuOps::Load(runtime.InsertCompletion(1U)) == 0 &&
               CpuOps::LoadControl(
                   &plan_fixture.control.build_release.value
               ) == plan::kBuildReleasePending &&
               exec::DecodeExecState(
                   runtime_fixture.state->exec_cells[1].control.state
               ).phase == exec::ExecPhase::Empty,
           "NoDependency SharedOutputRef was not rejected before side effects");
}

void TestAllFaninSourcesHonorHeapWindow()
{
    auto tasks = MakeWorkload();
    tasks[3] = TaskDefinition{};
    tasks[3].engine = plan::EngineClass::Aiv;
    tasks[3].function_id = 944U;
    tasks[3].tensors[0] = OutputReference(plan::TensorTag::Input);
    simt::SimtCanonicalTensorDesc ordinary = OrdinaryTensor();
    ordinary.owner_task_id = 0U;
    tasks[3].tensors[1] = PlainTensor(
        plan::TensorTag::Input, ordinary
    );
    tasks[3].tensor_count = 2U;
    tasks[3].explicit_dependencies[0] = 1U;
    tasks[3].explicit_dep_count = 1U;

    PlanFixture plan_fixture(4U);
    RuntimeFixture runtime_fixture;
    Expect(runtime_fixture.Valid(), "failed to allocate fanin-window fixture");
    runtime_fixture.state->heap_window = 1;
    Expect(PublishPlan(plan_fixture, tasks, 4U),
           "failed to publish fanin-window Plan");
    RuntimeObservations observations;
    NarrowRuntime runtime = MakeRuntime(
        runtime_fixture, plan_fixture, tasks.data(), 4U, observations
    );

    SharedOutputCell &origin =
        runtime_fixture.state->shared_map.shared_outputs[0];
    origin.published[0].value = 0;
    origin.last_writer[0].value = 0;
    simt::SimtCanonicalTensorDesc origin_descriptor = OrdinaryTensor();
    origin_descriptor.owner_task_id = 0U;
    std::memcpy(
        &origin.tensors[0], &origin_descriptor,
        sizeof(origin.tensors[0])
    );

    simt::SimtWriterRegion old_region{};
    Expect(simt::SimtMakeWriterRegion(ordinary, 1U, old_region),
           "failed to make stale ordinary writer region");
    const SharedRegionValue old_entry{
        old_region.buffer_addr, old_region.lo, old_region.hi,
        old_region.producer, old_region.reserved,
    };
    const uint16_t old_bucket = static_cast<uint16_t>(
        TensorMapHash(old_region.buffer_addr)
    );
    Expect(SharedAppendPreparedTask<CpuOps, false>(
               runtime_fixture.state->shared_map, &old_entry,
               &old_bucket, 1U, 1
           ), "failed to seed stale ordinary writer");
    runtime_fixture.state->claim_tournament[2]
        .root.insert_completion.value = 2;

    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, plan_fixture.view, 3U, 0U, scratch
        );
    Expect(status == simt::SimtTaskBuildStatus::Published &&
               scratch.fanin_count == 0U &&
               ExpectBuiltTask(
                   *runtime_fixture.state, tasks, 3U, nullptr, 0U
               ),
           "symbol/owner/ordinary/explicit producers outside [N-H,N) leaked into Exec payload");
}

}  // namespace

int main()
{
    TestSharedSymbolHistoryKeyAbi();
    TestFourLeaderFullBuild();
    TestCorruptPlanFailsBeforeSideEffects();
    TestBrokenPredecessorFailsClosed();
    TestExecConflictFailsClosed();
    TestSecondOutputPublishConflictRollsBackOwnedPrefix();
    TestSecondOutputDescriptorFailureRollsBackReservations();
    TestNoDependencySharedOutputReferenceFailsBeforeSideEffects();
    TestAllFaninSourcesHonorHeapWindow();
    std::puts("[PASS] narrow SIMT Plan task builder CPU gate");
    return 0;
}
