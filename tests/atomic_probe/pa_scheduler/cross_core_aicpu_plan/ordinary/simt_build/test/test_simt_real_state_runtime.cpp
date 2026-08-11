/* CPU mapping gate for the narrow SIMT builder's production-state adapter. */

#define PTO_FDWIC_SHARED_MAP 1
#define PTO_FDWIC_TENSORMAP_RING_CAP 128
#define PTO_FDWIC_SHARED_INSERT_TURN_GROUPS 1
#define PA_RUNTIME_PLAN_BUILD_WORKERS 4
#define PA_BUILD_PERF_CLOCK 1
#define PA_DEVICE inline
#define PA_GM

#include "../common/simt_real_state_runtime.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sys/mman.h>

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace simt = pa_scheduler::aicpu_plan_simt;
using namespace pa_scheduler;

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(1);
}

void Expect(bool condition, const char *message)
{
    if (!condition) Fail(message);
}

struct CpuOps {
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

    static int32_t Exchange(
        volatile int32_t *address, int32_t value
    )
    {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(
        volatile int64_t *address, int64_t value
    )
    {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static uint64_t Exchange(
        volatile uint64_t *address, uint64_t value
    )
    {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t CompareExchange(
        volatile int64_t *address,
        int64_t expected, int64_t desired
    )
    {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }

    static int64_t FetchAdd(
        volatile int64_t *address, int64_t value
    )
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchAddControl(
        volatile int64_t *address, int64_t value
    )
    {
        return FetchAdd(address, value);
    }

    static void PublishControl(
        volatile int64_t *address, int64_t value
    )
    {
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }

    static void StorePayloadWord(
        volatile uint64_t *address, uint64_t value
    )
    {
        *address = value;
    }

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    template <typename Pointer>
    static void FlushRegion(Pointer, uint64_t)
    {
        StoreBarrier();
    }

    template <typename Pointer>
    static void InvalidateRegion(Pointer, uint64_t)
    {
        StoreBarrier();
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
    }
};

struct RoutePolicy {
    bool Validate(
        uint32_t task_id, uint8_t adapter_flags,
        uint16_t adapter_data, uint32_t function_id,
        plan::EngineClass engine
    ) const
    {
        const bool metadata =
            engine == plan::EngineClass::MetadataOnly &&
            function_id == plan::kInvalidFunctionId;
        const bool aic = engine == plan::EngineClass::Aic &&
                         function_id == 7U;
        return task_id == adapter_data && adapter_flags == 0xA5U &&
               (metadata || aic);
    }
};

using Runtime = simt::SimtRealStateRuntime<CpuOps, RoutePolicy>;

struct EmptyPlanSource {
    plan::TensorTag TensorTagAt(uint32_t) const
    {
        return plan::TensorTag::NoDependency;
    }

    bool TensorIsReference(uint32_t) const
    {
        return false;
    }

    uint64_t TensorWord(uint32_t, uint32_t) const
    {
        return 0U;
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

struct SingleTensorPlanSource {
    plan::TensorTag tag;
    bool reference;
    uint64_t words[plan::kTensorCanonicalWords];

    plan::TensorTag TensorTagAt(uint32_t tensor) const
    {
        return tensor == 0U ? tag : plan::TensorTag::NoDependency;
    }

    bool TensorIsReference(uint32_t tensor) const
    {
        return tensor == 0U && reference;
    }

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const
    {
        return tensor == 0U && word < plan::kTensorCanonicalWords
            ? words[word]
            : 0U;
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

SingleTensorPlanSource MakeOutputSource()
{
    SingleTensorPlanSource source{};
    source.tag = plan::TensorTag::Output;
    source.reference = false;
    source.words[4] = uint64_t{1} << 32U;  // version=0, ndims=1
    source.words[5] =
        static_cast<uint64_t>(DataType::Float32) |
        (uint64_t{1} << 16U) |  // contiguous
        (uint64_t{16} << 32U);  // shape[0]
    return source;
}

SingleTensorPlanSource MakeReferenceSource(plan::TensorTag tag)
{
    SingleTensorPlanSource source{};
    source.tag = tag;
    source.reference = true;
    const plan::RuntimeOutputReferenceWire reference{
        /*producer_task_id=*/0,
        /*output_slot=*/0,
        /*flags=*/0U,
        /*view_ndims=*/0U,
        /*view_shape0=*/0U,
        /*view_offset0=*/0U,
    };
    source.words[0] = plan::RuntimeOutputReferenceWireWord(reference, 0U);
    source.words[1] = plan::RuntimeOutputReferenceWireWord(reference, 1U);
    return source;
}

SchedulerState *MapState()
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

void UnmapState(SchedulerState *state)
{
    if (state != nullptr) {
        (void)munmap(state, sizeof(SchedulerState));
    }
}

plan::RuntimeTaskPlanCell *AllocatePlanCells(uint32_t count)
{
    void *memory = nullptr;
    const size_t bytes =
        static_cast<size_t>(count) * sizeof(plan::RuntimeTaskPlanCell);
    if (posix_memalign(
            &memory, plan::kAtomicIsolationBytes, bytes
        ) != 0) {
        return nullptr;
    }
    std::memset(memory, 0, bytes);
    return static_cast<plan::RuntimeTaskPlanCell *>(memory);
}

void TestMetadataBuildOnRealState()
{
    Expect(simt::SimtBuildOwner(0U) == 0U &&
               simt::SimtBuildOwner(3U) == 3U &&
               simt::SimtBuildOwner(4U) ==
                   simt::exec::kExecUnboundOwner,
           "SIMT Build owner does not satisfy the real A5 policy");
    SchedulerState *state = MapState();
    Expect(state != nullptr, "failed to map the real SchedulerState");
    plan::RuntimeTaskPlanCell *cells = AllocatePlanCells(3U);
    Expect(cells != nullptr, "failed to allocate Plan cells");

    state->heap_window = static_cast<int32_t>(kHeapWindow);
    state->runtime_plan_control.planned_frontier.value = 0;
    state->runtime_plan_control.closed_task_count.value =
        plan::kPlanOpenTaskCount;
    state->runtime_plan_control.build_next.value = 0;
    state->runtime_plan_control.build_workers_done.value = 0;
    state->runtime_plan_control.build_release.value =
        plan::kBuildReleasePending;
    state->runtime_plan_control.fatal.value = 0;
    state->fatal.value = 0;
    state->shared_map.shared_heap_vend.value = 0;
    state->claim_tournament[0].root.insert_completion.value =
        SharedInsertCompletionInitialValue(0U);
    state->tasks[0].flag = 0;
    state->tasks[0].vend = 0;

    const plan::RuntimePlanView view{
        &state->runtime_plan_control, cells, 3U,
    };
    const plan::RuntimeTaskPlanSpec spec{
        /*task_id=*/0U,
        /*function_id=*/plan::kInvalidFunctionId,
        /*tensor_count=*/0U,
        /*scalar_count=*/0U,
        /*explicit_dep_count=*/0U,
        /*output_count=*/0U,
        plan::EngineClass::MetadataOnly,
        /*adapter_flags=*/0xA5U,
        /*core_num=*/1,
        /*require_sync_start=*/0U,
        /*reserved=*/0U,
        /*adapter_data=*/0U,
        /*tensor_reference_mask=*/0U,
    };
    const EmptyPlanSource source{};
    Expect(
        plan::PublishRuntimeTaskPlan<CpuOps>(view, spec, source) ==
            plan::PlanPublishResult::Published &&
            plan::AdvancePlannedFrontier<CpuOps>(view, 0U) &&
            plan::CloseRuntimePlan<CpuOps>(view, 1U),
        "failed to publish the metadata-only Plan"
    );

    Runtime runtime{
        state, 3U, UINT32_MAX, kWatchdogTicks, RoutePolicy{},
    };
    Expect(runtime.BindTask(0U), "failed to bind task 0");
    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, view, 0U, simt::SimtBuildOwner(0U), scratch
        );
    Expect(status == simt::SimtTaskBuildStatus::Published,
           "metadata-only Build failed on real SchedulerState");
    Expect(state->fatal.value == 0 &&
               state->runtime_plan_control.fatal.value == 0,
           "successful real-state Build published fatal");
    Expect(state->claim_tournament[0]
                   .root.insert_completion.value == 0,
           "real insert-completion line was not published");
    Expect(state->tasks[0].vend == 0U && state->tasks[0].flag == 1,
           "metadata completion did not use the real TaskCell");

    std::free(cells);
    UnmapState(state);
}

void TestRealRingAndEightShardHeap()
{
    SchedulerState *state = MapState();
    Expect(state != nullptr, "failed to map ring fixture");
    state->heap_window = static_cast<int32_t>(kHeapWindow);
    state->heap_base = UINT64_C(0x100000000);
    state->heap_size = UINT64_C(0x8000);
    state->shared_map.shared_heap_vend.value = 0;
    for (uint32_t shard = 0U; shard < kSharedHeapShards; ++shard) {
        state->shared_map.shared_heap_cursor[shard].value = 0;
    }

    Runtime runtime{
        state, 3U, UINT32_MAX, kWatchdogTicks, RoutePolicy{},
    };
    Expect(runtime.BindTask(1U), "failed to bind task 1");
    uint64_t first = 0U;
    uint64_t vend = 0U;
    Expect(runtime.ReserveOutputHeap(1U, 1024U, first, vend),
           "real eight-shard heap reservation failed");
    const uint64_t shard_span = state->heap_size / kSharedHeapShards;
    Expect(first == state->heap_base + shard_span && vend == 1024U,
           "heap reservation did not map the real task-id shard");
    Expect(state->shared_map.shared_heap_cursor[1].value == 1024 &&
               state->shared_map.shared_heap_vend.value == 1024,
           "real heap cursor/vend controls were not updated");

    simt::SimtCanonicalTensorDesc descriptor{};
    descriptor.buffer_addr = UINT64_C(0x900000000);
    descriptor.buffer_size = 4096U;
    descriptor.owner_task_id = simt::kSimtInvalidTaskOwner;
    descriptor.ndims = 1U;
    descriptor.dtype = static_cast<uint8_t>(DataType::Float32);
    descriptor.is_contiguous = 1U;
    descriptor.shapes[0] = 1024U;
    descriptor.extent_elem_cache = 1024U;
    descriptor.strides[0] = 1U;
    simt::SimtWriterRegion entries[2]{};
    Expect(simt::SimtMakeWriterRegion(descriptor, 1U, entries[0]),
           "failed to create ordinary writer region");
    entries[1] = simt::SimtWriterRegion{
        descriptor.buffer_addr,
        entries[0].hi,
        entries[0].hi + 64U,
        /*producer=*/1,
        /*reserved=*/0U,
    };
    const uint32_t bucket = runtime.OrdinaryBucket(
        descriptor.buffer_addr
    );
    state->shared_map.buckets[bucket].head.value = 0;
    state->shared_map.buckets[bucket].tail.value = 0;
    SharedRegionSlot &slot0 = state->shared_map.slots[
        SharedTensorMapSlotIndex(bucket, 0U)
    ];
    SharedRegionSlot &slot1 = state->shared_map.slots[
        SharedTensorMapSlotIndex(bucket, 1U)
    ];
    slot0.seq.value = kSharedMapEmptySeq;
    slot1.seq.value = kSharedMapEmptySeq;
    const uint16_t buckets[2] = {
        static_cast<uint16_t>(bucket),
        static_cast<uint16_t>(bucket),
    };
    const uint8_t ordinals[2] = {0U, 1U};
    Expect(runtime.CheckOrdinaryAppend(
               entries, buckets, ordinals, 2U, 1U
           ), "real generic preflight rejected same-bucket entries");
    Expect(runtime.AppendOrdinary(entries, buckets, 2U, 1U),
           "same-bucket stcg append failed on the real ring");
    Expect(state->shared_map.buckets[bucket].tail.value == 2 &&
               slot0.seq.value == 0 && slot1.seq.value == 1 &&
               slot0.payload.value.producer == 1 &&
               slot1.payload.value.lo == entries[1].lo,
           "real same-bucket absolute-seq/tail publication is wrong");

    int32_t producer = -1;
    Expect(runtime.LookupOrdinary(descriptor, 2U, producer) &&
               producer == 1,
           "real generic lookup lost the inserted producer");

    Expect(runtime.OutputPublished(1U, 0U) ==
               &state->shared_map.shared_outputs[1].published[0].value &&
               runtime.OutputLastWriter(1U, 0U) ==
               &state->shared_map.shared_outputs[1]
                    .last_writer[0].value &&
               runtime.WriterHistory(1U) ==
                   &state->shared_map.writer_history[1] &&
               &runtime.ExecCell(1U) == &state->exec_cells[1],
           "runtime returned a mirror instead of real state addresses");

    UnmapState(state);
}

void TestExecBuildOwnerOnRealState()
{
    SchedulerState *state = MapState();
    Expect(state != nullptr, "failed to map exec-owner fixture");
    plan::RuntimeTaskPlanCell *cells = AllocatePlanCells(1U);
    Expect(cells != nullptr, "failed to allocate exec-owner Plan cell");

    state->heap_window = static_cast<int32_t>(kHeapWindow);
    state->runtime_plan_control.planned_frontier.value = 0;
    state->runtime_plan_control.closed_task_count.value =
        plan::kPlanOpenTaskCount;
    state->runtime_plan_control.build_next.value = 0;
    state->runtime_plan_control.build_workers_done.value = 0;
    state->runtime_plan_control.build_release.value =
        plan::kBuildReleasePending;
    state->runtime_plan_control.fatal.value = 0;
    state->fatal.value = 0;
    state->shared_map.shared_heap_vend.value = 0;
    state->claim_tournament[0].root.insert_completion.value =
        SharedInsertCompletionInitialValue(0U);
    state->exec_fatal.state = 0;
    state->exec_cells[0].control.state = 0;

    const plan::RuntimePlanView view{
        &state->runtime_plan_control, cells, 1U,
    };
    const plan::RuntimeTaskPlanSpec spec{
        /*task_id=*/0U,
        /*function_id=*/7U,
        /*tensor_count=*/0U,
        /*scalar_count=*/0U,
        /*explicit_dep_count=*/0U,
        /*output_count=*/0U,
        plan::EngineClass::Aic,
        /*adapter_flags=*/0xA5U,
        /*core_num=*/1,
        /*require_sync_start=*/0U,
        /*reserved=*/0U,
        /*adapter_data=*/0U,
        /*tensor_reference_mask=*/0U,
    };
    const EmptyPlanSource source{};
    Expect(
        plan::PublishRuntimeTaskPlan<CpuOps>(view, spec, source) ==
            plan::PlanPublishResult::Published &&
            plan::AdvancePlannedFrontier<CpuOps>(view, 0U) &&
            plan::CloseRuntimePlan<CpuOps>(view, 1U),
        "failed to publish the exec-owner Plan"
    );

    Runtime runtime{
        state, 1U, UINT32_MAX, kWatchdogTicks, RoutePolicy{},
    };
    Expect(runtime.BindTask(0U), "failed to bind exec-owner task");
    simt::SimtTaskBuildScratch scratch{};
    Expect(
        simt::BuildCanonicalPlanTask<CpuOps>(
            runtime, view, 0U, simt::SimtBuildOwner(3U), scratch
        ) == simt::SimtTaskBuildStatus::Published,
        "real ExecCell Build failed"
    );
    const simt::exec::DecodedExecState decoded =
        simt::exec::DecodeExecState(
            state->exec_cells[0].control.state
        );
    Expect(decoded.valid &&
               decoded.phase == simt::exec::ExecPhase::Built &&
               decoded.build_owner == 3U &&
               decoded.build_owner < kWorkers &&
               decoded.execute_owner ==
                   simt::exec::kExecUnboundOwner &&
               decoded.engine_class ==
                   simt::exec::ExecEngineClass::Aic &&
               decoded.task_id == 0U,
           "SIMT owner did not reach the real ExecCell BUILT state");

    std::free(cells);
    UnmapState(state);
}

void TestFreshOutputAndSymbolHistoryOnRealState()
{
    SchedulerState *state = MapState();
    Expect(state != nullptr, "failed to map output/history fixture");
    plan::RuntimeTaskPlanCell *cells = AllocatePlanCells(3U);
    Expect(cells != nullptr, "failed to allocate output/history Plan");

    state->heap_window = static_cast<int32_t>(kHeapWindow);
    state->heap_base = UINT64_C(0x180000000);
    state->heap_size = UINT64_C(0x8000);
    state->runtime_plan_control.planned_frontier.value = 0;
    state->runtime_plan_control.closed_task_count.value =
        plan::kPlanOpenTaskCount;
    state->runtime_plan_control.build_next.value = 0;
    state->runtime_plan_control.build_workers_done.value = 0;
    state->runtime_plan_control.build_release.value =
        plan::kBuildReleasePending;
    state->runtime_plan_control.fatal.value = 0;
    state->fatal.value = 0;
    state->shared_map.shared_heap_vend.value = 0;
    for (uint32_t shard = 0U; shard < kSharedHeapShards; ++shard) {
        state->shared_map.shared_heap_cursor[shard].value = 0;
    }
    for (uint32_t task = 0U; task < 3U; ++task) {
        state->claim_tournament[task]
            .root.insert_completion.value =
                SharedInsertCompletionInitialValue(task);
        state->tasks[task].vend = 0U;
        state->tasks[task].flag = 0;
    }
    state->shared_map.shared_outputs[0].published[0].value = -1;
    state->shared_map.shared_outputs[0].last_writer[0].value = -1;

    const plan::RuntimePlanView view{
        &state->runtime_plan_control, cells, 3U,
    };
    const SingleTensorPlanSource output_source = MakeOutputSource();
    const SingleTensorPlanSource input_source =
        MakeReferenceSource(plan::TensorTag::Input);
    const SingleTensorPlanSource inout_source =
        MakeReferenceSource(plan::TensorTag::Inout);
    const plan::RuntimeTaskPlanSpec output_spec{
        0U, plan::kInvalidFunctionId,
        1U, 0U, 0U, 1U,
        plan::EngineClass::MetadataOnly,
        0xA5U, 1, 0U, 0U, 0U, 0U,
    };
    const plan::RuntimeTaskPlanSpec input_spec{
        1U, plan::kInvalidFunctionId,
        1U, 0U, 0U, 0U,
        plan::EngineClass::MetadataOnly,
        0xA5U, 1, 0U, 0U, 1U, 1U,
    };
    const plan::RuntimeTaskPlanSpec inout_spec{
        2U, plan::kInvalidFunctionId,
        1U, 0U, 0U, 0U,
        plan::EngineClass::MetadataOnly,
        0xA5U, 1, 0U, 0U, 2U, 1U,
    };
    Expect(
        plan::PublishRuntimeTaskPlan<CpuOps>(
            view, output_spec, output_source
        ) == plan::PlanPublishResult::Published &&
            plan::AdvancePlannedFrontier<CpuOps>(view, 0U) &&
            plan::PublishRuntimeTaskPlan<CpuOps>(
                view, input_spec, input_source
            ) == plan::PlanPublishResult::Published &&
            plan::AdvancePlannedFrontier<CpuOps>(view, 1U) &&
            plan::PublishRuntimeTaskPlan<CpuOps>(
                view, inout_spec, inout_source
            ) == plan::PlanPublishResult::Published &&
            plan::AdvancePlannedFrontier<CpuOps>(view, 2U) &&
            plan::CloseRuntimePlan<CpuOps>(view, 3U),
        "failed to publish output/reference Plan chain"
    );

    Runtime runtime{
        state, 3U, UINT32_MAX, kWatchdogTicks, RoutePolicy{},
    };
    for (uint32_t task = 0U; task < 3U; ++task) {
        Expect(runtime.BindTask(task), "failed to bind output/history task");
        simt::SimtTaskBuildScratch scratch{};
        Expect(
            simt::BuildCanonicalPlanTask<CpuOps>(
                runtime, view, task,
                simt::SimtBuildOwner(task % simt::kBuilderLeaders),
                scratch
            ) == simt::SimtTaskBuildStatus::Published,
            "output/reference task failed on real state"
        );
    }

    const TensorDesc &descriptor =
        state->shared_map.shared_outputs[0].tensors[0];
    Expect(
        state->shared_map.shared_outputs[0].published[0].value == 0 &&
            descriptor.buffer_addr == state->heap_base &&
            descriptor.buffer_size == 64U &&
            descriptor.owner_task_id == 0U &&
            descriptor.ndims == 1U &&
            descriptor.shapes[0] == 16U &&
            descriptor.strides[0] == 1U,
        "fresh descriptor was not published into SharedOutputCell"
    );
    const SharedWriterHistoryCell &history =
        state->shared_map.writer_history[2];
    Expect(
        state->shared_map.shared_outputs[0].last_writer[0].value == 2 &&
            history.magic == kSharedWriterHistoryMagic &&
            history.writer_task == 2 && history.count == 1U &&
            history.reserved == 0U &&
            history.entries[0].symbol_key == 0U &&
            history.entries[0].previous_writer == 0,
        "symbol writer history did not publish the future-writer chain"
    );
    const plan::RuntimeOutputReferenceWire reference{
        0, 0, 0U, 0U, 0U, 0U,
    };
    int32_t resolved = -1;
    Expect(
        simt::ResolveSymbolWriterBefore<CpuOps>(
            runtime, reference, /*reader_task=*/1U, resolved
        ) && resolved == 0,
        "reader did not traverse future writer 2 back to producer 0"
    );

    Expect(runtime.OutputPublished(3U, 0U) == nullptr &&
               runtime.OutputPublished(0U, 8U) == nullptr &&
               runtime.OutputLastWriter(3U, 0U) == nullptr &&
               runtime.OutputDescriptorWords(0U, 8U) == nullptr &&
               runtime.InsertCompletion(3U) == nullptr &&
               runtime.WriterHistory(3U) == nullptr &&
               runtime.GlobalFatal() == &state->fatal.value &&
               &runtime.ExecFatal() == &state->exec_fatal,
           "real-state accessor bounds or addresses are wrong");
    Expect(runtime.PublishBuildFatal(2U, 7U) &&
               state->fatal.value == 1 &&
               state->runtime_plan_control.fatal.value == 1,
           "fatal did not publish to both real atomic controls");

    std::free(cells);
    UnmapState(state);
}

void TestHeapRejectsNullBaseBeforeAtomics()
{
    SchedulerState *state = MapState();
    Expect(state != nullptr, "failed to map null-heap fixture");
    state->heap_window = static_cast<int32_t>(kHeapWindow);
    state->heap_base = 0U;
    state->heap_size = UINT64_C(0x8000);
    state->shared_map.shared_heap_vend.value = 0;
    state->shared_map.shared_heap_cursor[0].value = 0;
    Runtime runtime{
        state, 1U, UINT32_MAX, kWatchdogTicks, RoutePolicy{},
    };
    Expect(runtime.BindTask(0U), "failed to bind null-heap task");
    uint64_t first = UINT64_MAX;
    uint64_t vend = UINT64_MAX;
    Expect(!runtime.ReserveOutputHeap(0U, 1024U, first, vend) &&
               first == 0U && vend == 0U &&
               state->shared_map.shared_heap_cursor[0].value == 0 &&
               state->shared_map.shared_heap_vend.value == 0,
           "null heap base consumed a shard/vend FetchAdd");
    UnmapState(state);
}

}  // namespace

int main()
{
    TestMetadataBuildOnRealState();
    TestRealRingAndEightShardHeap();
    TestExecBuildOwnerOnRealState();
    TestFreshOutputAndSymbolHistoryOnRealState();
    TestHeapRejectsNullBaseBeforeAtomics();
    std::puts("[PASS] SIMT narrow builder maps the real Scalar state");
    return 0;
}
