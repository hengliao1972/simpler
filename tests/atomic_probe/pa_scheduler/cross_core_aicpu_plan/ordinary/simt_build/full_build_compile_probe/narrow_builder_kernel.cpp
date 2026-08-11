/*
 * Compile/link gate for the complete narrow canonical-Plan SIMT Build leaf.
 * The state below is a structural compile harness, not a second request ABI:
 * task identity and all tensor/scalar/dependency data still come from Plan v2.
 */

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include <cstddef>
#include <cstdint>

#define PA_DEVICE \
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline
#define PA_GM __gm__

#include "../common/simt_plan_task_builder.h"

namespace {

namespace simt = pa_scheduler::aicpu_plan_simt;
namespace plan = pa_scheduler::aicpu_plan;
namespace exec = pa_scheduler::cross_core;

constexpr uint32_t kProbeBuckets = 16U;
constexpr uint32_t kProbeBucketCapacity = 8U;

struct alignas(64) NarrowAtomicLine {
    volatile int64_t value;
    uint8_t padding[56];
};

struct alignas(64) NarrowOrdinaryPayload {
    simt::SimtWriterRegion value;
    uint8_t padding[32];
};

struct alignas(64) NarrowOrdinarySlot {
    NarrowOrdinaryPayload payload;
    NarrowAtomicLine seq;
};

struct alignas(64) NarrowOrdinaryBucket {
    NarrowAtomicLine head;
    NarrowAtomicLine tail;
};

struct alignas(64) NarrowMetadataCell {
    uint64_t vend;
    uint8_t payload_padding[56];
    NarrowAtomicLine completion;
};

struct SimtOps {
    static constexpr bool kAtomicReturnReadyObserved = true;

    PA_DEVICE static int64_t LoadControl(
        __gm__ volatile int64_t *address
    ) {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), int64_t{0}
        );
    }

    PA_DEVICE static int64_t Load(
        __gm__ volatile int64_t *address
    ) {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), int64_t{0}
        );
    }

    PA_DEVICE static int64_t FetchAdd(
        __gm__ volatile int64_t *address, int64_t value
    ) {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static int64_t CompareExchange(
        __gm__ volatile int64_t *address,
        int64_t expected, int64_t desired
    ) {
        return asc_atomic_cas(
            const_cast<__gm__ int64_t *>(address), expected, desired
        );
    }

    PA_DEVICE static void StorePayloadWord(
        __gm__ volatile uint64_t *address, uint64_t value
    ) {
        asc_stcg(const_cast<__gm__ uint64_t *>(address), value);
    }

    PA_DEVICE static void StoreBarrier() {
        asc_threadfence();
    }

    PA_DEVICE static uint64_t Now() {
        return clock();
    }

    PA_DEVICE static void SpinHint() {}

    PA_DEVICE static void PreloadBuildDestination(
        __gm__ void *, uint64_t
    ) {}

    PA_DEVICE static void BeforeBuiltPublish(uint32_t) {}

    PA_DEVICE static void FlushRegion(
        __gm__ void *address, uint64_t bytes
    ) {
        // 所有 builder payload word 都由 asc_stcg bypass 写入。writer
        // publication 只需要 fence 排序；DCCI 是 reader invalidate，不能
        // 在这里对 writer cache line 执行。
        (void)address;
        (void)bytes;
        asc_threadfence();
    }

    PA_DEVICE static void FlushRegion(
        __gm__ volatile uint64_t *address, uint64_t bytes
    ) {
        FlushRegion(
            static_cast<__gm__ void *>(
                const_cast<__gm__ uint64_t *>(address)
            ), bytes
        );
    }

    PA_DEVICE static void InvalidateRegion(
        __gm__ const void *address, uint64_t bytes
    ) {
        if (bytes == 0U) return;
        __gm__ uint8_t *start = const_cast<__gm__ uint8_t *>(
            reinterpret_cast<__gm__ const uint8_t *>(address)
        );
        const uint64_t lines = (bytes + 63U) / 64U;
        for (uint64_t line = 0U; line < lines; ++line) {
            asc_dcci_single(static_cast<__gm__ void *>(
                start + line * 64U
            ));
        }
        asc_threadfence();
    }

    PA_DEVICE static void InvalidateRegion(
        __gm__ void *address, uint64_t bytes
    ) {
        InvalidateRegion(
            static_cast<__gm__ const void *>(address), bytes
        );
    }
};

struct NarrowRuntime {
    __gm__ volatile int64_t *fatal;
    __gm__ volatile int64_t *output_published;
    __gm__ volatile int64_t *output_last_writer;
    __gm__ volatile uint64_t *output_descriptors;
    __gm__ volatile int64_t *insert_completion;
    __gm__ simt::SimtWriterHistoryCell *writer_history;
    __gm__ NarrowOrdinaryBucket *ordinary_buckets;
    __gm__ NarrowOrdinarySlot *ordinary_slots;
    __gm__ volatile int64_t *heap_cursor;
    __gm__ volatile int64_t *heap_vend;
    __gm__ NarrowMetadataCell *metadata_cells;
    __gm__ exec::SharedExecCell *exec_cells;
    __gm__ exec::SharedExecFatalControl *exec_fatal;
    uint64_t heap_base;
    uint64_t heap_size;
    uint32_t task_capacity;
    uint32_t heap_window;

    PA_DEVICE uint32_t OutputIndex(
        uint32_t task_id, uint32_t slot
    ) const {
        return task_id * plan::kMaxRuntimeOutputsPerTask + slot;
    }

    PA_DEVICE __gm__ volatile int64_t *OutputPublished(
        uint32_t task_id, uint32_t slot
    ) const {
        return &output_published[OutputIndex(task_id, slot)];
    }

    PA_DEVICE __gm__ volatile int64_t *OutputLastWriter(
        uint32_t task_id, uint32_t slot
    ) const {
        return &output_last_writer[OutputIndex(task_id, slot)];
    }

    PA_DEVICE __gm__ volatile uint64_t *OutputDescriptorWords(
        uint32_t task_id, uint32_t slot
    ) const {
        return &output_descriptors[
            static_cast<uint64_t>(OutputIndex(task_id, slot)) *
                plan::kTensorDescWords
        ];
    }

    PA_DEVICE __gm__ volatile int64_t *InsertCompletion(
        uint32_t task_id
    ) const {
        return &insert_completion[task_id];
    }

    PA_DEVICE __gm__ volatile int64_t *GlobalFatal() const {
        return fatal;
    }

    PA_DEVICE uint64_t WatchdogTicks() const {
        return uint64_t{1} << 40U;
    }

    PA_DEVICE uint32_t TaskCapacity() const {
        return task_capacity;
    }

    PA_DEVICE bool FaninLowerBound(
        uint32_t reader_task, int32_t &lower_bound
    ) const {
        if (reader_task > static_cast<uint32_t>(INT32_MAX) ||
            heap_window > static_cast<uint32_t>(INT32_MAX)) {
            return false;
        }
        lower_bound = static_cast<int32_t>(
            reader_task > heap_window
                ? reader_task - heap_window
                : 0U
        );
        return true;
    }

    PA_DEVICE __gm__ simt::SimtWriterHistoryCell *WriterHistory(
        uint32_t task_id
    ) const {
        return task_id < task_capacity
            ? &writer_history[task_id]
            : nullptr;
    }

    PA_DEVICE bool PublishBuildFatal(
        uint32_t task_id, uint32_t reason
    ) const {
        const uint64_t encoded =
            static_cast<uint64_t>(reason + 1U) |
            (static_cast<uint64_t>(task_id) << 16U);
        return SimtOps::CompareExchange(
                   fatal, 0, static_cast<int64_t>(encoded)
               ) == 0;
    }

    PA_DEVICE bool ValidateAdapterRoute(
        uint8_t adapter_flags, uint16_t adapter_data,
        uint32_t function_id, plan::EngineClass engine
    ) const {
        // Probe 只证明 adapter 入口消费显式 header provenance；它不能
        // 从 task-id 算术或固定 PA offset 重建路由。
        const bool metadata = engine == plan::EngineClass::MetadataOnly;
        return adapter_flags != 0U && adapter_data != UINT16_MAX &&
               (metadata
                    ? function_id == plan::kInvalidFunctionId
                    : function_id != plan::kInvalidFunctionId);
    }

    PA_DEVICE bool ReserveOutputHeap(
        uint32_t, uint64_t total, uint64_t &first_address,
        uint64_t &aggregate_vend
    ) const {
        // 在任何返回型 FetchAdd 前完成容量与地址区间检查。这样 total>
        // heap_size 不会让后面的 heap_size-total 下溢，base+size 也不会
        // 在即使零输出的异常配置上静默回绕。
        if (total > heap_size || heap_base > UINT64_MAX - heap_size) {
            return false;
        }
        if (total == 0U) {
            const int64_t observed = SimtOps::Load(heap_vend);
            if (observed < 0) return false;
            aggregate_vend = static_cast<uint64_t>(observed);
            first_address = 0U;
            return true;
        }
        if (total > static_cast<uint64_t>(INT64_MAX)) return false;
        const int64_t cursor = SimtOps::FetchAdd(
            heap_cursor, static_cast<int64_t>(total)
        );
        const int64_t vend = SimtOps::FetchAdd(
            heap_vend, static_cast<int64_t>(total)
        );
        // 并发 reservation 仍可能让 FetchAdd 返回越界旧值。此时控制字
        // 已经推进，不能用 Exchange 回滚并覆盖其他 leader 的合法份额；
        // BuildCanonicalPlanTask 会发布 terminal fatal，Host 在下一轮启动
        // 前重置整份 sidecar/heap control。
        if (cursor < 0 || vend < 0 ||
            static_cast<uint64_t>(cursor) > heap_size - total ||
            static_cast<uint64_t>(vend) > heap_size - total ||
            heap_base > UINT64_MAX - static_cast<uint64_t>(cursor)) {
            return false;
        }
        first_address = heap_base + static_cast<uint64_t>(cursor);
        aggregate_vend = static_cast<uint64_t>(vend) + total;
        return true;
    }

    PA_DEVICE uint32_t OrdinaryBucket(uint64_t address) const {
        address *= 0x9E3779B97F4A7C15ULL;
        return static_cast<uint32_t>(address >> 60U) &
               (kProbeBuckets - 1U);
    }

    PA_DEVICE bool CheckOrdinaryAppend(
        const simt::SimtWriterRegion *, const uint16_t *buckets,
        const uint8_t *ordinals, uint32_t count,
        uint32_t
    ) const {
        for (uint32_t entry = 0U; entry < count; ++entry) {
            const uint32_t bucket = buckets[entry];
            if (bucket >= kProbeBuckets) return false;
            const int64_t head = SimtOps::Load(
                &ordinary_buckets[bucket].head.value
            );
            const int64_t tail = SimtOps::Load(
                &ordinary_buckets[bucket].tail.value
            );
            if (head < 0 || tail < head ||
                static_cast<uint64_t>(tail - head) +
                    ordinals[entry] >= kProbeBucketCapacity) {
                return false;
            }
        }
        return true;
    }

    PA_DEVICE bool AppendOrdinary(
        const simt::SimtWriterRegion *entries,
        const uint16_t *buckets, uint32_t count,
        uint32_t task_id
    ) const {
        for (uint32_t entry = 0U; entry < count; ++entry) {
            const uint32_t bucket = buckets[entry];
            const int64_t cursor = SimtOps::FetchAdd(
                &ordinary_buckets[bucket].tail.value, 1
            );
            if (cursor < 0) return false;
            __gm__ NarrowOrdinarySlot &slot = ordinary_slots[
                bucket * kProbeBucketCapacity +
                (static_cast<uint32_t>(cursor) &
                 (kProbeBucketCapacity - 1U))
            ];
            __gm__ volatile uint64_t *payload_words =
                reinterpret_cast<__gm__ volatile uint64_t *>(
                    &slot.payload
                );
            SimtOps::StorePayloadWord(
                &payload_words[0], entries[entry].buffer_addr
            );
            SimtOps::StorePayloadWord(
                &payload_words[1], entries[entry].lo
            );
            SimtOps::StorePayloadWord(
                &payload_words[2], entries[entry].hi
            );
            SimtOps::StorePayloadWord(
                &payload_words[3],
                static_cast<uint64_t>(static_cast<uint32_t>(
                    entries[entry].producer
                ))
            );
            SimtOps::StoreBarrier();
            if (SimtOps::CompareExchange(
                    &slot.seq.value, -1,
                    static_cast<int64_t>(task_id)
                ) != -1) {
                return false;
            }
        }
        return true;
    }

    PA_DEVICE bool LookupOrdinary(
        const simt::SimtCanonicalTensorDesc &tensor,
        uint32_t reader_task, int32_t &producer
    ) const {
        simt::SimtWriterRegion query{};
        if (!simt::SimtMakeWriterRegion(
                tensor, reader_task, query
            )) {
            return false;
        }
        producer = -1;
        const uint32_t bucket = OrdinaryBucket(tensor.buffer_addr);
        const int64_t head = SimtOps::Load(
            &ordinary_buckets[bucket].head.value
        );
        const int64_t tail = SimtOps::Load(
            &ordinary_buckets[bucket].tail.value
        );
        if (head < 0 || tail < head ||
            tail - head > kProbeBucketCapacity) {
            return false;
        }
        for (int64_t cursor = head; cursor < tail; ++cursor) {
            __gm__ NarrowOrdinarySlot &slot = ordinary_slots[
                bucket * kProbeBucketCapacity +
                (static_cast<uint32_t>(cursor) &
                 (kProbeBucketCapacity - 1U))
            ];
            const int64_t sequence = SimtOps::Load(&slot.seq.value);
            if (sequence < 0 ||
                sequence >= static_cast<int64_t>(reader_task)) {
                continue;
            }
            SimtOps::InvalidateRegion(
                &slot.payload, sizeof(slot.payload)
            );
            const simt::SimtWriterRegion value{
                slot.payload.value.buffer_addr,
                slot.payload.value.lo,
                slot.payload.value.hi,
                slot.payload.value.producer,
                slot.payload.value.reserved,
            };
            if (value.reserved != 0U ||
                value.producer != sequence ||
                value.buffer_addr != query.buffer_addr) {
                return false;
            }
            if (value.lo < query.hi && query.lo < value.hi &&
                value.producer > producer) {
                producer = value.producer;
            }
        }
        return true;
    }

    PA_DEVICE bool PublishMetadataCompletion(
        uint32_t task_id, uint64_t vend
    ) const {
        __gm__ NarrowMetadataCell &cell = metadata_cells[task_id];
        SimtOps::StorePayloadWord(
            reinterpret_cast<__gm__ volatile uint64_t *>(&cell.vend),
            vend
        );
        SimtOps::StoreBarrier();
        return SimtOps::CompareExchange(
                   &cell.completion.value, -1,
                   static_cast<int64_t>(task_id)
               ) == -1;
    }

    PA_DEVICE __gm__ exec::SharedExecCell &ExecCell(
        uint32_t task_id
    ) const {
        return exec_cells[task_id];
    }

    PA_DEVICE __gm__ exec::SharedExecFatalControl &ExecFatal() const {
        return *exec_fatal;
    }
};

static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void
BuildCanonicalPlanTaskNarrowVf(
    __gm__ plan::RuntimePlanControl *plan_control,
    __gm__ plan::RuntimeTaskPlanCell *plan_cells,
    uint32_t task_capacity,
    uint32_t first_task_id,
    __gm__ volatile int64_t *fatal,
    __gm__ volatile int64_t *output_published,
    __gm__ volatile int64_t *output_last_writer,
    __gm__ volatile uint64_t *output_descriptors,
    __gm__ volatile int64_t *insert_completion,
    __gm__ simt::SimtWriterHistoryCell *writer_history,
    __gm__ NarrowOrdinaryBucket *ordinary_buckets,
    __gm__ NarrowOrdinarySlot *ordinary_slots,
    __gm__ volatile int64_t *heap_cursor,
    __gm__ volatile int64_t *heap_vend,
    __gm__ NarrowMetadataCell *metadata_cells,
    __gm__ exec::SharedExecCell *exec_cells,
    __gm__ exec::SharedExecFatalControl *exec_fatal,
    uint64_t heap_base,
    uint64_t heap_size,
    __gm__ uint64_t *reports
)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (!simt::IsBuilderLeader(thread)) return;
    const uint32_t leader = simt::BuilderLeaderId(thread);
    const uint32_t task_id = first_task_id + leader;
    if (task_id >= task_capacity) return;

    NarrowRuntime runtime{
        fatal,
        output_published,
        output_last_writer,
        output_descriptors,
        insert_completion,
        writer_history,
        ordinary_buckets,
        ordinary_slots,
        heap_cursor,
        heap_vend,
        metadata_cells,
        exec_cells,
        exec_fatal,
        heap_base,
        heap_size,
        task_capacity,
        task_capacity,
    };
    const plan::RuntimePlanView view{
        plan_control, plan_cells, task_capacity,
    };
    simt::SimtTaskBuildScratch scratch{};
    const simt::SimtTaskBuildStatus status =
        simt::BuildCanonicalPlanTask<SimtOps>(
            runtime, view, task_id, leader, scratch
        );
    asc_stcg(
        reports + leader,
        static_cast<uint64_t>(status) |
            (static_cast<uint64_t>(scratch.ordinary_count) << 8U) |
            (static_cast<uint64_t>(scratch.symbol_count) << 16U) |
            (static_cast<uint64_t>(scratch.fanin_count) << 24U)
    );
}

}  // namespace

// Production 采用双 TU 身份隔离：本 TU 的 Plan/shared-exec helper 只以
// simt_callee 身份实例化；VF join 后跳转到另一 TU 的 Scalar continuation。
// 不能在现有 Scalar scheduler TU 中再次 include 本窄头。
extern "C" __aicore__ __attribute__((noinline, visibility("hidden")))
void aicpu_plan_narrow_scalar_continuation(
    __gm__ uint64_t *reports, uint32_t task_count
);

extern "C" __global__ __aicore__ void
aicpu_plan_narrow_simt_builder_compile_0_mix_aiv(
    __gm__ pa_scheduler::aicpu_plan::RuntimePlanControl *plan_control,
    __gm__ pa_scheduler::aicpu_plan::RuntimeTaskPlanCell *plan_cells,
    uint32_t task_capacity,
    uint32_t first_task_id,
    __gm__ volatile int64_t *fatal,
    __gm__ volatile int64_t *output_published,
    __gm__ volatile int64_t *output_last_writer,
    __gm__ volatile uint64_t *output_descriptors,
    __gm__ volatile int64_t *insert_completion,
    __gm__ pa_scheduler::aicpu_plan_simt::SimtWriterHistoryCell *writer_history,
    __gm__ NarrowOrdinaryBucket *ordinary_buckets,
    __gm__ NarrowOrdinarySlot *ordinary_slots,
    __gm__ volatile int64_t *heap_cursor,
    __gm__ volatile int64_t *heap_vend,
    __gm__ NarrowMetadataCell *metadata_cells,
    __gm__ pa_scheduler::cross_core::SharedExecCell *exec_cells,
    __gm__ pa_scheduler::cross_core::SharedExecFatalControl *exec_fatal,
    uint64_t heap_base,
    uint64_t heap_size,
    __gm__ uint64_t *reports
)
{
    cce::async_invoke<BuildCanonicalPlanTaskNarrowVf>(
        cce::dim3(128U), plan_control, plan_cells, task_capacity,
        first_task_id, fatal, output_published, output_last_writer,
        output_descriptors, insert_completion, writer_history,
        ordinary_buckets, ordinary_slots, heap_cursor, heap_vend,
        metadata_cells, exec_cells, exec_fatal, heap_base, heap_size,
        reports
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    aicpu_plan_narrow_scalar_continuation(reports, task_capacity);
}
