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

#ifndef PA_SCHEDULER_CROSS_CORE_AICPU_PLAN_PROTOCOL_H
#define PA_SCHEDULER_CROSS_CORE_AICPU_PLAN_PROTOCOL_H

#include <cstddef>
#include <cstdint>

#ifndef AICPU_PLAN_DEVICE
#ifdef PA_DEVICE
#define AICPU_PLAN_DEVICE PA_DEVICE
#else
#define AICPU_PLAN_DEVICE inline
#endif
#define AICPU_PLAN_UNDEFINE_DEVICE 1
#endif

#ifndef AICPU_PLAN_GM
#ifdef PA_GM
#define AICPU_PLAN_GM PA_GM
#else
#define AICPU_PLAN_GM
#endif
#define AICPU_PLAN_UNDEFINE_GM 1
#endif

namespace pa_scheduler::aicpu_plan {

constexpr uint32_t kRuntimePlanAbiVersion = 1U;
constexpr uint32_t kPlanCacheLineBytes = 64U;
constexpr uint32_t kAtomicIsolationBytes = 128U;
constexpr uint32_t kMaxRuntimeTasks = 32768U;
constexpr uint32_t kMaxTaskTensors = 32U;
constexpr uint32_t kMaxTaskScalars = 16U;
constexpr uint32_t kMaxExplicitDependencies = 16U;
constexpr uint32_t kTensorDescWords = 16U;
constexpr uint32_t kTensorCreateInfoWords = 8U;
constexpr uint32_t kOutputReferenceWords = 2U;
constexpr uint32_t kTensorCanonicalWords = kTensorDescWords;
constexpr uint32_t kMaxRuntimeOutputsPerTask = 8U;
constexpr uint32_t kPlanHeaderWords = kPlanCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kMaxPlanPayloadWords =
    kPlanHeaderWords + kMaxTaskTensors * kTensorDescWords +
    kMaxTaskScalars + kMaxExplicitDependencies;
constexpr uint32_t kMaxPlanPayloadBytes =
    kMaxPlanPayloadWords * sizeof(uint64_t);
constexpr uint32_t kMaxPlanPayloadLines =
    kMaxPlanPayloadBytes / kPlanCacheLineBytes;

static_assert(kMaxPlanPayloadBytes == 4416U, "runtime Plan payload bound changed");
static_assert(kMaxPlanPayloadLines == 69U, "runtime Plan payload line bound changed");

enum class EngineClass : uint8_t {
    MetadataOnly = 0,
    Aic = 1,
    Aiv = 2,
};

enum class TensorTag : uint8_t {
    Input = 0,
    Output = 1,
    Inout = 2,
    OutputExisting = 3,
    NoDependency = 4,
};

enum class PlanCellPhase : uint8_t {
    Empty = 0,
    Published = 1,
};

enum class PlanPublishResult : uint8_t {
    Published,
    InvalidInput,
    CellUnavailable,
};

enum class PlanAcquireResult : uint8_t {
    Acquired,
    NotPublished,
    InvalidControl,
    InvalidPayload,
    FatalObserved,
};

enum class BuildReservationStatus : uint8_t {
    Reserved,
    Closed,
    Fatal,
};

constexpr uint32_t kInvalidFunctionId = UINT32_MAX;
constexpr int64_t kPlanOpenTaskCount = -1;
constexpr int64_t kBuildReleasePending = -1;
constexpr int64_t kBuildReleaseFailed = -2;

struct RuntimeTaskPlanSpec {
    uint32_t task_id;
    uint32_t function_id;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t explicit_dep_count;
    uint16_t output_count;
    EngineClass engine_class;
    // 该字节由算子 adapter 定义和校验。公共协议只负责原样搬运，不能
    // 在这里固化 PA 的 kind/group/last-batch 等业务语义。
    uint8_t adapter_flags;
    int16_t core_num;
    uint8_t require_sync_start;
    uint8_t reserved;
    uint32_t tensor_reference_mask;
};

// 头部固定为一个 cache line。function_id 是稳定的通用函数身份；AICPU
// 不能把自己的函数指针或 callback 栈地址发布给 AIC/AIV。
struct RuntimeTaskPlanHeader {
    uint32_t task_id;
    uint32_t function_id;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t explicit_dep_count;
    uint16_t output_count;
    uint8_t engine_class;
    uint8_t adapter_flags;
    int16_t core_num;
    uint8_t require_sync_start;
    uint8_t reserved0;
    uint16_t reserved1;
    uint8_t tensor_tags[kMaxTaskTensors];
    uint32_t tensor_reference_mask;
    uint32_t abi_version;
};

static_assert(sizeof(RuntimeTaskPlanHeader) == kPlanCacheLineBytes, "runtime Plan header must be one line");
static_assert(offsetof(RuntimeTaskPlanHeader, tensor_tags) == 24U, "runtime Plan tag offset changed");
static_assert(offsetof(RuntimeTaskPlanHeader, tensor_reference_mask) == 56U, "runtime Plan ref-mask offset changed");

// 公共 Plan 不依赖 PA 的 FdwicOutputRef 类型，但 wire ABI 必须完整保留
// 16B 引用，不能退化成只带 producer/slot 的 8B 特例。字段顺序与真实
// runtime ABI 一致；编码和解码仍按两个 uint64_t 的明确 bit 位完成，避免
// 用 reinterpret_cast 读取对象表示而触发 strict-aliasing 问题。
struct RuntimeOutputReferenceWire {
    int32_t producer_task_id;
    int16_t output_slot;
    uint8_t flags;
    uint8_t view_ndims;
    uint32_t view_shape0;
    uint32_t view_offset0;
};

static_assert(sizeof(RuntimeOutputReferenceWire) == 16U, "runtime output-reference wire size changed");
static_assert(kOutputReferenceWords * sizeof(uint64_t) == sizeof(RuntimeOutputReferenceWire), "runtime output-reference word count changed");
static_assert(alignof(RuntimeOutputReferenceWire) == alignof(uint32_t), "runtime output-reference alignment changed");
static_assert(offsetof(RuntimeOutputReferenceWire, producer_task_id) == 0U, "runtime output-reference producer offset changed");
static_assert(offsetof(RuntimeOutputReferenceWire, output_slot) == 4U, "runtime output-reference slot offset changed");
static_assert(offsetof(RuntimeOutputReferenceWire, flags) == 6U, "runtime output-reference flags offset changed");
static_assert(offsetof(RuntimeOutputReferenceWire, view_ndims) == 7U, "runtime output-reference rank offset changed");
static_assert(offsetof(RuntimeOutputReferenceWire, view_shape0) == 8U, "runtime output-reference shape offset changed");
static_assert(offsetof(RuntimeOutputReferenceWire, view_offset0) == 12U, "runtime output-reference offset offset changed");

struct RuntimeTaskPlanLayout {
    uint32_t payload_words;
    uint32_t payload_bytes;
    uint32_t payload_lines;
    uint32_t scalar_word_offset;
    uint32_t explicit_dep_word_offset;
};

struct alignas(kPlanCacheLineBytes) RuntimeTaskPlanStorage {
    volatile uint64_t words[kMaxPlanPayloadWords];
};

// A5 返回型 atomic 的冲突范围按已有实测以 128B 隔离。padding 只能做
// 地址隔离，不能存 ordinary payload，也不能执行覆盖整块的 DCCI。
struct alignas(kAtomicIsolationBytes) PlanAtomicLine {
    volatile int64_t value;
    uint8_t isolation_padding[kAtomicIsolationBytes - sizeof(int64_t)];
};

struct alignas(kAtomicIsolationBytes) RuntimeTaskPlanCell {
    PlanAtomicLine control;
    RuntimeTaskPlanStorage payload;
    uint8_t tail_padding[kPlanCacheLineBytes];
};

static_assert(sizeof(PlanAtomicLine) == kAtomicIsolationBytes, "Plan atomic isolation changed");
static_assert(alignof(PlanAtomicLine) == kAtomicIsolationBytes, "Plan atomic base alignment changed");
static_assert(offsetof(PlanAtomicLine, value) == 0U, "Plan atomic value offset changed");
static_assert(sizeof(RuntimeTaskPlanStorage) == kMaxPlanPayloadBytes, "Plan storage size changed");
static_assert(offsetof(RuntimeTaskPlanCell, control) == 0U, "Plan cell control offset changed");
static_assert(offsetof(RuntimeTaskPlanCell, payload) == kAtomicIsolationBytes, "Plan payload offset changed");
static_assert(sizeof(RuntimeTaskPlanCell) == 4608U, "Plan cell stride changed");
static_assert(alignof(RuntimeTaskPlanCell) == kAtomicIsolationBytes, "Plan cell base alignment changed");
static_assert(sizeof(RuntimeTaskPlanCell) % kAtomicIsolationBytes == 0U, "Plan cell stride must preserve 128B alignment");

// planned_frontier 是连续已发布条目数，即首个尚未发布的 task_id；有效
// 前缀严格为 [0, planned_frontier)。closed_task_count=-1 表示仍开放。
struct alignas(kAtomicIsolationBytes) RuntimePlanControl {
    PlanAtomicLine planned_frontier;
    PlanAtomicLine closed_task_count;
    PlanAtomicLine build_next;
    PlanAtomicLine build_workers_done;
    PlanAtomicLine build_release;
    PlanAtomicLine fatal;
};

static_assert(sizeof(RuntimePlanControl) == 6U * kAtomicIsolationBytes, "RuntimePlanControl layout changed");
static_assert(alignof(RuntimePlanControl) == kAtomicIsolationBytes, "RuntimePlanControl base alignment changed");
static_assert(offsetof(RuntimePlanControl, planned_frontier) == 0U, "planned-frontier offset changed");
static_assert(offsetof(RuntimePlanControl, closed_task_count) == 128U, "closed-task offset changed");
static_assert(offsetof(RuntimePlanControl, build_next) == 256U, "build-next offset changed");
static_assert(offsetof(RuntimePlanControl, build_workers_done) == 384U, "build-done offset changed");
static_assert(offsetof(RuntimePlanControl, build_release) == 512U, "build-release offset changed");
static_assert(offsetof(RuntimePlanControl, fatal) == 640U, "fatal offset changed");

// Cell 数组按运行期容量单独分配，避免在 1GiB SchedulerState 中固定追加
// 32768×4608B。Host 只提供容量和裸 GM 地址，不填任何 task identity。
struct alignas(kAtomicIsolationBytes) RuntimePlanStorageRef {
    uint64_t cells_base;
    uint64_t cells_bytes;
    uint32_t capacity;
    uint32_t cell_bytes;
    uint32_t abi_version;
    uint32_t reserved0;
    // 显式占满 128B，不把跨处理器 wire ABI 的后一半留成不可校验的
    // C++ tail padding。
    uint64_t reserved[12];
};

static_assert(sizeof(RuntimePlanStorageRef) == kAtomicIsolationBytes, "Plan storage ref must occupy one isolated line");
static_assert(alignof(RuntimePlanStorageRef) == kAtomicIsolationBytes, "Plan storage ref base alignment changed");
static_assert(offsetof(RuntimePlanStorageRef, cells_base) == 0U, "Plan storage-ref base offset changed");
static_assert(offsetof(RuntimePlanStorageRef, reserved) == 32U, "Plan storage-ref reserved offset changed");

struct RuntimePlanView {
    AICPU_PLAN_GM RuntimePlanControl *control;
    AICPU_PLAN_GM RuntimeTaskPlanCell *cells;
    uint32_t capacity;
};

struct BuildReservation {
    BuildReservationStatus status;
    uint32_t task_id;
};

enum class BuildArrivalStatus : uint8_t {
    Arrived,
    Last,
    Invalid,
    Fatal,
};

struct DecodedPlanCellControl {
    PlanCellPhase phase;
    uint32_t payload_lines;
    uint32_t task_id;
    bool valid;
};

constexpr uint64_t kPlanPhaseMask = 0x3ULL;
constexpr uint64_t kPlanPayloadLinesShift = 2U;
constexpr uint64_t kPlanPayloadLinesMask = 0x7FULL;
constexpr uint64_t kPlanTaskIdShift = 9U;
constexpr uint64_t kPlanTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kPlanKnownMask =
    kPlanPhaseMask |
    (kPlanPayloadLinesMask << kPlanPayloadLinesShift) |
    (kPlanTaskIdMask << kPlanTaskIdShift);

AICPU_PLAN_DEVICE uint64_t EncodePlanCellControl(
    PlanCellPhase phase, uint32_t payload_lines, uint32_t task_id
)
{
    return static_cast<uint64_t>(phase) |
           (static_cast<uint64_t>(payload_lines) << kPlanPayloadLinesShift) |
           (static_cast<uint64_t>(task_id) << kPlanTaskIdShift);
}

AICPU_PLAN_DEVICE DecodedPlanCellControl DecodePlanCellControl(
    int64_t raw_state
)
{
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedPlanCellControl decoded{
        static_cast<PlanCellPhase>(raw & kPlanPhaseMask),
        static_cast<uint32_t>((raw >> kPlanPayloadLinesShift) & kPlanPayloadLinesMask),
        static_cast<uint32_t>((raw >> kPlanTaskIdShift) & kPlanTaskIdMask),
        false,
    };
    if ((raw & ~kPlanKnownMask) != 0U) return decoded;
    if (decoded.phase == PlanCellPhase::Empty) {
        decoded.valid = raw == 0U;
    } else if (decoded.phase == PlanCellPhase::Published) {
        decoded.valid = decoded.payload_lines >= 1U &&
                        decoded.payload_lines <= kMaxPlanPayloadLines;
    }
    return decoded;
}

AICPU_PLAN_DEVICE bool RuntimePlanStorageRefValid(
    AICPU_PLAN_GM const RuntimePlanStorageRef &storage
)
{
    constexpr uint64_t kMaxAddress = ~uint64_t{0};
    const uint64_t cell_bytes = sizeof(RuntimeTaskPlanCell);
    if (storage.cells_base == 0U || storage.capacity == 0U ||
        storage.capacity > kMaxRuntimeTasks ||
        storage.cell_bytes != sizeof(RuntimeTaskPlanCell) ||
        storage.abi_version != kRuntimePlanAbiVersion ||
        storage.reserved0 != 0U ||
        (storage.cells_base % kAtomicIsolationBytes) != 0U ||
        static_cast<uint64_t>(storage.capacity) > kMaxAddress / cell_bytes) {
        return false;
    }
    const uint64_t required_bytes =
        static_cast<uint64_t>(storage.capacity) * cell_bytes;
    if (storage.cells_bytes != required_bytes ||
        (storage.cells_bytes % kAtomicIsolationBytes) != 0U ||
        storage.cells_base > kMaxAddress - storage.cells_bytes) {
        return false;
    }
    for (uint32_t index = 0U; index < 12U; ++index) {
        if (storage.reserved[index] != 0U) return false;
    }
    return true;
}

// StorageRef 是 Host/AICPU 与 AICore 之间唯一携带 cell 基址的 ABI。
// 上层不得各自复制 uint64_t->GM pointer 的转换逻辑；只有完整校验容量、
// stride、ABI、128B 对齐以及地址区间无溢出后，才构造可访问的 view。
AICPU_PLAN_DEVICE bool MakeRuntimePlanView(
    AICPU_PLAN_GM RuntimePlanControl *control,
    AICPU_PLAN_GM const RuntimePlanStorageRef &storage,
    RuntimePlanView &view
)
{
    view.control = nullptr;
    view.cells = nullptr;
    view.capacity = 0U;
    if (control == nullptr ||
        (reinterpret_cast<uintptr_t>(control) % kAtomicIsolationBytes) != 0U ||
        !RuntimePlanStorageRefValid(storage)) {
        return false;
    }
    view.control = control;
    view.cells = reinterpret_cast<AICPU_PLAN_GM RuntimeTaskPlanCell *>(
        static_cast<uintptr_t>(storage.cells_base)
    );
    view.capacity = storage.capacity;
    return true;
}

AICPU_PLAN_DEVICE bool TensorTagValid(TensorTag tag)
{
    return tag == TensorTag::Input || tag == TensorTag::Output ||
           tag == TensorTag::Inout || tag == TensorTag::OutputExisting ||
           tag == TensorTag::NoDependency;
}

AICPU_PLAN_DEVICE uint32_t TensorMaskForCount(uint32_t tensor_count)
{
    return tensor_count >= 32U ? UINT32_MAX : ((uint32_t{1} << tensor_count) - 1U);
}

AICPU_PLAN_DEVICE uint32_t TensorStorageWords(
    TensorTag, bool
)
{
    // canonical payload 为每个 tensor 固定 16 words。这样任意 tensor 的
    // 地址只由 index 决定，AICPU/Scalar/SIMT 不会各自维护一套变长 offset。
    return kTensorCanonicalWords;
}

AICPU_PLAN_DEVICE uint32_t TensorMeaningfulWords(
    TensorTag tag, bool reference
)
{
    if (reference) return kOutputReferenceWords;
    return tag == TensorTag::Output
        ? kTensorCreateInfoWords
        : kTensorDescWords;
}

AICPU_PLAN_DEVICE int32_t DecodeRuntimeWireInt32(uint32_t bits)
{
    if (bits <= 0x7FFFFFFFU) return static_cast<int32_t>(bits);
    return static_cast<int32_t>(
        -1 - static_cast<int64_t>(~bits)
    );
}

AICPU_PLAN_DEVICE int16_t DecodeRuntimeWireInt16(uint16_t bits)
{
    if (bits <= 0x7FFFU) return static_cast<int16_t>(bits);
    return static_cast<int16_t>(
        -1 - static_cast<int32_t>(static_cast<uint16_t>(~bits))
    );
}

AICPU_PLAN_DEVICE RuntimeOutputReferenceWire DecodeRuntimeOutputReferenceWire(
    uint64_t first_word, uint64_t second_word
)
{
    return RuntimeOutputReferenceWire{
        DecodeRuntimeWireInt32(static_cast<uint32_t>(first_word)),
        DecodeRuntimeWireInt16(
            static_cast<uint16_t>(first_word >> 32U)
        ),
        static_cast<uint8_t>(first_word >> 48U),
        static_cast<uint8_t>(first_word >> 56U),
        static_cast<uint32_t>(second_word),
        static_cast<uint32_t>(second_word >> 32U),
    };
}

AICPU_PLAN_DEVICE uint64_t RuntimeOutputReferenceWireWord(
    const RuntimeOutputReferenceWire &reference, uint32_t word
)
{
    if (word == 0U) {
        return static_cast<uint64_t>(
                   static_cast<uint32_t>(reference.producer_task_id)
               ) |
               (static_cast<uint64_t>(
                    static_cast<uint16_t>(reference.output_slot)
                ) << 32U) |
               (static_cast<uint64_t>(reference.flags) << 48U) |
               (static_cast<uint64_t>(reference.view_ndims) << 56U);
    }
    if (word == 1U) {
        return static_cast<uint64_t>(reference.view_shape0) |
               (static_cast<uint64_t>(reference.view_offset0) << 32U);
    }
    return 0U;
}

AICPU_PLAN_DEVICE bool RuntimeOutputReferenceWireValid(
    const RuntimeOutputReferenceWire &reference, uint32_t consumer_task_id
)
{
    if (reference.producer_task_id < 0 ||
        static_cast<uint32_t>(reference.producer_task_id) >=
            consumer_task_id ||
        reference.output_slot < 0 ||
        static_cast<uint32_t>(reference.output_slot) >=
            kMaxRuntimeOutputsPerTask ||
        (reference.flags & ~uint8_t{1}) != 0U) {
        return false;
    }
    if (reference.flags == 0U) {
        return reference.view_ndims == 0U &&
               reference.view_shape0 == 0U &&
               reference.view_offset0 == 0U;
    }
    // 当前 wire 只定义一维 view。shape 为 0 没有业务含义；同时显式检查
    // offset+shape，避免消费端在 32bit 地址/shape 运算中回绕。
    return reference.view_ndims == 1U &&
           reference.view_shape0 != 0U &&
           reference.view_offset0 <=
               UINT32_MAX - reference.view_shape0;
}

AICPU_PLAN_DEVICE bool RuntimeTaskPlanTensorWordOffset(
    const RuntimeTaskPlanHeader &header, uint32_t tensor,
    uint32_t &word_offset
)
{
    if (tensor >= header.tensor_count ||
        header.tensor_count > kMaxTaskTensors) {
        return false;
    }
    word_offset = kPlanHeaderWords + tensor * kTensorCanonicalWords;
    return true;
}

AICPU_PLAN_DEVICE bool ComputeRuntimeTaskPlanLayout(
    const RuntimeTaskPlanSpec &spec, const uint8_t tags[kMaxTaskTensors],
    RuntimeTaskPlanLayout &layout
)
{
    if (spec.task_id == UINT32_MAX || spec.tensor_count > kMaxTaskTensors ||
        spec.scalar_count > kMaxTaskScalars || spec.explicit_dep_count > kMaxExplicitDependencies ||
        spec.output_count > spec.tensor_count ||
        spec.require_sync_start > 1U || spec.reserved != 0U ||
        (spec.engine_class != EngineClass::MetadataOnly && spec.engine_class != EngineClass::Aic &&
         spec.engine_class != EngineClass::Aiv) ||
        (spec.engine_class == EngineClass::MetadataOnly
             ? spec.function_id != kInvalidFunctionId
             : spec.function_id == kInvalidFunctionId) ||
        (spec.tensor_reference_mask & ~TensorMaskForCount(spec.tensor_count)) != 0U) {
        return false;
    }
    uint32_t words = kPlanHeaderWords;
    uint32_t outputs = 0U;
    for (uint32_t tensor = 0U; tensor < spec.tensor_count; ++tensor) {
        const TensorTag tag = static_cast<TensorTag>(tags[tensor]);
        if (!TensorTagValid(tag)) return false;
        const bool reference = (spec.tensor_reference_mask & (uint32_t{1} << tensor)) != 0U;
        if (reference && tag == TensorTag::Output) return false;
        if (tag == TensorTag::Output) ++outputs;
    }
    if (outputs != spec.output_count) return false;
    words += static_cast<uint32_t>(spec.tensor_count) *
             kTensorCanonicalWords;
    layout.scalar_word_offset = words;
    words += spec.scalar_count;
    layout.explicit_dep_word_offset = words;
    words += spec.explicit_dep_count;
    layout.payload_words = words;
    layout.payload_bytes = words * sizeof(uint64_t);
    layout.payload_lines = (layout.payload_bytes + kPlanCacheLineBytes - 1U) / kPlanCacheLineBytes;
    return words <= kMaxPlanPayloadWords && layout.payload_lines >= 1U &&
           layout.payload_lines <= kMaxPlanPayloadLines;
}

// Header 的 wire ABI 明确定义为 8 个 little-word 位域。Host 与 A5 都按
// 这些 bit 位编码，而不是把 RuntimeTaskPlanHeader 强转成 uint64_t 数组；
// 后者同时违反 strict-aliasing，并可能让编译器读取结构体 padding。
AICPU_PLAN_DEVICE uint64_t RuntimeTaskPlanHeaderWord(
    const RuntimeTaskPlanHeader &header, uint32_t word
)
{
    if (word == 0U) {
        return static_cast<uint64_t>(header.task_id) |
               (static_cast<uint64_t>(header.function_id) << 32U);
    }
    if (word == 1U) {
        return static_cast<uint64_t>(header.tensor_count) |
               (static_cast<uint64_t>(header.scalar_count) << 16U) |
               (static_cast<uint64_t>(header.explicit_dep_count) << 32U) |
               (static_cast<uint64_t>(header.output_count) << 48U);
    }
    if (word == 2U) {
        return static_cast<uint64_t>(header.engine_class) |
               (static_cast<uint64_t>(header.adapter_flags) << 8U) |
               (static_cast<uint64_t>(
                    static_cast<uint16_t>(header.core_num)
                ) << 16U) |
               (static_cast<uint64_t>(header.require_sync_start) << 32U) |
               (static_cast<uint64_t>(header.reserved0) << 40U) |
               (static_cast<uint64_t>(header.reserved1) << 48U);
    }
    if (word >= 3U && word <= 6U) {
        uint64_t tags = 0U;
        const uint32_t first_tag = (word - 3U) * 8U;
        for (uint32_t index = 0U; index < 8U; ++index) {
            tags |= static_cast<uint64_t>(
                        header.tensor_tags[first_tag + index]
                    ) << (index * 8U);
        }
        return tags;
    }
    if (word == 7U) {
        return static_cast<uint64_t>(header.tensor_reference_mask) |
               (static_cast<uint64_t>(header.abi_version) << 32U);
    }
    return 0U;
}

template <typename Ops, typename Source>
AICPU_PLAN_DEVICE bool PackRuntimeTaskPlan(
    AICPU_PLAN_GM RuntimeTaskPlanCell &cell,
    const RuntimeTaskPlanSpec &spec,
    const Source &source, RuntimeTaskPlanLayout &layout
)
{
    uint8_t tags[kMaxTaskTensors] = {};
    for (uint32_t tensor = 0U; tensor < spec.tensor_count; ++tensor) {
        const TensorTag tag = source.TensorTagAt(tensor);
        if (!TensorTagValid(tag)) return false;
        tags[tensor] = static_cast<uint8_t>(tag);
    }
    if (!ComputeRuntimeTaskPlanLayout(spec, tags, layout)) return false;

    AICPU_PLAN_GM volatile uint64_t *destination = cell.payload.words;
    RuntimeTaskPlanHeader header{};
    header.task_id = spec.task_id;
    header.function_id = spec.function_id;
    header.tensor_count = spec.tensor_count;
    header.scalar_count = spec.scalar_count;
    header.explicit_dep_count = spec.explicit_dep_count;
    header.output_count = spec.output_count;
    header.engine_class = static_cast<uint8_t>(spec.engine_class);
    header.adapter_flags = spec.adapter_flags;
    header.core_num = spec.core_num;
    header.require_sync_start = spec.require_sync_start;
    for (uint32_t tensor = 0U; tensor < spec.tensor_count; ++tensor) {
        header.tensor_tags[tensor] = tags[tensor];
    }
    header.tensor_reference_mask = spec.tensor_reference_mask;
    header.abi_version = kRuntimePlanAbiVersion;
    for (uint32_t word = 0U; word < kPlanHeaderWords; ++word) {
        Ops::StorePayloadWord(
            &destination[word], RuntimeTaskPlanHeaderWord(header, word)
        );
    }

    uint32_t word = kPlanHeaderWords;
    for (uint32_t tensor = 0U; tensor < spec.tensor_count; ++tensor) {
        const TensorTag tag = static_cast<TensorTag>(tags[tensor]);
        const bool reference = (spec.tensor_reference_mask & (uint32_t{1} << tensor)) != 0U;
        if (reference != source.TensorIsReference(tensor)) return false;
        const uint32_t meaningful_words =
            TensorMeaningfulWords(tag, reference);
        uint64_t reference_first = 0U;
        uint64_t reference_second = 0U;
        if (reference) {
            reference_first = source.TensorWord(tensor, 0U);
            reference_second = source.TensorWord(tensor, 1U);
            const RuntimeOutputReferenceWire output_reference =
                DecodeRuntimeOutputReferenceWire(
                    reference_first, reference_second
                );
            if (!RuntimeOutputReferenceWireValid(
                    output_reference, spec.task_id
                )) {
                return false;
            }
        }
        for (uint32_t tensor_word = 0U;
             tensor_word < kTensorCanonicalWords; ++tensor_word) {
            uint64_t value = 0U;
            if (tensor_word < meaningful_words) {
                value = reference
                    ? (tensor_word == 0U
                           ? reference_first
                           : reference_second)
                    : source.TensorWord(tensor, tensor_word);
            }
            Ops::StorePayloadWord(&destination[word++], value);
        }
    }
    for (uint32_t scalar = 0U; scalar < spec.scalar_count; ++scalar) {
        Ops::StorePayloadWord(&destination[word++], source.Scalar(scalar));
    }
    for (uint32_t dependency = 0U; dependency < spec.explicit_dep_count; ++dependency) {
        Ops::StorePayloadWord(&destination[word++], source.ExplicitDependency(dependency));
    }
    if (word != layout.payload_words) return false;
    // 最后一条 cache line 中尚未使用的 word 一并清零。AICPU 发布时会
    // clean 整条 line；若保留上一轮脏字节，既会污染诊断，也会妨碍将来
    // 对同一 cell 地址做受控复用。
    const uint32_t flushed_words =
        layout.payload_lines * kPlanCacheLineBytes / sizeof(uint64_t);
    while (word < flushed_words) {
        Ops::StorePayloadWord(&destination[word++], 0U);
    }
    return true;
}

AICPU_PLAN_DEVICE RuntimeTaskPlanHeader DecodeRuntimeTaskPlanHeader(
    AICPU_PLAN_GM const RuntimeTaskPlanStorage &payload
)
{
    RuntimeTaskPlanHeader header{};
    const uint64_t word0 = payload.words[0];
    const uint64_t word1 = payload.words[1];
    const uint64_t word2 = payload.words[2];
    header.task_id = static_cast<uint32_t>(word0);
    header.function_id = static_cast<uint32_t>(word0 >> 32U);
    header.tensor_count = static_cast<uint16_t>(word1);
    header.scalar_count = static_cast<uint16_t>(word1 >> 16U);
    header.explicit_dep_count = static_cast<uint16_t>(word1 >> 32U);
    header.output_count = static_cast<uint16_t>(word1 >> 48U);
    header.engine_class = static_cast<uint8_t>(word2);
    header.adapter_flags = static_cast<uint8_t>(word2 >> 8U);
    header.core_num = DecodeRuntimeWireInt16(
        static_cast<uint16_t>(word2 >> 16U)
    );
    header.require_sync_start = static_cast<uint8_t>(word2 >> 32U);
    header.reserved0 = static_cast<uint8_t>(word2 >> 40U);
    header.reserved1 = static_cast<uint16_t>(word2 >> 48U);
    for (uint32_t group = 0U; group < 4U; ++group) {
        const uint64_t tag_word = payload.words[3U + group];
        for (uint32_t index = 0U; index < 8U; ++index) {
            header.tensor_tags[group * 8U + index] =
                static_cast<uint8_t>(tag_word >> (index * 8U));
        }
    }
    const uint64_t word7 = payload.words[7];
    header.tensor_reference_mask = static_cast<uint32_t>(word7);
    header.abi_version = static_cast<uint32_t>(word7 >> 32U);
    return header;
}

AICPU_PLAN_DEVICE bool ValidateRuntimeTaskPlanPayload(
    AICPU_PLAN_GM const RuntimeTaskPlanStorage &payload,
    uint32_t expected_task_id,
    uint32_t published_lines, RuntimeTaskPlanHeader &header,
    RuntimeTaskPlanLayout &layout
)
{
    header = DecodeRuntimeTaskPlanHeader(payload);
    const RuntimeTaskPlanSpec spec{
        header.task_id,
        header.function_id,
        header.tensor_count,
        header.scalar_count,
        header.explicit_dep_count,
        header.output_count,
        static_cast<EngineClass>(header.engine_class),
        header.adapter_flags,
        header.core_num,
        header.require_sync_start,
        header.reserved0,
        header.tensor_reference_mask,
    };
    if (header.reserved1 != 0U || header.abi_version != kRuntimePlanAbiVersion ||
        header.task_id != expected_task_id ||
        !ComputeRuntimeTaskPlanLayout(spec, header.tensor_tags, layout) ||
        layout.payload_lines != published_lines) {
        return false;
    }
    for (uint32_t tensor = header.tensor_count;
         tensor < kMaxTaskTensors; ++tensor) {
        if (header.tensor_tags[tensor] != 0U) return false;
    }
    uint32_t word = kPlanHeaderWords;
    for (uint32_t tensor = 0U; tensor < header.tensor_count; ++tensor) {
        const TensorTag tag = static_cast<TensorTag>(header.tensor_tags[tensor]);
        const bool reference = (header.tensor_reference_mask & (uint32_t{1} << tensor)) != 0U;
        if (reference) {
            const RuntimeOutputReferenceWire output_reference =
                DecodeRuntimeOutputReferenceWire(
                    payload.words[word], payload.words[word + 1U]
                );
            if (!RuntimeOutputReferenceWireValid(
                    output_reference, expected_task_id
                )) {
                return false;
            }
        }
        const uint32_t meaningful_words =
            TensorMeaningfulWords(tag, reference);
        for (uint32_t tensor_word = meaningful_words;
             tensor_word < kTensorCanonicalWords; ++tensor_word) {
            if (payload.words[word + tensor_word] != 0U) return false;
        }
        word += kTensorCanonicalWords;
    }
    if (word != layout.scalar_word_offset) return false;
    // scalar/explicit-dependency 后的最后一条已发布 cache line 也必须为
    // canonical zero。这样同地址重用或局部 flush 缺失不会被静默接受。
    const uint32_t flushed_words =
        published_lines * kPlanCacheLineBytes / sizeof(uint64_t);
    for (uint32_t tail = layout.payload_words;
         tail < flushed_words; ++tail) {
        if (payload.words[tail] != 0U) return false;
    }
    return true;
}

// ProducerOps：LoadControl、StorePayloadWord、FlushRegion、StoreBarrier、
// PublishControl。AICPU 的 PublishControl 是 ordinary store + exact clean；
// AIC/AIV 通过 return-ready atomic observe 读取它。
template <typename ProducerOps, typename Source>
AICPU_PLAN_DEVICE PlanPublishResult PublishRuntimeTaskPlan(
    const RuntimePlanView &view, const RuntimeTaskPlanSpec &spec,
    const Source &source
)
{
    if (view.control == nullptr || view.cells == nullptr || spec.task_id >= view.capacity ||
        ProducerOps::LoadControl(&view.control->fatal.value) != 0 ||
        ProducerOps::LoadControl(&view.control->closed_task_count.value) !=
            kPlanOpenTaskCount ||
        ProducerOps::LoadControl(&view.control->planned_frontier.value) !=
            static_cast<int64_t>(spec.task_id) ||
        ProducerOps::LoadControl(&view.cells[spec.task_id].control.value) != 0) {
        return PlanPublishResult::CellUnavailable;
    }
    RuntimeTaskPlanLayout layout{};
    if (!PackRuntimeTaskPlan<ProducerOps>(view.cells[spec.task_id], spec, source, layout)) {
        return PlanPublishResult::InvalidInput;
    }
    ProducerOps::FlushRegion(&view.cells[spec.task_id].payload, layout.payload_lines * kPlanCacheLineBytes);
    ProducerOps::StoreBarrier();
    ProducerOps::PublishControl(
        &view.cells[spec.task_id].control.value,
        static_cast<int64_t>(EncodePlanCellControl(PlanCellPhase::Published, layout.payload_lines, spec.task_id))
    );
    return PlanPublishResult::Published;
}

template <typename ProducerOps>
AICPU_PLAN_DEVICE bool AdvancePlannedFrontier(
    const RuntimePlanView &view, uint32_t expected_task_id
)
{
    if (expected_task_id >= view.capacity ||
        ProducerOps::LoadControl(&view.control->planned_frontier.value) !=
            static_cast<int64_t>(expected_task_id)) {
        return false;
    }
    ProducerOps::StoreBarrier();
    ProducerOps::PublishControl(
        &view.control->planned_frontier.value,
        static_cast<int64_t>(expected_task_id) + 1
    );
    return true;
}

template <typename ProducerOps>
AICPU_PLAN_DEVICE bool CloseRuntimePlan(
    const RuntimePlanView &view, uint32_t final_task_count
)
{
    if (final_task_count > view.capacity ||
        ProducerOps::LoadControl(&view.control->planned_frontier.value) != static_cast<int64_t>(final_task_count) ||
        ProducerOps::LoadControl(&view.control->closed_task_count.value) !=
            kPlanOpenTaskCount ||
        ProducerOps::LoadControl(&view.control->build_next.value) != 0 ||
        ProducerOps::LoadControl(&view.control->build_workers_done.value) != 0 ||
        ProducerOps::LoadControl(&view.control->build_release.value) !=
            kBuildReleasePending ||
        ProducerOps::LoadControl(&view.control->fatal.value) != 0) {
        return false;
    }
    ProducerOps::StoreBarrier();
    ProducerOps::PublishControl(&view.control->closed_task_count.value, final_task_count);
    return true;
}

// 首版固定 Plan-ahead：AICPU 完整关闭 Plan 后才唤醒 96 个 Scalar。
// 因此这里复用经过实测的中央 FetchAdd ticket，而不是为尚未实现的流式
// Plan 引入 CAS frontier reservation。每个 worker 在首次越界后退出，
// 正常总调用数严格为 N + worker_count。
template <typename ConsumerOps>
AICPU_PLAN_DEVICE BuildReservation TakeClosedPlanBuildTicket(
    const RuntimePlanView &view
)
{
    if (view.control == nullptr || view.cells == nullptr || view.capacity == 0U) {
        return {BuildReservationStatus::Fatal, 0U};
    }
    if (ConsumerOps::LoadControl(&view.control->fatal.value) != 0) {
        return {BuildReservationStatus::Fatal, 0U};
    }
    const int64_t closed =
        ConsumerOps::LoadControl(&view.control->closed_task_count.value);
    const int64_t frontier =
        ConsumerOps::LoadControl(&view.control->planned_frontier.value);
    if (closed < 0 || closed != frontier ||
        closed > static_cast<int64_t>(view.capacity)) {
        return {BuildReservationStatus::Fatal, 0U};
    }
    const int64_t ticket = ConsumerOps::FetchAddControl(
        &view.control->build_next.value, 1
    );
    if (ticket < 0) return {BuildReservationStatus::Fatal, 0U};
    if (ticket >= closed) return {BuildReservationStatus::Closed, 0U};
    return {
        BuildReservationStatus::Reserved,
        static_cast<uint32_t>(ticket),
    };
}


template <typename ConsumerOps>
AICPU_PLAN_DEVICE BuildArrivalStatus ArriveClosedPlanBuildWorker(
    const RuntimePlanView &view, uint32_t worker_count
)
{
    if (view.control == nullptr || worker_count == 0U ||
        ConsumerOps::LoadControl(&view.control->fatal.value) != 0) {
        return BuildArrivalStatus::Fatal;
    }
    const int64_t old = ConsumerOps::FetchAddControl(
        &view.control->build_workers_done.value, 1
    );
    if (old < 0 || old >= static_cast<int64_t>(worker_count)) {
        return BuildArrivalStatus::Invalid;
    }
    return old + 1 == static_cast<int64_t>(worker_count)
        ? BuildArrivalStatus::Last
        : BuildArrivalStatus::Arrived;
}

template <typename ConsumerOps>
AICPU_PLAN_DEVICE bool PublishClosedPlanBuildRelease(
    const RuntimePlanView &view, uint32_t final_task_count,
    uint32_t worker_count
)
{
    const uint64_t expected_ticket_count =
        static_cast<uint64_t>(final_task_count) + worker_count;
    if (view.control == nullptr || final_task_count > view.capacity ||
        worker_count == 0U || expected_ticket_count > INT64_MAX ||
        ConsumerOps::LoadControl(&view.control->fatal.value) != 0 ||
        ConsumerOps::LoadControl(&view.control->closed_task_count.value) !=
            static_cast<int64_t>(final_task_count) ||
        // 每个 worker 只有在首次取得越界 ticket 后才允许报到，因此正常
        // 终值必须精确为 N+W。只校验 workers_done 会漏掉“提前报到但任务
        // 尚未完整领取”的实现错误，并可能错误放行 Execute。
        ConsumerOps::LoadControl(&view.control->build_next.value) !=
            static_cast<int64_t>(expected_ticket_count) ||
        ConsumerOps::LoadControl(&view.control->build_workers_done.value) !=
            static_cast<int64_t>(worker_count) ||
        ConsumerOps::LoadControl(&view.control->build_release.value) !=
            kBuildReleasePending) {
        return false;
    }
    ConsumerOps::StoreBarrier();
    ConsumerOps::PublishControl(
        &view.control->build_release.value,
        static_cast<int64_t>(final_task_count)
    );
    return true;
}

template <typename ConsumerOps>
AICPU_PLAN_DEVICE PlanAcquireResult AcquireRuntimeTaskPlan(
    const RuntimePlanView &view, uint32_t task_id,
    RuntimeTaskPlanHeader &header, RuntimeTaskPlanLayout &layout
)
{
    if (task_id >= view.capacity || ConsumerOps::LoadControl(&view.control->fatal.value) != 0) {
        return PlanAcquireResult::FatalObserved;
    }
    AICPU_PLAN_GM RuntimeTaskPlanCell &cell = view.cells[task_id];
    const int64_t first = ConsumerOps::LoadControl(&cell.control.value);
    const DecodedPlanCellControl decoded = DecodePlanCellControl(first);
    if (!decoded.valid) return PlanAcquireResult::InvalidControl;
    if (decoded.phase != PlanCellPhase::Published) return PlanAcquireResult::NotPublished;
    if (decoded.task_id != task_id) return PlanAcquireResult::InvalidControl;
    ConsumerOps::InvalidateRegion(&cell.payload, decoded.payload_lines * kPlanCacheLineBytes);
    if (ConsumerOps::LoadControl(&cell.control.value) != first) return PlanAcquireResult::InvalidControl;
    if (!ValidateRuntimeTaskPlanPayload(cell.payload, task_id, decoded.payload_lines, header, layout)) {
        return PlanAcquireResult::InvalidPayload;
    }
    return PlanAcquireResult::Acquired;
}

}  // namespace pa_scheduler::aicpu_plan

#if defined(AICPU_PLAN_UNDEFINE_GM)
#undef AICPU_PLAN_GM
#undef AICPU_PLAN_UNDEFINE_GM
#endif
#if defined(AICPU_PLAN_UNDEFINE_DEVICE)
#undef AICPU_PLAN_DEVICE
#undef AICPU_PLAN_UNDEFINE_DEVICE
#endif

#endif  // PA_SCHEDULER_CROSS_CORE_AICPU_PLAN_PROTOCOL_H
