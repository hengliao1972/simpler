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

#ifndef PA_SCHEDULER_CROSS_CORE_SHARED_EXEC_PROTOCOL_H
#define PA_SCHEDULER_CROSS_CORE_SHARED_EXEC_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#ifndef PA_DEVICE
#error "PA_DEVICE must be defined before including shared_exec_protocol.h"
#endif

#ifndef PA_GM
#error "PA_GM must be defined before including shared_exec_protocol.h"
#endif

namespace pa_scheduler::cross_core {

constexpr uint32_t kExecCacheLineBytes = 64;
constexpr uint32_t kExecTensorDescBytes = 128;
constexpr uint32_t kExecTensorDescWords =
    kExecTensorDescBytes / sizeof(uint64_t);
constexpr uint32_t kExecMaxTensors = 32;
constexpr uint32_t kExecMaxScalars = 16;
constexpr uint32_t kExecMaxFanin = 16;
constexpr uint32_t kExecHeaderWords =
    kExecCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kExecMaxPayloadBytes =
    kExecCacheLineBytes +
    kExecMaxTensors * kExecTensorDescBytes +
    kExecMaxScalars * sizeof(uint64_t) +
    kExecMaxFanin * sizeof(int32_t);
constexpr uint32_t kExecMaxPayloadLines =
    (kExecMaxPayloadBytes + kExecCacheLineBytes - 1U) /
    kExecCacheLineBytes;
constexpr uint32_t kExecMaxPayloadWords =
    kExecMaxPayloadLines * kExecHeaderWords;
constexpr uint32_t kExecInvalidFunctionId = UINT32_MAX;
constexpr uint32_t kExecMaxOwner = 254;
constexpr uint32_t kExecUnboundOwner = 255;
constexpr uint32_t kExecDispatchArgCount = 50;
constexpr uint32_t kExecLocalContextBytes = 48;
constexpr uint32_t kExecGlobalContextBytes = 4;
constexpr uint32_t kExecDispatchBindingBytes = 512;
constexpr uint32_t kExecDispatchLocalContextIndex = 48;
constexpr uint32_t kExecDispatchGlobalContextIndex = 49;
// 当前调度模型允许同一个 Scalar 同时保存两个已领取、尚未完成的 task。
// 容量先作为协议常量固定为 2；以后若参数化，必须重新验证状态大小、
// FinalDrain 和“容量满时不再 Claim”的边界。
constexpr uint32_t kExecTokensPerWorker = 2;

static_assert(
    kExecTensorDescBytes % sizeof(uint64_t) == 0,
    "portable TensorDesc must be copied as complete uint64 words"
);
static_assert(
    kExecMaxPayloadBytes == 4352 &&
        kExecMaxPayloadLines == 68,
    "shared execution payload capacity changed"
);
static_assert(
    kExecMaxTensors + kExecMaxScalars ==
            kExecDispatchLocalContextIndex &&
        kExecDispatchGlobalContextIndex + 1 ==
            kExecDispatchArgCount,
    "dispatch argument indexes no longer match payload capacity"
);
static_assert(
    kExecTokensPerWorker == 2,
    "the first Claim-first implementation requires exactly two tokens"
);

enum class ExecPhase : uint8_t {
    Empty = 0,
    Building = 1,
    Built = 2,
    Claimed = 3,
    Done = 4,
};

enum class ExecEngineClass : uint8_t {
    None = 0,
    Aic = 1,
    Aiv = 2,
    Joint = 3,
};

enum class ExecTokenPhase : uint32_t {
    Idle = 0,
    Binding = 1,
    WaitingFanin = 2,
    EngineInflight = 3,
    Completing = 4,
    VendPublished = 5,
    CompletionPublished = 6,
    Faulted = 7,
};

enum class ExecFatalReason : uint8_t {
    None = 0,
    InvalidBuildInput = 1,
    BuildPackFailed = 2,
    InvalidBuiltControl = 3,
    ClaimedPayloadInvalid = 4,
    ControlPublishConflict = 5,
    InvalidTokenPayload = 6,
    CompletionPublishFailed = 7,
    CompletionStateConflict = 8,
};

enum class ExecBuildResult : uint32_t {
    Published = 0,
    InvalidInput = 1,
    CellUnavailable = 2,
    PublishConflict = 3,
    FatalObserved = 4,
};

enum class ExecClaimResult : uint32_t {
    Claimed = 0,
    TokenBusy = 1,
    NotBuilt = 2,
    Incompatible = 3,
    Lost = 4,
    InvalidControl = 5,
    InvalidPayload = 6,
    FatalObserved = 7,
};

enum class ExecDoneResult : uint32_t {
    Done = 0,
    TokenNotCompleting = 1,
    InvalidTokenPayload = 2,
    VendPublishFailed = 3,
    FlagPublishFailed = 4,
    StateConflict = 5,
    FatalObserved = 6,
};

constexpr uint64_t kExecStatePhaseShift = 0;
constexpr uint64_t kExecStatePhaseMask = 0x7ULL;
constexpr uint64_t kExecStateBuildOwnerShift = 3;
constexpr uint64_t kExecStateBuildOwnerMask = 0xFFULL;
constexpr uint64_t kExecStateExecuteOwnerShift = 11;
constexpr uint64_t kExecStateExecuteOwnerMask = 0xFFULL;
constexpr uint64_t kExecStateEngineShift = 19;
constexpr uint64_t kExecStateEngineMask = 0x7ULL;
constexpr uint64_t kExecStatePayloadLinesShift = 22;
constexpr uint64_t kExecStatePayloadLinesMask = 0x7FULL;
constexpr uint64_t kExecStateTaskIdShift = 29;
constexpr uint64_t kExecStateTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kExecStateKnownMask =
    (kExecStatePhaseMask << kExecStatePhaseShift) |
    (kExecStateBuildOwnerMask << kExecStateBuildOwnerShift) |
    (kExecStateExecuteOwnerMask << kExecStateExecuteOwnerShift) |
    (kExecStateEngineMask << kExecStateEngineShift) |
    (kExecStatePayloadLinesMask << kExecStatePayloadLinesShift) |
    (kExecStateTaskIdMask << kExecStateTaskIdShift);

constexpr uint64_t kExecFatalReasonShift = 0;
constexpr uint64_t kExecFatalReasonMask = 0xFFULL;
constexpr uint64_t kExecFatalOwnerShift = 8;
constexpr uint64_t kExecFatalOwnerMask = 0xFFULL;
constexpr uint64_t kExecFatalTaskIdShift = 16;
constexpr uint64_t kExecFatalTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kExecFatalKnownMask =
    (kExecFatalReasonMask << kExecFatalReasonShift) |
    (kExecFatalOwnerMask << kExecFatalOwnerShift) |
    (kExecFatalTaskIdMask << kExecFatalTaskIdShift);

struct DecodedExecState {
    ExecPhase phase;
    uint32_t build_owner;
    uint32_t execute_owner;
    ExecEngineClass engine_class;
    uint32_t payload_lines;
    uint32_t task_id;
    bool valid;
};

struct DecodedExecFatal {
    ExecFatalReason reason;
    uint32_t reporter_owner;
    uint32_t task_id;
    bool valid;
};

struct ExecPayloadLayout {
    uint32_t payload_bytes;
    uint32_t payload_lines;
    uint32_t tensor_word_offset;
    uint32_t scalar_word_offset;
    uint32_t fanin_word_offset;
    uint32_t written_words;
    uint32_t tensor_reference_mask;
    uint32_t inline_tensor_count;
};

struct ExecPayloadSpec {
    uint32_t task_id;
    uint64_t function_address;
    uint64_t completion_vend;
    uint32_t function_id;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
    ExecEngineClass engine_class;
    uint8_t flags;
    uint32_t multicore_group_id;
    uint16_t multicore_rank;
    uint16_t multicore_size;
    // bit=1 表示对应 tensor 参数携带一个生命周期稳定的 GM TensorDesc
    // 地址；bit=0 仍在 payload 内完整内联 128B descriptor。
    uint32_t tensor_reference_mask;
};

struct ExecPayloadHeader {
    uint32_t task_id;
    uint64_t function_address;
    uint64_t completion_vend;
    uint32_t function_id;
    uint32_t payload_bytes;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
    ExecEngineClass engine_class;
    uint8_t flags;
    uint32_t multicore_group_id;
    uint16_t multicore_rank;
    uint16_t multicore_size;
    uint32_t tensor_reference_mask;
};

struct alignas(kExecCacheLineBytes) SharedExecControl {
    volatile int64_t state;
    uint8_t padding[kExecCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kExecCacheLineBytes) SharedExecFatalControl {
    volatile int64_t state;
    uint8_t padding[kExecCacheLineBytes - sizeof(int64_t)];
};

// 所有 replay actor 停产之后，再用一条独立 atomic-only line 汇合本核
// execution token 的排空证据。最后到达者逐 task 核对 cell 终态后发布
// release；这样 device 退出不依赖 host 事后发现遗失的 BUILT task。
struct alignas(kExecCacheLineBytes) SharedExecDrainControl {
    volatile int64_t arrived;
    volatile int64_t release;
    uint8_t padding[
        kExecCacheLineBytes - 2U * sizeof(int64_t)
    ];
};

struct alignas(kExecCacheLineBytes) ExecPayloadStorage {
    volatile uint64_t words[kExecMaxPayloadWords];
};

struct alignas(kExecCacheLineBytes) SharedExecCell {
    SharedExecControl control;
    ExecPayloadStorage payload;
};

struct alignas(kExecCacheLineBytes) ExecutionTokenControl {
    ExecTokenPhase phase;
    uint32_t task_id;
    uint32_t build_owner;
    uint32_t execute_owner;
    ExecEngineClass engine_class;
    uint32_t payload_lines;
    uint32_t payload_bytes;
    uint32_t fanin_ready_prefix;
    uint8_t padding[32];
};

struct alignas(kExecCacheLineBytes) ExecutionDispatchBinding {
    // 这 50 个入口对应通用 dispatch ABI：有效 tensor/scalar 参数在
    // 前缀中，local/global context 固定占最后两个入口。这里不保存
    // builder 侧 self-pointer，executor 取得 payload 后必须重新绑定。
    uint64_t args[kExecDispatchArgCount];
    uint8_t local_context[kExecLocalContextBytes];
    uint8_t global_context[kExecGlobalContextBytes];
    uint8_t padding[
        kExecDispatchBindingBytes -
        kExecDispatchArgCount * sizeof(uint64_t) -
        kExecLocalContextBytes - kExecGlobalContextBytes
    ];
};

struct alignas(kExecCacheLineBytes) ExecutionToken {
    ExecutionTokenControl control;
    ExecPayloadStorage payload;
    ExecutionDispatchBinding dispatch;
};

static_assert(
    sizeof(SharedExecControl) == kExecCacheLineBytes &&
        alignof(SharedExecControl) == kExecCacheLineBytes,
    "shared execution control must own one cache line"
);
static_assert(
    sizeof(SharedExecFatalControl) == kExecCacheLineBytes &&
        alignof(SharedExecFatalControl) == kExecCacheLineBytes,
    "global fatal control must own one atomic-only cache line"
);
static_assert(
    sizeof(SharedExecDrainControl) == kExecCacheLineBytes &&
        alignof(SharedExecDrainControl) == kExecCacheLineBytes,
    "execution drain control must own one atomic-only cache line"
);
static_assert(
    offsetof(SharedExecCell, payload) == kExecCacheLineBytes,
    "shared execution payload must not share its control line"
);
static_assert(
    sizeof(ExecPayloadStorage) == kExecMaxPayloadBytes &&
        alignof(ExecPayloadStorage) == kExecCacheLineBytes,
    "shared execution payload storage ABI changed"
);
static_assert(
    sizeof(SharedExecCell) ==
        kExecCacheLineBytes + kExecMaxPayloadBytes,
    "adjacent shared execution cells must remain cache-line isolated"
);
static_assert(
    sizeof(ExecutionTokenControl) == kExecCacheLineBytes &&
        offsetof(ExecutionToken, payload) == kExecCacheLineBytes,
    "execution token control and binding must remain separate"
);
static_assert(
    sizeof(ExecutionDispatchBinding) == kExecDispatchBindingBytes &&
        alignof(ExecutionDispatchBinding) == kExecCacheLineBytes &&
        offsetof(ExecutionDispatchBinding, args) == 0 &&
        offsetof(ExecutionDispatchBinding, local_context) == 400 &&
        offsetof(ExecutionDispatchBinding, global_context) == 448,
    "executor-private dispatch binding ABI changed"
);
static_assert(
    offsetof(ExecutionToken, dispatch) ==
            kExecCacheLineBytes + kExecMaxPayloadBytes &&
        offsetof(ExecutionToken, dispatch) % kExecCacheLineBytes == 0 &&
        sizeof(ExecutionToken) ==
            kExecCacheLineBytes + kExecMaxPayloadBytes +
                kExecDispatchBindingBytes,
    "token control, payload and dispatch binding must not share lines"
);

PA_DEVICE uint64_t EncodeExecState(
    ExecPhase phase, uint32_t build_owner,
    uint32_t execute_owner,
    ExecEngineClass engine_class, uint32_t payload_lines,
    uint32_t task_id
) {
    return
        (static_cast<uint64_t>(phase) << kExecStatePhaseShift) |
        (static_cast<uint64_t>(build_owner) <<
         kExecStateBuildOwnerShift) |
        (static_cast<uint64_t>(execute_owner) <<
         kExecStateExecuteOwnerShift) |
        (static_cast<uint64_t>(engine_class) <<
         kExecStateEngineShift) |
        (static_cast<uint64_t>(payload_lines) <<
         kExecStatePayloadLinesShift) |
        (static_cast<uint64_t>(task_id) << kExecStateTaskIdShift);
}

// 仅保留给现有独立探针构造状态的源码兼容入口。正式协议必须调用上面的
// 六参数版本，分别携带 build/execute owner；该入口不能用于验证跨核
// owner 保留语义。
PA_DEVICE uint64_t EncodeExecState(
    ExecPhase phase, uint32_t owner,
    ExecEngineClass engine_class, uint32_t payload_lines,
    uint32_t task_id
) {
    if (phase == ExecPhase::Empty) {
        return EncodeExecState(
            phase, 0, 0, engine_class, payload_lines, task_id
        );
    }
    const uint32_t execute_owner =
        phase == ExecPhase::Claimed || phase == ExecPhase::Done
            ? owner : kExecUnboundOwner;
    return EncodeExecState(
        phase, owner, execute_owner, engine_class,
        payload_lines, task_id
    );
}

PA_DEVICE bool ExecOwnerValid(uint32_t owner) {
    return owner <= kExecMaxOwner;
}

PA_DEVICE bool ExecFatalReasonValid(ExecFatalReason reason) {
    return reason >= ExecFatalReason::InvalidBuildInput &&
           reason <= ExecFatalReason::CompletionStateConflict;
}

PA_DEVICE uint64_t EncodeExecFatal(
    ExecFatalReason reason, uint32_t reporter_owner,
    uint32_t task_id
) {
    return
        (static_cast<uint64_t>(reason) << kExecFatalReasonShift) |
        (static_cast<uint64_t>(reporter_owner) <<
         kExecFatalOwnerShift) |
        (static_cast<uint64_t>(task_id) << kExecFatalTaskIdShift);
}

PA_DEVICE DecodedExecFatal DecodeExecFatal(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedExecFatal fatal{
        static_cast<ExecFatalReason>(
            (raw >> kExecFatalReasonShift) &
            kExecFatalReasonMask
        ),
        static_cast<uint32_t>(
            (raw >> kExecFatalOwnerShift) & kExecFatalOwnerMask
        ),
        static_cast<uint32_t>(
            (raw >> kExecFatalTaskIdShift) &
            kExecFatalTaskIdMask
        ),
        false,
    };
    fatal.valid = raw != 0 &&
                  (raw & ~kExecFatalKnownMask) == 0 &&
                  ExecFatalReasonValid(fatal.reason) &&
                  ExecOwnerValid(fatal.reporter_owner);
    return fatal;
}

template <typename Ops>
PA_DEVICE bool ExecFatalPublished(
    PA_GM SharedExecFatalControl &fatal
) {
    // 任意非零值都必须 fail-closed；即使记录本身已损坏，也不能继续。
    return Ops::Load(&fatal.state) != 0;
}

template <typename Ops>
PA_DEVICE bool PublishExecFatal(
    PA_GM SharedExecFatalControl &fatal,
    ExecFatalReason reason, uint32_t task_id,
    uint32_t reporter_owner
) {
    if (!ExecFatalReasonValid(reason) ||
        !ExecOwnerValid(reporter_owner)) {
        return false;
    }
    const int64_t desired = static_cast<int64_t>(
        EncodeExecFatal(reason, reporter_owner, task_id)
    );
    // first-failure-wins。fatal line 永不清零，也不做 ordinary store/DCCI。
    return Ops::CompareExchange(&fatal.state, 0, desired) == 0;
}

PA_DEVICE bool ExecEngineValid(ExecEngineClass engine_class) {
    return engine_class == ExecEngineClass::Aic ||
           engine_class == ExecEngineClass::Aiv ||
           engine_class == ExecEngineClass::Joint;
}

PA_DEVICE DecodedExecState DecodeExecState(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedExecState state{
        static_cast<ExecPhase>(
            (raw >> kExecStatePhaseShift) & kExecStatePhaseMask
        ),
        static_cast<uint32_t>(
            (raw >> kExecStateBuildOwnerShift) &
            kExecStateBuildOwnerMask
        ),
        static_cast<uint32_t>(
            (raw >> kExecStateExecuteOwnerShift) &
            kExecStateExecuteOwnerMask
        ),
        static_cast<ExecEngineClass>(
            (raw >> kExecStateEngineShift) & kExecStateEngineMask
        ),
        static_cast<uint32_t>(
            (raw >> kExecStatePayloadLinesShift) &
            kExecStatePayloadLinesMask
        ),
        static_cast<uint32_t>(
            (raw >> kExecStateTaskIdShift) &
            kExecStateTaskIdMask
        ),
        false,
    };
    if ((raw & ~kExecStateKnownMask) != 0) {
        return state;
    }
    switch (state.phase) {
        case ExecPhase::Empty:
            state.valid = state.build_owner == 0 &&
                          state.execute_owner == 0 &&
                          state.engine_class ==
                              ExecEngineClass::None &&
                          state.payload_lines == 0 &&
                          state.task_id == 0;
            break;
        case ExecPhase::Building:
            state.valid = ExecOwnerValid(state.build_owner) &&
                          state.execute_owner ==
                              kExecUnboundOwner &&
                          state.engine_class ==
                              ExecEngineClass::None &&
                          state.payload_lines == 0;
            break;
        case ExecPhase::Built:
            state.valid = ExecOwnerValid(state.build_owner) &&
                          state.execute_owner ==
                              kExecUnboundOwner &&
                          ExecEngineValid(state.engine_class) &&
                          state.payload_lines >= 1 &&
                          state.payload_lines <=
                              kExecMaxPayloadLines;
            break;
        case ExecPhase::Claimed:
        case ExecPhase::Done:
            state.valid = ExecOwnerValid(state.build_owner) &&
                          ExecOwnerValid(state.execute_owner) &&
                          ExecEngineValid(state.engine_class) &&
                          state.payload_lines >= 1 &&
                          state.payload_lines <=
                              kExecMaxPayloadLines;
            break;
        default:
            break;
    }
    return state;
}

PA_DEVICE uint32_t ExecTensorMaskForCount(uint32_t tensor_count) {
    return tensor_count >= 32U
        ? UINT32_MAX
        : ((uint32_t{1} << tensor_count) - 1U);
}

PA_DEVICE uint32_t ExecTensorReferenceCount(
    uint32_t tensor_reference_mask
) {
    uint32_t count = 0;
    while (tensor_reference_mask != 0) {
        count += tensor_reference_mask & 1U;
        tensor_reference_mask >>= 1U;
    }
    return count;
}

PA_DEVICE uint32_t ExecTensorPayloadWordOffset(
    uint32_t tensor, uint32_t tensor_reference_mask
) {
    const uint32_t preceding_mask = tensor == 0
        ? 0
        : tensor_reference_mask &
              ExecTensorMaskForCount(tensor);
    const uint32_t preceding_references =
        ExecTensorReferenceCount(preceding_mask);
    return kExecHeaderWords +
           tensor * kExecTensorDescWords -
           preceding_references * (kExecTensorDescWords - 1U);
}

PA_DEVICE bool ComputeExecPayloadLayout(
    uint32_t tensor_count, uint32_t scalar_count,
    uint32_t fanin_count, uint32_t tensor_reference_mask,
    ExecPayloadLayout &layout
) {
    if (tensor_count > kExecMaxTensors ||
        scalar_count > kExecMaxScalars ||
        fanin_count > kExecMaxFanin ||
        (tensor_reference_mask &
         ~ExecTensorMaskForCount(tensor_count)) != 0) {
        return false;
    }
    const uint32_t reference_count =
        ExecTensorReferenceCount(tensor_reference_mask);
    const uint32_t inline_tensor_count =
        tensor_count - reference_count;
    layout.tensor_word_offset = kExecHeaderWords;
    layout.scalar_word_offset =
        layout.tensor_word_offset +
        inline_tensor_count * kExecTensorDescWords +
        reference_count;
    layout.fanin_word_offset =
        layout.scalar_word_offset + scalar_count;
    layout.written_words =
        layout.fanin_word_offset + (fanin_count + 1U) / 2U;
    layout.payload_bytes =
        kExecCacheLineBytes +
        inline_tensor_count * kExecTensorDescBytes +
        reference_count * sizeof(uint64_t) +
        scalar_count * sizeof(uint64_t) +
        fanin_count * sizeof(int32_t);
    layout.payload_lines =
        (layout.payload_bytes + kExecCacheLineBytes - 1U) /
        kExecCacheLineBytes;
    layout.tensor_reference_mask = tensor_reference_mask;
    layout.inline_tensor_count = inline_tensor_count;
    return layout.payload_lines >= 1 &&
           layout.payload_lines <= kExecMaxPayloadLines &&
           layout.written_words <= kExecMaxPayloadWords;
}

PA_DEVICE bool ComputeExecPayloadLayout(
    uint32_t tensor_count, uint32_t scalar_count,
    uint32_t fanin_count, ExecPayloadLayout &layout
) {
    return ComputeExecPayloadLayout(
        tensor_count, scalar_count, fanin_count,
        /*tensor_reference_mask=*/0, layout
    );
}

PA_DEVICE bool ValidateExecPayloadSpec(
    const ExecPayloadSpec &spec, ExecPayloadLayout &layout
) {
    if (!ExecEngineValid(spec.engine_class) ||
        (spec.function_id == kExecInvalidFunctionId &&
         spec.function_address == 0) ||
        (spec.flags & ~1U) != 0) {
        return false;
    }
    const bool multicore = (spec.flags & 1U) != 0;
    if ((!multicore &&
         (spec.multicore_group_id != 0 ||
          spec.multicore_rank != 0 ||
          spec.multicore_size != 1)) ||
        (multicore &&
         (spec.multicore_size < 2 ||
          spec.multicore_rank >= spec.multicore_size))) {
        return false;
    }
    return ComputeExecPayloadLayout(
        spec.tensor_count, spec.scalar_count,
        spec.fanin_count, spec.tensor_reference_mask, layout
    );
}

PA_DEVICE uint64_t PackExecHeaderWord0(uint32_t task_id) {
    // 高 32 位保留并固定为 0。PA 的 output 由
    // (producer task_id, output_slot) 标识，completion 也发布到
    // 当前 task；这里不能再引入第二个含义不清的 task 身份。
    return static_cast<uint64_t>(task_id);
}

PA_DEVICE uint64_t PackExecHeaderWord3(
    uint32_t function_id, uint32_t payload_bytes
) {
    return static_cast<uint64_t>(function_id) |
           (static_cast<uint64_t>(payload_bytes) << 32U);
}

PA_DEVICE uint64_t PackExecHeaderWord4(
    const ExecPayloadSpec &spec
) {
    return static_cast<uint64_t>(spec.tensor_count) |
           (static_cast<uint64_t>(spec.scalar_count) << 16U) |
           (static_cast<uint64_t>(spec.fanin_count) << 32U) |
           (static_cast<uint64_t>(spec.engine_class) << 48U) |
           (static_cast<uint64_t>(spec.flags) << 56U);
}

PA_DEVICE uint64_t PackExecHeaderWord5(
    const ExecPayloadSpec &spec
) {
    return static_cast<uint64_t>(spec.multicore_group_id) |
           (static_cast<uint64_t>(spec.multicore_rank) << 32U) |
           (static_cast<uint64_t>(spec.multicore_size) << 48U);
}

template <typename Ops, typename Source>
PA_DEVICE bool PackExecPayload(
    PA_GM SharedExecCell &cell, const ExecPayloadSpec &spec,
    const Source &source, ExecPayloadLayout &layout
) {
    if (!ValidateExecPayloadSpec(spec, layout)) {
        return false;
    }
    PA_GM volatile uint64_t *destination = cell.payload.words;
    Ops::StorePayloadWord(
        &destination[0],
        PackExecHeaderWord0(spec.task_id)
    );
    Ops::StorePayloadWord(&destination[1], spec.function_address);
    Ops::StorePayloadWord(&destination[2], spec.completion_vend);
    Ops::StorePayloadWord(
        &destination[3],
        PackExecHeaderWord3(spec.function_id, layout.payload_bytes)
    );
    Ops::StorePayloadWord(
        &destination[4], PackExecHeaderWord4(spec)
    );
    Ops::StorePayloadWord(
        &destination[5], PackExecHeaderWord5(spec)
    );
    Ops::StorePayloadWord(
        &destination[6], spec.tensor_reference_mask
    );
    Ops::StorePayloadWord(&destination[7], 0);

    uint32_t destination_word = layout.tensor_word_offset;
    for (uint32_t tensor = 0;
         tensor < spec.tensor_count; ++tensor) {
        if ((spec.tensor_reference_mask &
             (uint32_t{1} << tensor)) != 0) {
            const uint64_t reference =
                source.TensorReference(tensor);
            if (reference == 0 ||
                reference % alignof(uint64_t) != 0) {
                return false;
            }
            Ops::StorePayloadWord(
                &destination[destination_word++], reference
            );
            continue;
        }
        for (uint32_t word = 0;
             word < kExecTensorDescWords; ++word) {
            Ops::StorePayloadWord(
                &destination[destination_word++],
                source.TensorWord(tensor, word)
            );
        }
    }
    for (uint32_t scalar = 0;
         scalar < spec.scalar_count; ++scalar) {
        Ops::StorePayloadWord(
            &destination[destination_word++],
            source.Scalar(scalar)
        );
    }
    for (uint32_t fanin = 0;
         fanin < spec.fanin_count; fanin += 2U) {
        const int32_t low_producer = source.Fanin(fanin);
        if (low_producer < 0 ||
            static_cast<uint32_t>(low_producer) >= spec.task_id) {
            return false;
        }
        const uint64_t low =
            static_cast<uint32_t>(low_producer);
        uint64_t high = 0;
        if (fanin + 1U < spec.fanin_count) {
            const int32_t high_producer = source.Fanin(fanin + 1U);
            if (high_producer < 0 ||
                static_cast<uint32_t>(high_producer) >=
                    spec.task_id) {
                return false;
            }
            high = static_cast<uint32_t>(high_producer);
        }
        Ops::StorePayloadWord(
            &destination[destination_word++], low | (high << 32U)
        );
    }
    return destination_word == layout.written_words;
}

PA_DEVICE ExecPayloadHeader DecodeExecPayloadHeader(
    PA_GM const ExecPayloadStorage &payload
) {
    const uint64_t word0 = payload.words[0];
    const uint64_t word3 = payload.words[3];
    const uint64_t word4 = payload.words[4];
    const uint64_t word5 = payload.words[5];
    const uint64_t word6 = payload.words[6];
    return ExecPayloadHeader{
        static_cast<uint32_t>(word0),
        payload.words[1],
        payload.words[2],
        static_cast<uint32_t>(word3),
        static_cast<uint32_t>(word3 >> 32U),
        static_cast<uint16_t>(word4),
        static_cast<uint16_t>(word4 >> 16U),
        static_cast<uint16_t>(word4 >> 32U),
        static_cast<ExecEngineClass>(
            static_cast<uint8_t>(word4 >> 48U)
        ),
        static_cast<uint8_t>(word4 >> 56U),
        static_cast<uint32_t>(word5),
        static_cast<uint16_t>(word5 >> 32U),
        static_cast<uint16_t>(word5 >> 48U),
        static_cast<uint32_t>(word6),
    };
}

PA_DEVICE bool ValidateBoundExecPayload(
    PA_GM const ExecutionToken &token,
    uint32_t expected_task_id,
    ExecEngineClass expected_engine,
    uint32_t published_lines,
    ExecPayloadHeader &header,
    ExecPayloadLayout &layout
) {
    header = DecodeExecPayloadHeader(token.payload);
    if ((token.payload.words[0] >> 32U) != 0 ||
        (token.payload.words[6] >> 32U) != 0 ||
        token.payload.words[7] != 0 ||
        header.task_id != expected_task_id ||
        header.engine_class != expected_engine ||
        (header.function_id == kExecInvalidFunctionId &&
         header.function_address == 0) ||
        (header.flags & ~1U) != 0 ||
        !ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count,
            header.tensor_reference_mask, layout
        ) ||
        header.payload_bytes != layout.payload_bytes ||
        published_lines != layout.payload_lines) {
        return false;
    }
    for (uint32_t tensor = 0;
         tensor < header.tensor_count; ++tensor) {
        if ((header.tensor_reference_mask &
             (uint32_t{1} << tensor)) == 0) {
            continue;
        }
        const uint64_t reference = token.payload.words[
            ExecTensorPayloadWordOffset(
                tensor, header.tensor_reference_mask
            )
        ];
        if (reference == 0 ||
            reference % alignof(uint64_t) != 0) {
            return false;
        }
    }
    const bool multicore = (header.flags & 1U) != 0;
    return
        (!multicore && header.multicore_group_id == 0 &&
         header.multicore_rank == 0 &&
         header.multicore_size == 1) ||
        (multicore && header.multicore_size >= 2 &&
         header.multicore_rank < header.multicore_size);
}

PA_DEVICE PA_GM uint64_t *ExecutionTokenDispatchArgs(
    PA_GM ExecutionToken &token
) {
    return &token.dispatch.args[0];
}

PA_DEVICE PA_GM const uint64_t *ExecutionTokenDispatchArgs(
    PA_GM const ExecutionToken &token
) {
    return &token.dispatch.args[0];
}

template <typename Ops>
PA_DEVICE bool RebuildExecutionTokenDispatchArgs(
    PA_GM ExecutionToken &token
) {
    const ExecPayloadHeader header =
        DecodeExecPayloadHeader(token.payload);
    ExecPayloadLayout layout{};
    if ((token.control.phase != ExecTokenPhase::Binding &&
         token.control.phase != ExecTokenPhase::WaitingFanin) ||
        !ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count,
            header.tensor_reference_mask, layout
        ) ||
        header.payload_bytes != layout.payload_bytes ||
        token.control.payload_bytes != layout.payload_bytes ||
        token.control.payload_lines != layout.payload_lines) {
        return false;
    }

    // builder-local descriptor 必须重绑到本 executor token；只有由
    // adapter 明确声明生命周期稳定的 GM descriptor 才保留绝对地址。
    // A5 Scalar 间没有 cache coherence，executor 在第一次使用引用前
    // 必须 invalidate 对应 128B，不能依赖 builder 或前一轮的 DCache。
    for (uint32_t tensor = 0;
         tensor < header.tensor_count; ++tensor) {
        const uint32_t word_offset = ExecTensorPayloadWordOffset(
            tensor, header.tensor_reference_mask
        );
        if ((header.tensor_reference_mask &
             (uint32_t{1} << tensor)) != 0) {
            const uint64_t reference =
                token.payload.words[word_offset];
            PA_GM const void *descriptor =
                reinterpret_cast<PA_GM const void *>(
                    static_cast<uintptr_t>(reference)
                );
            Ops::InvalidateRegion(
                descriptor, kExecTensorDescBytes
            );
            token.dispatch.args[tensor] = reference;
        } else {
            token.dispatch.args[tensor] = static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(
                    &token.payload.words[word_offset]
                )
            );
        }
    }
    for (uint32_t scalar = 0;
         scalar < header.scalar_count; ++scalar) {
        token.dispatch.args[header.tensor_count + scalar] =
            token.payload.words[layout.scalar_word_offset + scalar];
    }
    token.dispatch.args[kExecDispatchLocalContextIndex] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            &token.dispatch.local_context[0]
        ));
    token.dispatch.args[kExecDispatchGlobalContextIndex] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            &token.dispatch.global_context[0]
        ));
    return true;
}

PA_DEVICE void ResetExecutionToken(
    PA_GM ExecutionToken &token
) {
    token.control.task_id = UINT32_MAX;
    token.control.build_owner = UINT32_MAX;
    token.control.execute_owner = UINT32_MAX;
    token.control.engine_class = ExecEngineClass::None;
    token.control.payload_lines = 0;
    token.control.payload_bytes = 0;
    token.control.fanin_ready_prefix = 0;
    token.control.phase = ExecTokenPhase::Idle;
}

template <typename Ops, typename Source>
PA_DEVICE ExecBuildResult BuildAndPublishExecPayload(
    PA_GM SharedExecCell &cell, uint32_t build_owner,
    const ExecPayloadSpec &spec, const Source &source,
    PA_GM SharedExecFatalControl &fatal
) {
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecBuildResult::FatalObserved;
    }
    ExecPayloadLayout checked_layout{};
    if (!ExecOwnerValid(build_owner) ||
        !ValidateExecPayloadSpec(spec, checked_layout)) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuildInput,
            spec.task_id,
            ExecOwnerValid(build_owner) ? build_owner : 0
        );
        return ExecBuildResult::InvalidInput;
    }
    const int64_t empty_state = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Empty, 0, 0,
            ExecEngineClass::None, 0, 0
        )
    );
    const int64_t building_state = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Building, build_owner,
            kExecUnboundOwner,
            ExecEngineClass::None, 0, spec.task_id
        )
    );
    if (Ops::CompareExchange(
            &cell.control.state, empty_state, building_state
        ) != empty_state) {
        return ExecBuildResult::CellUnavailable;
    }
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecBuildResult::FatalObserved;
    }
    Ops::PreloadBuildDestination(
        &cell.payload,
        static_cast<uint64_t>(checked_layout.payload_lines) *
            kExecCacheLineBytes
    );
    ExecPayloadLayout packed_layout{};
    if (!PackExecPayload<Ops>(
            cell, spec, source, packed_layout
        ) ||
        packed_layout.payload_bytes !=
            checked_layout.payload_bytes ||
        packed_layout.payload_lines !=
            checked_layout.payload_lines) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::BuildPackFailed,
            spec.task_id, build_owner
        );
        return ExecBuildResult::InvalidInput;
    }
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecBuildResult::FatalObserved;
    }
    Ops::FlushRegion(
        &cell.payload,
        static_cast<uint64_t>(packed_layout.payload_lines) *
            kExecCacheLineBytes
    );
    // 默认实现只保留编译器边界；S2 探针会在这里注入受控延迟，
    // 证明 Flush 返回并不等价于 BUILT 已经发布。
    Ops::BeforeBuiltPublish(spec.task_id);
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecBuildResult::FatalObserved;
    }
    const int64_t built_state = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Built, build_owner,
            kExecUnboundOwner,
            spec.engine_class, packed_layout.payload_lines,
            spec.task_id
        )
    );
    if (Ops::CompareExchange(
            &cell.control.state, building_state, built_state
        ) != building_state) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::ControlPublishConflict,
            spec.task_id, build_owner
        );
        return ExecBuildResult::PublishConflict;
    }
    return ExecBuildResult::Published;
}

PA_DEVICE bool ExecEngineCompatible(
    ExecEngineClass task_engine, ExecEngineClass executor_engine
) {
    return task_engine == executor_engine;
}

template <typename Ops>
PA_DEVICE ExecClaimResult ClaimAndBindExecPayload(
    PA_GM SharedExecCell &cell, uint32_t task_id,
    uint32_t execute_owner, ExecEngineClass executor_engine,
    PA_GM ExecutionToken &token,
    PA_GM SharedExecFatalControl &fatal
) {
    if (token.control.phase != ExecTokenPhase::Idle) {
        return ExecClaimResult::TokenBusy;
    }
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecClaimResult::FatalObserved;
    }
    if (!ExecOwnerValid(execute_owner) ||
        !ExecEngineValid(executor_engine)) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuiltControl,
            task_id,
            ExecOwnerValid(execute_owner) ? execute_owner : 0
        );
        return ExecClaimResult::InvalidControl;
    }
    const int64_t observed_raw = Ops::Load(&cell.control.state);
    const DecodedExecState observed = DecodeExecState(observed_raw);
    if (!observed.valid) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuiltControl,
            task_id, execute_owner
        );
        return ExecClaimResult::InvalidControl;
    }
    if (observed.phase != ExecPhase::Built) {
        return ExecClaimResult::NotBuilt;
    }
    if (observed.task_id != task_id) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuiltControl,
            task_id, execute_owner
        );
        return ExecClaimResult::InvalidControl;
    }
    if (!ExecEngineCompatible(
            observed.engine_class, executor_engine
        )) {
        return ExecClaimResult::Incompatible;
    }
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecClaimResult::FatalObserved;
    }
    const int64_t claimed_raw = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Claimed, observed.build_owner,
            execute_owner,
            observed.engine_class, observed.payload_lines,
            observed.task_id
        )
    );
    if (Ops::CompareExchange(
            &cell.control.state, observed_raw, claimed_raw
        ) != observed_raw) {
        return ExecClaimResult::Lost;
    }

    token.control.task_id = task_id;
    token.control.build_owner = observed.build_owner;
    token.control.execute_owner = execute_owner;
    token.control.engine_class = observed.engine_class;
    token.control.payload_lines = observed.payload_lines;
    token.control.payload_bytes = 0;
    token.control.fanin_ready_prefix = 0;
    token.control.phase = ExecTokenPhase::Binding;

    if (ExecFatalPublished<Ops>(fatal)) {
        token.control.phase = ExecTokenPhase::Faulted;
        return ExecClaimResult::FatalObserved;
    }

    const uint64_t published_bytes =
        static_cast<uint64_t>(observed.payload_lines) *
        kExecCacheLineBytes;
    // CAS 返回值已经在上面的分支中被消费。这个窄 hook 只用于比较
    // “直接 Invalidate”与“额外前置 DSB”，不得在其中读取 payload。
    Ops::BeforePayloadAcquire(task_id);
    Ops::InvalidateRegion(&cell.payload, published_bytes);
    Ops::PreloadPayloadSource(&cell.payload, published_bytes);
    Ops::PreloadTokenDestination(&token.payload, published_bytes);
    const uint32_t published_words =
        observed.payload_lines * kExecHeaderWords;
    for (uint32_t word = 0;
         word < published_words; ++word) {
        Ops::StoreTokenPayloadWord(
            &token.payload.words[word],
            Ops::LoadPayloadWord(&cell.payload.words[word])
        );
    }

    ExecPayloadHeader header{};
    ExecPayloadLayout layout{};
    if (!ValidateBoundExecPayload(
            token, task_id, observed.engine_class,
            observed.payload_lines, header, layout
        )) {
        // 已取得共享所有权但 payload 不可信时必须永久 fail-closed；
        // Faulted 既阻止再次 Claim，也不满足 completion 的入口状态。
        token.control.phase = ExecTokenPhase::Faulted;
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::ClaimedPayloadInvalid,
            task_id, execute_owner
        );
        return ExecClaimResult::InvalidPayload;
    }
    token.control.payload_bytes = layout.payload_bytes;
    if (!RebuildExecutionTokenDispatchArgs<Ops>(token)) {
        token.control.phase = ExecTokenPhase::Faulted;
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidTokenPayload,
            task_id, execute_owner
        );
        return ExecClaimResult::InvalidPayload;
    }
    token.control.phase = ExecTokenPhase::WaitingFanin;
    return ExecClaimResult::Claimed;
}

PA_DEVICE ExecPayloadHeader ExecutionTokenHeader(
    PA_GM const ExecutionToken &token
) {
    return DecodeExecPayloadHeader(token.payload);
}

PA_DEVICE bool ExecutionTokenTensorWord(
    PA_GM const ExecutionToken &token, uint32_t tensor,
    uint32_t word, uint64_t &value
) {
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    if (tensor >= header.tensor_count ||
        word >= kExecTensorDescWords) {
        return false;
    }
    const uint32_t word_offset = ExecTensorPayloadWordOffset(
        tensor, header.tensor_reference_mask
    );
    if ((header.tensor_reference_mask &
         (uint32_t{1} << tensor)) != 0) {
        const uint64_t reference =
            token.payload.words[word_offset];
        if (reference == 0 ||
            reference % alignof(uint64_t) != 0) {
            return false;
        }
        PA_GM const volatile uint64_t *words =
            reinterpret_cast<PA_GM const volatile uint64_t *>(
                static_cast<uintptr_t>(reference)
            );
        value = words[word];
    } else {
        value = token.payload.words[word_offset + word];
    }
    return true;
}

PA_DEVICE bool ExecutionTokenScalar(
    PA_GM const ExecutionToken &token, uint32_t scalar,
    uint64_t &value
) {
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    if (scalar >= header.scalar_count) {
        return false;
    }
    ExecPayloadLayout layout{};
    if (!ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count,
            header.tensor_reference_mask, layout
        )) {
        return false;
    }
    value = token.payload.words[layout.scalar_word_offset + scalar];
    return true;
}

PA_DEVICE bool ExecutionTokenFanin(
    PA_GM const ExecutionToken &token, uint32_t edge,
    int32_t &producer
) {
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    if (edge >= header.fanin_count) {
        return false;
    }
    ExecPayloadLayout layout{};
    if (!ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count,
            header.tensor_reference_mask, layout
        )) {
        return false;
    }
    const uint32_t word_offset =
        layout.fanin_word_offset + edge / 2U;
    const uint64_t packed = token.payload.words[word_offset];
    producer = static_cast<int32_t>(
        edge % 2U == 0
            ? static_cast<uint32_t>(packed)
            : static_cast<uint32_t>(packed >> 32U)
    );
    return true;
}

template <typename ReadySource>
PA_DEVICE bool ExecutionTokenFaninReady(
    PA_GM ExecutionToken &token,
    const ReadySource &ready_source
) {
    if (token.control.phase != ExecTokenPhase::WaitingFanin) {
        return false;
    }
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    if (token.control.fanin_ready_prefix > header.fanin_count) {
        return false;
    }
    for (uint32_t edge = token.control.fanin_ready_prefix;
         edge < header.fanin_count; ++edge) {
        int32_t producer = -1;
        if (!ExecutionTokenFanin(token, edge, producer) ||
            producer < 0 ||
            static_cast<uint32_t>(producer) >= header.task_id ||
            !ready_source.IsReady(producer)) {
            return false;
        }
        // 已完成的前缀属于 executor-private token，可在每一项确认后
        // 立即推进；后续 poll 不再重复读取同一个 completion flag。
        token.control.fanin_ready_prefix = edge + 1U;
    }
    return true;
}

template <typename Ops, typename ReadySource>
PA_DEVICE bool TryMarkExecutionTokenEngineInflight(
    PA_GM ExecutionToken &token,
    const ReadySource &ready_source,
    PA_GM SharedExecFatalControl &fatal
) {
    // fanin 判断与状态推进保持在同一个 helper 中，调用方不能绕过
    // ready 检查直接把 WAITING_FANIN 改成 ENGINE_INFLIGHT。
    if (ExecFatalPublished<Ops>(fatal)) {
        token.control.phase = ExecTokenPhase::Faulted;
        return false;
    }
    if (!ExecutionTokenFaninReady(token, ready_source)) {
        return false;
    }
    if (ExecFatalPublished<Ops>(fatal)) {
        token.control.phase = ExecTokenPhase::Faulted;
        return false;
    }
    token.control.phase = ExecTokenPhase::EngineInflight;
    return true;
}

template <typename Ops, typename EngineCompletionSource>
PA_DEVICE bool TryMarkExecutionTokenCompleting(
    PA_GM ExecutionToken &token,
    const EngineCompletionSource &engine_completion,
    PA_GM SharedExecFatalControl &fatal
) {
    if (token.control.phase != ExecTokenPhase::EngineInflight) {
        return false;
    }
    // fatal 发生在 engine in-flight 时仍需先等真实 engine 完成，避免
    // Scalar 提前退出而遗留尚在访问 GM 的 AIC/AIV 指令流。
    if (!engine_completion.IsComplete(token)) return false;
    if (ExecFatalPublished<Ops>(fatal)) {
        token.control.phase = ExecTokenPhase::Faulted;
        return false;
    }
    token.control.phase = ExecTokenPhase::Completing;
    return true;
}

template <typename Ops, typename CompletionSink>
PA_DEVICE ExecDoneResult PublishExecDoneAfterCompletion(
    PA_GM SharedExecCell &cell, PA_GM ExecutionToken &token,
    CompletionSink &completion,
    PA_GM SharedExecFatalControl &fatal
) {
    if (token.control.phase != ExecTokenPhase::Completing &&
        token.control.phase != ExecTokenPhase::VendPublished &&
        token.control.phase != ExecTokenPhase::CompletionPublished) {
        return ExecDoneResult::TokenNotCompleting;
    }
    if (!ExecOwnerValid(token.control.execute_owner) ||
        !ExecOwnerValid(token.control.build_owner) ||
        !ExecEngineValid(token.control.engine_class) ||
        token.control.payload_lines < 1 ||
        token.control.payload_lines > kExecMaxPayloadLines) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidTokenPayload,
            token.control.task_id,
            ExecOwnerValid(token.control.execute_owner)
                ? token.control.execute_owner : 0
        );
        token.control.phase = ExecTokenPhase::Faulted;
        return ExecDoneResult::InvalidTokenPayload;
    }
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecDoneResult::FatalObserved;
    }
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    if (header.task_id != token.control.task_id ||
        header.engine_class != token.control.engine_class) {
        token.control.phase = ExecTokenPhase::Faulted;
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidTokenPayload,
            token.control.task_id, token.control.execute_owner
        );
        return ExecDoneResult::InvalidTokenPayload;
    }
    if (token.control.phase == ExecTokenPhase::Completing) {
        if (ExecFatalPublished<Ops>(fatal)) {
            return ExecDoneResult::FatalObserved;
        }
        if (!completion.PublishVend(
                header.task_id, header.completion_vend
            )) {
            (void)PublishExecFatal<Ops>(
                fatal, ExecFatalReason::CompletionPublishFailed,
                header.task_id, token.control.execute_owner
            );
            return ExecDoneResult::VendPublishFailed;
        }
        token.control.phase = ExecTokenPhase::VendPublished;
    }
    if (token.control.phase == ExecTokenPhase::VendPublished) {
        if (ExecFatalPublished<Ops>(fatal)) {
            return ExecDoneResult::FatalObserved;
        }
        if (!completion.PublishFlag(header.task_id)) {
            (void)PublishExecFatal<Ops>(
                fatal, ExecFatalReason::CompletionPublishFailed,
                header.task_id, token.control.execute_owner
            );
            return ExecDoneResult::FlagPublishFailed;
        }
        token.control.phase = ExecTokenPhase::CompletionPublished;
    }
    const int64_t claimed_raw = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Claimed, token.control.build_owner,
            token.control.execute_owner,
            token.control.engine_class,
            token.control.payload_lines,
            token.control.task_id
        )
    );
    const int64_t done_raw = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Done, token.control.build_owner,
            token.control.execute_owner,
            token.control.engine_class,
            token.control.payload_lines,
            token.control.task_id
        )
    );
    if (ExecFatalPublished<Ops>(fatal)) {
        return ExecDoneResult::FatalObserved;
    }
    if (Ops::CompareExchange(
            &cell.control.state, claimed_raw, done_raw
        ) != claimed_raw) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::CompletionStateConflict,
            header.task_id, token.control.execute_owner
        );
        return ExecDoneResult::StateConflict;
    }
    ResetExecutionToken(token);
    return ExecDoneResult::Done;
}

}  // namespace pa_scheduler::cross_core

#endif  // PA_SCHEDULER_CROSS_CORE_SHARED_EXEC_PROTOCOL_H
