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
// execution drain 按物理 block 分成 16 组；96 个 Scalar 在当前
// 1 AIC + 2 AIV/block 拓扑下每组精确包含 6 个 worker。
constexpr uint32_t kExecDrainArrivalGroups = 16;
// drain group word 复用同一次 FetchAdd 同时汇合两类单调证据：低 8 位
// 累加到达 worker 数，高位累加各 worker 已成功发布 DONE 的 kernel 数。
// 当前每组固定 6 个 worker，低位不会产生进位；完成数保持完整 56 位，
// 不需要新增 cache line 或第二次 Atomic。
constexpr uint32_t kExecDrainArrivalCountBits = 8;
constexpr uint64_t kExecDrainArrivalCountMask =
    (uint64_t{1} << kExecDrainArrivalCountBits) - 1U;

PA_DEVICE int64_t EncodeExecDrainArrivalContribution(
    uint64_t completed_tasks
) {
    if (completed_tasks >
        (static_cast<uint64_t>(INT64_MAX) >>
         kExecDrainArrivalCountBits)) {
        return -1;
    }
    return static_cast<int64_t>(
        (completed_tasks << kExecDrainArrivalCountBits) | 1U
    );
}

PA_DEVICE uint32_t DecodeExecDrainArrivalCount(int64_t raw) {
    return raw < 0
        ? UINT32_MAX
        : static_cast<uint32_t>(
              static_cast<uint64_t>(raw) &
              kExecDrainArrivalCountMask
          );
}

PA_DEVICE uint64_t DecodeExecDrainCompletionCount(int64_t raw) {
    return raw < 0
        ? UINT64_MAX
        : static_cast<uint64_t>(raw) >>
              kExecDrainArrivalCountBits;
}

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

// immutable dispatch plan 对公共执行器只发布“本 task 是否需要执行”以及
// 所需 engine。算子自己的 kind/batch/group 等身份不得被公共 scanner 或
// FinalDrain 用来反推这两项。present 位使 metadata-only task 的合法编码
// 仍不同于 host 清零后的未发布 entry。
constexpr uint8_t kExecDispatchRoutePresent = 1U << 0;
constexpr uint8_t kExecDispatchRouteExecutable = 1U << 1;
constexpr uint8_t kExecDispatchRouteEngineShift = 2;
constexpr uint8_t kExecDispatchRouteEngineMask = 0x7U;
constexpr uint8_t kExecDispatchRouteKnownMask =
    kExecDispatchRoutePresent |
    kExecDispatchRouteExecutable |
    (kExecDispatchRouteEngineMask <<
     kExecDispatchRouteEngineShift);

PA_DEVICE uint8_t EncodeExecDispatchRoute(
    bool executable, ExecEngineClass engine_class
) {
    const uint32_t engine = static_cast<uint32_t>(engine_class);
    if (engine > static_cast<uint32_t>(ExecEngineClass::Joint) ||
        (executable && engine_class == ExecEngineClass::None) ||
        (!executable && engine_class != ExecEngineClass::None)) {
        return 0;
    }
    return static_cast<uint8_t>(
        kExecDispatchRoutePresent |
        (executable ? kExecDispatchRouteExecutable : 0U) |
        (engine << kExecDispatchRouteEngineShift)
    );
}

PA_DEVICE bool DecodeExecDispatchRoute(
    uint8_t encoded, bool &executable,
    ExecEngineClass &engine_class
) {
    executable = false;
    engine_class = ExecEngineClass::None;
    if ((encoded & kExecDispatchRoutePresent) == 0 ||
        (encoded & ~kExecDispatchRouteKnownMask) != 0) {
        return false;
    }
    const uint32_t engine =
        (encoded >> kExecDispatchRouteEngineShift) &
        kExecDispatchRouteEngineMask;
    if (engine > static_cast<uint32_t>(ExecEngineClass::Joint)) {
        return false;
    }
    executable =
        (encoded & kExecDispatchRouteExecutable) != 0;
    engine_class = static_cast<ExecEngineClass>(engine);
    return executable
        ? engine_class != ExecEngineClass::None
        : engine_class == ExecEngineClass::None;
}

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

// 所有 replay actor 停产之后，再汇合本核 execution token 的排空证据。
// 96 个 worker 不再竞争同一 arrival，而是按 block%16 分到 16 条
// atomic-only line。非 root 在发布本核排空证据后即可结束；指定 root
// worker 观察每组精确 6 个到达，并核对所有组携带的完成数等于计划 kernel
// 数后最后结束。唯一 Claim CAS 保证每个 task 至多计数一次；所有 scanner
// 与 owner-local token 排空保证没有已领取任务遗留。整个 device kernel
// 由 root 的最终退出收口，不需要反向 release 广播或逐 task 原子扫描。
struct alignas(kExecCacheLineBytes) SharedExecDrainControl {
    SharedExecControl arrivals[kExecDrainArrivalGroups];
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
    // 首版 SharedExecCell 按 task-id 固定且全程不复用；Claim winner 完成
    // payload invalidate 后可直接引用这份 immutable storage。地址只由
    // execute owner 写入和消费，不参与跨核发布，也不替代 control CAS。
    uint64_t payload_address;
    // Claim 已经验证过的 immutable payload 元数据缓存在 owner-local control
    // line。A5 Scalar 间没有 cache coherence，但这些字段从写入到 Reset 都只
    // 由 execute owner 消费；因此不需要 DCCI，也不能被其他核当发布证据。
    uint64_t completion_vend;
    // low32=function_id，high32=tensor_reference_mask。
    uint64_t function_and_reference;
    // 依次打包 tensor/scalar/fanin count 与 scalar_word_offset，各占 16 bit。
    uint64_t shape_and_scalar_offset;
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
    ExecutionDispatchBinding dispatch;
};

// Claim 成功后 payload_address 始终指向对应 task-indexed
// SharedExecCell 的 immutable payload。S6.63 已证明正式路径不再向
// token 复制 payload；因此 token 只保留 owner-local control 和 dispatch
// 两个独占区域，不再为已删除的私有副本保留 68 条 cache line。
// 调用者必须在解引用前验证 payload_address 非零；Reset 后的 IDLE
// token 不得读取 payload。
PA_DEVICE PA_GM const ExecPayloadStorage &ExecutionTokenPayload(
    PA_GM const ExecutionToken &token
) {
    return *reinterpret_cast<PA_GM const ExecPayloadStorage *>(
        static_cast<uintptr_t>(token.control.payload_address)
    );
}

PA_DEVICE uint32_t ExecutionTokenFunctionId(
    PA_GM const ExecutionToken &token
) {
    return static_cast<uint32_t>(
        token.control.function_and_reference
    );
}

PA_DEVICE uint32_t ExecutionTokenTensorReferenceMask(
    PA_GM const ExecutionToken &token
) {
    return static_cast<uint32_t>(
        token.control.function_and_reference >> 32U
    );
}

PA_DEVICE uint16_t ExecutionTokenTensorCount(
    PA_GM const ExecutionToken &token
) {
    return static_cast<uint16_t>(
        token.control.shape_and_scalar_offset
    );
}

PA_DEVICE uint16_t ExecutionTokenScalarCount(
    PA_GM const ExecutionToken &token
) {
    return static_cast<uint16_t>(
        token.control.shape_and_scalar_offset >> 16U
    );
}

PA_DEVICE uint16_t ExecutionTokenFaninCount(
    PA_GM const ExecutionToken &token
) {
    return static_cast<uint16_t>(
        token.control.shape_and_scalar_offset >> 32U
    );
}

PA_DEVICE uint16_t ExecutionTokenScalarWordOffset(
    PA_GM const ExecutionToken &token
) {
    return static_cast<uint16_t>(
        token.control.shape_and_scalar_offset >> 48U
    );
}

PA_DEVICE void CacheValidatedExecPayloadMetadata(
    PA_GM ExecutionToken &token,
    const ExecPayloadHeader &header,
    const ExecPayloadLayout &layout
) {
    static_assert(
        kExecMaxPayloadWords <= UINT16_MAX,
        "token-local scalar offset must fit 16 bits"
    );
    token.control.completion_vend = header.completion_vend;
    token.control.function_and_reference =
        static_cast<uint64_t>(header.function_id) |
        (static_cast<uint64_t>(header.tensor_reference_mask) << 32U);
    token.control.shape_and_scalar_offset =
        static_cast<uint64_t>(header.tensor_count) |
        (static_cast<uint64_t>(header.scalar_count) << 16U) |
        (static_cast<uint64_t>(header.fanin_count) << 32U) |
        (static_cast<uint64_t>(layout.scalar_word_offset) << 48U);
}

// portable execution 协议不依赖 standalone 的 TraceContext/AtomicSite，
// 但正式接线必须能够在真实原语边界观察 atomic 与 DCCI。这里定义一份
// 最窄的操作适配器：隔离测试沿用 DirectExecObserver，scheduler 则传入
// 带泳道记录的 observer。这样观察逻辑不会反向污染协议数据结构，也不会
// 让无泳道构建多执行一次共享原语。
template <typename Ops>
struct DirectExecObserver {
    PA_DEVICE int64_t LoadFatal(
        PA_GM volatile int64_t *address, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::Load(address);
    }

    PA_DEVICE int64_t PublishFatal(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::CompareExchange(address, expected, desired);
    }

    PA_DEVICE int64_t LoadCellState(
        PA_GM volatile int64_t *address, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::Load(address);
    }

    PA_DEVICE int64_t ReserveBuild(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::CompareExchange(address, expected, desired);
    }

    PA_DEVICE int64_t PublishBuilt(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::CompareExchange(address, expected, desired);
    }

    PA_DEVICE int64_t ClaimCell(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::CompareExchange(address, expected, desired);
    }

    PA_DEVICE int64_t PublishDone(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::CompareExchange(address, expected, desired);
    }

    template <typename Pointer>
    PA_DEVICE void FlushBuildPayload(
        Pointer address, uint64_t bytes, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        Ops::FlushRegion(address, bytes);
    }

    template <typename Pointer>
    PA_DEVICE void InvalidateClaimPayload(
        Pointer address, uint64_t bytes, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        Ops::InvalidateRegion(address, bytes);
    }

    template <typename Pointer>
    PA_DEVICE void InvalidateTokenDescriptor(
        Pointer address, uint64_t bytes, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        Ops::InvalidateRegion(address, bytes);
    }

    template <typename Pointer>
    PA_DEVICE void InvalidateBuildDescriptor(
        Pointer address, uint64_t bytes, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        Ops::InvalidateRegion(address, bytes);
    }

    PA_DEVICE int64_t LoadFaninFlag(
        PA_GM volatile int64_t *address, uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::Load(address);
    }

    template <typename T>
    PA_DEVICE T PublishCompletionVend(
        PA_GM volatile T *address, T value,
        uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::Exchange(address, value);
    }

    PA_DEVICE int64_t PublishCompletionFlag(
        PA_GM volatile int64_t *address, int64_t value,
        uint32_t
    ) const {
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: trace-free - portable 协议单测与无观察构建的中央 observer fallback
        return Ops::Exchange(address, value);
    }
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
    sizeof(SharedExecDrainControl) ==
            kExecDrainArrivalGroups *
                kExecCacheLineBytes &&
        alignof(SharedExecDrainControl) == kExecCacheLineBytes,
    "execution drain arrival groups must own separate cache lines"
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
        offsetof(ExecutionTokenControl, payload_address) == 32 &&
        offsetof(ExecutionTokenControl, completion_vend) == 40 &&
        offsetof(ExecutionTokenControl, function_and_reference) == 48 &&
        offsetof(ExecutionTokenControl, shape_and_scalar_offset) == 56,
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
            kExecCacheLineBytes &&
        offsetof(ExecutionToken, dispatch) % kExecCacheLineBytes == 0 &&
        sizeof(ExecutionToken) ==
            kExecCacheLineBytes + kExecDispatchBindingBytes,
    "token control and dispatch binding must own disjoint lines"
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

template <typename Ops, typename Observer>
PA_DEVICE bool ExecFatalPublished(
    PA_GM SharedExecFatalControl &fatal, uint32_t task_id,
    Observer &observer
) {
    // 任意非零值都必须 fail-closed；即使记录本身已损坏，也不能继续。
    return observer.LoadFatal(&fatal.state, task_id) != 0;
}

template <typename Ops>
PA_DEVICE bool ExecFatalPublished(
    PA_GM SharedExecFatalControl &fatal
) {
    DirectExecObserver<Ops> observer{};
    return ExecFatalPublished<Ops>(fatal, 0, observer);
}

template <typename Ops, typename Observer>
PA_DEVICE bool PublishExecFatal(
    PA_GM SharedExecFatalControl &fatal,
    ExecFatalReason reason, uint32_t task_id,
    uint32_t reporter_owner, Observer &observer
) {
    if (!ExecFatalReasonValid(reason) ||
        !ExecOwnerValid(reporter_owner)) {
        return false;
    }
    const int64_t desired = static_cast<int64_t>(
        EncodeExecFatal(reason, reporter_owner, task_id)
    );
    // first-failure-wins。fatal line 永不清零，也不做 ordinary store/DCCI。
    return observer.PublishFatal(
               &fatal.state, 0, desired, task_id
           ) == 0;
}

template <typename Ops>
PA_DEVICE bool PublishExecFatal(
    PA_GM SharedExecFatalControl &fatal,
    ExecFatalReason reason, uint32_t task_id,
    uint32_t reporter_owner
) {
    DirectExecObserver<Ops> observer{};
    return PublishExecFatal<Ops>(
        fatal, reason, task_id, reporter_owner, observer
    );
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
    // 高 32 位保留并固定为 0。公共 execution payload 只携带
    // 当前逻辑 task-id；output/completion 的索引方式由算子适配层
    // 和 completion sink 解释，不在通用 header 中隐式增加第二个身份。
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
    if (token.control.payload_address == 0) {
        return false;
    }
    PA_GM const ExecPayloadStorage &payload =
        ExecutionTokenPayload(token);
    header = DecodeExecPayloadHeader(payload);
    if ((payload.words[0] >> 32U) != 0 ||
        (payload.words[6] >> 32U) != 0 ||
        payload.words[7] != 0 ||
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
        const uint64_t reference = payload.words[
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

template <typename Ops, typename Observer>
PA_DEVICE bool RebuildExecutionTokenDispatchArgsFromValidatedPayload(
    PA_GM ExecutionToken &token,
    const ExecPayloadHeader &header,
    const ExecPayloadLayout &layout,
    Observer &observer
) {
    if (token.control.payload_address == 0) {
        return false;
    }
    PA_GM const ExecPayloadStorage &payload =
        ExecutionTokenPayload(token);
    if ((token.control.phase != ExecTokenPhase::Binding &&
         token.control.phase != ExecTokenPhase::WaitingFanin) ||
        header.payload_bytes != layout.payload_bytes ||
        token.control.payload_bytes != layout.payload_bytes ||
        token.control.payload_lines != layout.payload_lines) {
        return false;
    }

    // inline descriptor 已位于本轮不复用的 immutable shared payload，可
    // 直接保留稳定地址；adapter 明确声明的外部 GM descriptor 同样保留
    // 绝对地址。A5 Scalar 间没有 cache coherence，executor 在第一次使用
    // 外部引用前必须 invalidate 对应 128B，不能依赖 builder 的 DCache。
    for (uint32_t tensor = 0;
         tensor < header.tensor_count; ++tensor) {
        const uint32_t word_offset = ExecTensorPayloadWordOffset(
            tensor, header.tensor_reference_mask
        );
        if ((header.tensor_reference_mask &
             (uint32_t{1} << tensor)) != 0) {
            const uint64_t reference =
                payload.words[word_offset];
            PA_GM const void *descriptor =
                reinterpret_cast<PA_GM const void *>(
                    static_cast<uintptr_t>(reference)
                );
            observer.InvalidateTokenDescriptor(
                descriptor, kExecTensorDescBytes,
                token.control.task_id
            );
            token.dispatch.args[tensor] = reference;
        } else {
            token.dispatch.args[tensor] = static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(
                    &payload.words[word_offset]
                )
            );
        }
    }
    for (uint32_t scalar = 0;
         scalar < header.scalar_count; ++scalar) {
        token.dispatch.args[header.tensor_count + scalar] =
            payload.words[layout.scalar_word_offset + scalar];
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

template <typename Ops, typename Observer>
PA_DEVICE bool RebuildExecutionTokenDispatchArgs(
    PA_GM ExecutionToken &token, Observer &observer
) {
    if (token.control.payload_address == 0) {
        return false;
    }
    PA_GM const ExecPayloadStorage &payload =
        ExecutionTokenPayload(token);
    const ExecPayloadHeader header =
        DecodeExecPayloadHeader(payload);
    ExecPayloadLayout layout{};
    return ComputeExecPayloadLayout(
               header.tensor_count, header.scalar_count,
               header.fanin_count,
               header.tensor_reference_mask, layout
           ) &&
           RebuildExecutionTokenDispatchArgsFromValidatedPayload<Ops>(
               token, header, layout, observer
           );
}

template <typename Ops>
PA_DEVICE bool RebuildExecutionTokenDispatchArgs(
    PA_GM ExecutionToken &token
) {
    DirectExecObserver<Ops> observer{};
    return RebuildExecutionTokenDispatchArgs<Ops>(token, observer);
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
    token.control.payload_address = 0;
    token.control.completion_vend = 0;
    token.control.function_and_reference = 0;
    token.control.shape_and_scalar_offset = 0;
    token.control.phase = ExecTokenPhase::Idle;
}

template <typename Ops, typename Source, typename Observer>
PA_DEVICE ExecBuildResult BuildAndPublishExecPayload(
    PA_GM SharedExecCell &cell, uint32_t build_owner,
    const ExecPayloadSpec &spec, const Source &source,
    PA_GM SharedExecFatalControl &fatal, Observer &observer
) {
    // exec_fatal 只保存 execution 协议的精确首错，不再充当第二条全局
    // 停止线。生产调用方在进入 Build 前检查 scheduler fatal；本 helper
    // 发现错误后发布精确原因并把失败结果返回给调用方，由调用方同步置
    // scheduler fatal。正常成功路径不需要再读取原因记录。
    ExecPayloadLayout checked_layout{};
    if (!ExecOwnerValid(build_owner) ||
        !ValidateExecPayloadSpec(spec, checked_layout)) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuildInput,
            spec.task_id,
            ExecOwnerValid(build_owner) ? build_owner : 0,
            observer
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
    if (observer.ReserveBuild(
            &cell.control.state, empty_state, building_state,
            spec.task_id
        ) != empty_state) {
        return ExecBuildResult::CellUnavailable;
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
            spec.task_id, build_owner, observer
        );
        return ExecBuildResult::InvalidInput;
    }
    observer.FlushBuildPayload(
        &cell.payload,
        static_cast<uint64_t>(packed_layout.payload_lines) *
            kExecCacheLineBytes,
        spec.task_id
    );
    // 默认实现只保留编译器边界；S2 探针会在这里注入受控延迟，
    // 证明 Flush 返回并不等价于 BUILT 已经发布。
    Ops::BeforeBuiltPublish(spec.task_id);
    // 性能优先合同：Build 开始后不再轮询全局错误。并发 fatal 即使发生在
    // reserve、Pack 或 flush 期间，也允许当前 task 完成 BUILT 发布；
    // executor 在 Claim 和执行前检查 fatal，FinalDrain 负责最终退出。
    const int64_t built_state = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Built, build_owner,
            kExecUnboundOwner,
            spec.engine_class, packed_layout.payload_lines,
            spec.task_id
        )
    );
    if (observer.PublishBuilt(
            &cell.control.state, building_state, built_state,
            spec.task_id
        ) != building_state) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::ControlPublishConflict,
            spec.task_id, build_owner, observer
        );
        return ExecBuildResult::PublishConflict;
    }
    return ExecBuildResult::Published;
}

template <typename Ops, typename Source>
PA_DEVICE ExecBuildResult BuildAndPublishExecPayload(
    PA_GM SharedExecCell &cell, uint32_t build_owner,
    const ExecPayloadSpec &spec, const Source &source,
    PA_GM SharedExecFatalControl &fatal
) {
    DirectExecObserver<Ops> observer{};
    return BuildAndPublishExecPayload<Ops>(
        cell, build_owner, spec, source, fatal, observer
    );
}

PA_DEVICE bool ExecEngineCompatible(
    ExecEngineClass task_engine, ExecEngineClass executor_engine
) {
    return task_engine == executor_engine;
}

template <typename Ops, typename Observer>
PA_DEVICE ExecClaimResult ClaimAndBindObservedExecPayload(
    PA_GM SharedExecCell &cell, int64_t observed_raw,
    uint32_t task_id,
    uint32_t execute_owner, ExecEngineClass executor_engine,
    PA_GM ExecutionToken &token,
    PA_GM SharedExecFatalControl &fatal, Observer &observer
) {
    if (token.control.phase != ExecTokenPhase::Idle) {
        return ExecClaimResult::TokenBusy;
    }
    if (!ExecOwnerValid(execute_owner) ||
        !ExecEngineValid(executor_engine)) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuiltControl,
            task_id,
            ExecOwnerValid(execute_owner) ? execute_owner : 0,
            observer
        );
        return ExecClaimResult::InvalidControl;
    }
    const DecodedExecState observed = DecodeExecState(observed_raw);
    if (!observed.valid) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuiltControl,
            task_id, execute_owner, observer
        );
        return ExecClaimResult::InvalidControl;
    }
    if (observed.phase != ExecPhase::Built) {
        return ExecClaimResult::NotBuilt;
    }
    if (observed.task_id != task_id) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuiltControl,
            task_id, execute_owner, observer
        );
        return ExecClaimResult::InvalidControl;
    }
    if (!ExecEngineCompatible(
            observed.engine_class, executor_engine
        )) {
        return ExecClaimResult::Incompatible;
    }
    const int64_t claimed_raw = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Claimed, observed.build_owner,
            execute_owner,
            observed.engine_class, observed.payload_lines,
            observed.task_id
        )
    );
    if (observer.ClaimCell(
            &cell.control.state, observed_raw, claimed_raw,
            task_id
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
    token.control.payload_address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(&cell.payload)
    );
    token.control.phase = ExecTokenPhase::Binding;

    const uint64_t published_bytes =
        static_cast<uint64_t>(observed.payload_lines) *
        kExecCacheLineBytes;
    // CAS 返回值已经在上面的分支中被消费。这个窄 hook 只用于比较
    // “直接 Invalidate”与“额外前置 DSB”，不得在其中读取 payload。
    Ops::BeforePayloadAcquire(task_id);
    observer.InvalidateClaimPayload(
        &cell.payload, published_bytes, task_id
    );
    Ops::PreloadPayloadSource(&cell.payload, published_bytes);
    // SharedExecCell 由 task-id 唯一索引，当前一轮运行内既不回收也不复用；
    // BUILT 发布后没有 ordinary writer。唯一 Claim winner invalidate 后直接
    // 使用这份 immutable payload，只把可变 phase/fanin 前缀留在 owner-local
    // token，避免再向 GM token 复制 10~16 条 cacheline。

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
            task_id, execute_owner, observer
        );
        return ExecClaimResult::InvalidPayload;
    }
    token.control.payload_bytes = layout.payload_bytes;
    // ValidateBoundExecPayload 已从 immutable volatile payload 取得完整 header
    // 并计算 layout；同一 Claim 边界直接复用这份本地证据，避免再次读取首行
    // 和重复执行布局计算。BUILT 后没有 ordinary writer，因此无需二次确认。
    if (!RebuildExecutionTokenDispatchArgsFromValidatedPayload<Ops>(
            token, header, layout, observer
        )) {
        token.control.phase = ExecTokenPhase::Faulted;
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidTokenPayload,
            task_id, execute_owner, observer
        );
        return ExecClaimResult::InvalidPayload;
    }
    CacheValidatedExecPayloadMetadata(token, header, layout);
    token.control.phase = ExecTokenPhase::WaitingFanin;
    return ExecClaimResult::Claimed;
}

template <typename Ops, typename Observer>
PA_DEVICE ExecClaimResult ClaimAndBindExecPayload(
    PA_GM SharedExecCell &cell, uint32_t task_id,
    uint32_t execute_owner, ExecEngineClass executor_engine,
    PA_GM ExecutionToken &token,
    PA_GM SharedExecFatalControl &fatal, Observer &observer
) {
    // 忙 token 和非法 owner 必须在共享读取前拒绝，保持调用方可以用
    // “无 shared operation”识别本地入口错误。正常路径读取一次完整
    // control 快照，随后由 CAS 负责验证该快照是否仍是线性化前状态。
    if (token.control.phase != ExecTokenPhase::Idle) {
        return ExecClaimResult::TokenBusy;
    }
    if (!ExecOwnerValid(execute_owner) ||
        !ExecEngineValid(executor_engine)) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::InvalidBuiltControl,
            task_id,
            ExecOwnerValid(execute_owner) ? execute_owner : 0,
            observer
        );
        return ExecClaimResult::InvalidControl;
    }
    return ClaimAndBindObservedExecPayload<Ops>(
        cell,
        observer.LoadCellState(&cell.control.state, task_id),
        task_id, execute_owner, executor_engine,
        token, fatal, observer
    );
}

template <typename Ops>
PA_DEVICE ExecClaimResult ClaimAndBindExecPayload(
    PA_GM SharedExecCell &cell, uint32_t task_id,
    uint32_t execute_owner, ExecEngineClass executor_engine,
    PA_GM ExecutionToken &token,
    PA_GM SharedExecFatalControl &fatal
) {
    DirectExecObserver<Ops> observer{};
    return ClaimAndBindExecPayload<Ops>(
        cell, task_id, execute_owner, executor_engine,
        token, fatal, observer
    );
}

PA_DEVICE ExecPayloadHeader ExecutionTokenHeader(
    PA_GM const ExecutionToken &token
) {
    if (token.control.payload_address == 0) {
        return ExecPayloadHeader{};
    }
    return DecodeExecPayloadHeader(ExecutionTokenPayload(token));
}

PA_DEVICE bool ExecutionTokenTensorWord(
    PA_GM const ExecutionToken &token, uint32_t tensor,
    uint32_t word, uint64_t &value
) {
    if (token.control.payload_address == 0) {
        return false;
    }
    PA_GM const ExecPayloadStorage &payload =
        ExecutionTokenPayload(token);
    const uint32_t tensor_count =
        ExecutionTokenTensorCount(token);
    const uint32_t tensor_reference_mask =
        ExecutionTokenTensorReferenceMask(token);
    if (tensor >= tensor_count ||
        word >= kExecTensorDescWords) {
        return false;
    }
    const uint32_t word_offset = ExecTensorPayloadWordOffset(
        tensor, tensor_reference_mask
    );
    if ((tensor_reference_mask &
         (uint32_t{1} << tensor)) != 0) {
        const uint64_t reference =
            payload.words[word_offset];
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
        value = payload.words[word_offset + word];
    }
    return true;
}

PA_DEVICE bool ExecutionTokenScalar(
    PA_GM const ExecutionToken &token, uint32_t scalar,
    uint64_t &value
) {
    if (token.control.payload_address == 0) {
        return false;
    }
    PA_GM const ExecPayloadStorage &payload =
        ExecutionTokenPayload(token);
    const uint32_t scalar_count =
        ExecutionTokenScalarCount(token);
    const uint32_t word_offset =
        ExecutionTokenScalarWordOffset(token) + scalar;
    if (scalar >= scalar_count ||
        word_offset >=
            token.control.payload_lines * kExecHeaderWords) {
        return false;
    }
    value = payload.words[word_offset];
    return true;
}

PA_DEVICE bool ExecutionTokenFanin(
    PA_GM const ExecutionToken &token, uint32_t edge,
    int32_t &producer
) {
    if (token.control.payload_address == 0) {
        return false;
    }
    PA_GM const ExecPayloadStorage &payload =
        ExecutionTokenPayload(token);
    const uint32_t fanin_count =
        ExecutionTokenFaninCount(token);
    const uint32_t fanin_word_offset =
        ExecutionTokenScalarWordOffset(token) +
        ExecutionTokenScalarCount(token);
    const uint32_t word_offset =
        fanin_word_offset + edge / 2U;
    if (edge >= fanin_count ||
        word_offset >=
            token.control.payload_lines * kExecHeaderWords) {
        return false;
    }
    const uint64_t packed = payload.words[word_offset];
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
    const uint32_t fanin_count =
        ExecutionTokenFaninCount(token);
    const uint32_t task_id = token.control.task_id;
    if (token.control.fanin_ready_prefix > fanin_count) {
        return false;
    }
    for (uint32_t edge = token.control.fanin_ready_prefix;
         edge < fanin_count; ++edge) {
        int32_t producer = -1;
        if (!ExecutionTokenFanin(token, edge, producer) ||
            producer < 0 ||
            static_cast<uint32_t>(producer) >= task_id ||
            !ready_source.IsReady(producer)) {
            return false;
        }
        // 已完成的前缀属于 executor-private token，可在每一项确认后
        // 立即推进；后续 poll 不再重复读取同一个 completion flag。
        token.control.fanin_ready_prefix = edge + 1U;
    }
    return true;
}

template <typename Ops, typename ReadySource, typename Observer>
PA_DEVICE bool TryMarkExecutionTokenEngineInflight(
    PA_GM ExecutionToken &token,
    const ReadySource &ready_source,
    PA_GM SharedExecFatalControl &fatal, Observer &observer
) {
    // fanin 判断与状态推进保持在同一个 helper 中，调用方不能绕过
    // ready 检查直接把 WAITING_FANIN 改成 ENGINE_INFLIGHT。
    (void)fatal;
    (void)observer;
    if (!ExecutionTokenFaninReady(token, ready_source)) {
        return false;
    }
    token.control.phase = ExecTokenPhase::EngineInflight;
    return true;
}

template <typename Ops, typename ReadySource>
PA_DEVICE bool TryMarkExecutionTokenEngineInflight(
    PA_GM ExecutionToken &token,
    const ReadySource &ready_source,
    PA_GM SharedExecFatalControl &fatal
) {
    DirectExecObserver<Ops> observer{};
    return TryMarkExecutionTokenEngineInflight<Ops>(
        token, ready_source, fatal, observer
    );
}

template <typename Ops, typename EngineCompletionSource,
          typename Observer>
PA_DEVICE bool TryMarkExecutionTokenCompleting(
    PA_GM ExecutionToken &token,
    const EngineCompletionSource &engine_completion,
    PA_GM SharedExecFatalControl &fatal, Observer &observer
) {
    if (token.control.phase != ExecTokenPhase::EngineInflight) {
        return false;
    }
    // fatal 发生在 engine in-flight 时仍需先等真实 engine 完成，避免
    // Scalar 提前退出而遗留尚在访问 GM 的 AIC/AIV 指令流。
    if (!engine_completion.IsComplete(token)) return false;
    (void)fatal;
    (void)observer;
    token.control.phase = ExecTokenPhase::Completing;
    return true;
}

template <typename Ops, typename EngineCompletionSource>
PA_DEVICE bool TryMarkExecutionTokenCompleting(
    PA_GM ExecutionToken &token,
    const EngineCompletionSource &engine_completion,
    PA_GM SharedExecFatalControl &fatal
) {
    DirectExecObserver<Ops> observer{};
    return TryMarkExecutionTokenCompleting<Ops>(
        token, engine_completion, fatal, observer
    );
}

template <typename Ops, typename CompletionSink, typename Observer>
PA_DEVICE ExecDoneResult PublishExecDoneAfterCompletion(
    PA_GM SharedExecCell &cell, PA_GM ExecutionToken &token,
    CompletionSink &completion,
    PA_GM SharedExecFatalControl &fatal, Observer &observer
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
                ? token.control.execute_owner : 0,
            observer
        );
        token.control.phase = ExecTokenPhase::Faulted;
        return ExecDoneResult::InvalidTokenPayload;
    }
    // payload 的 immutable header 已在 Claim 成功前完整校验，并把完成
    // 所需的 vend 留在 execute owner 的 control line。完成路径不再为了
    // 重取 task/vend 再访问 shared payload；该缓存不承担跨核发布语义。
    const uint32_t task_id = token.control.task_id;
    const uint64_t completion_vend =
        token.control.completion_vend;
    if (token.control.phase == ExecTokenPhase::Completing) {
        if (!completion.PublishVend(
                task_id, completion_vend
            )) {
            (void)PublishExecFatal<Ops>(
                fatal, ExecFatalReason::CompletionPublishFailed,
                task_id, token.control.execute_owner,
                observer
            );
            return ExecDoneResult::VendPublishFailed;
        }
        token.control.phase = ExecTokenPhase::VendPublished;
    }
    if (token.control.phase == ExecTokenPhase::VendPublished) {
        if (!completion.PublishFlag(task_id)) {
            (void)PublishExecFatal<Ops>(
                fatal, ExecFatalReason::CompletionPublishFailed,
                task_id, token.control.execute_owner,
                observer
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
    if (observer.PublishDone(
            &cell.control.state, claimed_raw, done_raw,
            task_id
        ) != claimed_raw) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::CompletionStateConflict,
            task_id, token.control.execute_owner,
            observer
        );
        return ExecDoneResult::StateConflict;
    }
    ResetExecutionToken(token);
    return ExecDoneResult::Done;
}

template <typename Ops, typename CompletionSink>
PA_DEVICE ExecDoneResult PublishExecDoneAfterCompletion(
    PA_GM SharedExecCell &cell, PA_GM ExecutionToken &token,
    CompletionSink &completion,
    PA_GM SharedExecFatalControl &fatal
) {
    DirectExecObserver<Ops> observer{};
    return PublishExecDoneAfterCompletion<Ops>(
        cell, token, completion, fatal, observer
    );
}

}  // namespace pa_scheduler::cross_core

#endif  // PA_SCHEDULER_CROSS_CORE_SHARED_EXEC_PROTOCOL_H
