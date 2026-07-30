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

#ifndef PA_SCHEDULER_COMMON_PA_MODEL_H
#define PA_SCHEDULER_COMMON_PA_MODEL_H

#include <stddef.h>
#include <stdint.h>

// standalone 与真实 FDWIC 使用同一个模式宏，便于先在这里验证模式边界，
// 再把已证明的机制迁移到 runtime。构建脚本始终显式传 0/1；头文件默认
// private，保证原有命令和既有性能基线不变。
#ifndef PTO_FDWIC_SHARED_MAP
#define PTO_FDWIC_SHARED_MAP 0
#endif

#if PTO_FDWIC_SHARED_MAP != 0 && PTO_FDWIC_SHARED_MAP != 1
#error "PTO_FDWIC_SHARED_MAP must be 0 (private) or 1 (shared)"
#endif

// standalone 的正式产物仍固定使用已验证的 CAP=128。隔离 ring 门槛会用
// 同一份生产 helper 重编译多个 CAP，证明“每桶连续环”没有偷写 128。
// 这里故意采用构建期常量：hash、slot mask 与桶跨度都可被 CCEC 常量折叠；
// 运行期 auto/覆盖参数还需要静态任务图 planner，不能在本阶段冒充完成。
#ifndef PTO_FDWIC_TENSORMAP_RING_CAP
#define PTO_FDWIC_TENSORMAP_RING_CAP 128
#endif

// shared writer 插入仍是一条全局 task-id 顺序链，只把相邻 token 交错
// 放到 1/2/4/8/16/32/64/128 条独立 cache line，分散 future owner
// 的等待 load。
// 该值是构建身份，不允许运行期改变；private 构建不会读取这些控制字。
#ifndef PTO_FDWIC_SHARED_INSERT_TURN_GROUPS
#define PTO_FDWIC_SHARED_INSERT_TURN_GROUPS 1
#endif

#if PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 1 && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 2 && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 4 && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 8 && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 16 && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 32 && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 64 && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 128
#error "PTO_FDWIC_SHARED_INSERT_TURN_GROUPS must be a power of two from 1 through 128"
#endif

#if !PTO_FDWIC_SHARED_MAP && \
    PTO_FDWIC_SHARED_INSERT_TURN_GROUPS != 1
#error "shared insert-turn groups only apply to shared TensorMap builds"
#endif

// S0 的 fail-closed 门禁在 S2 接入真实 shared sidecar 后解除。模式仍由
// 三镜像统一的构建身份与 manifest 锁定，不能把 shared 目录指向 private 实现。

// 三类证据链在编译期严格互斥：swimlane 保存普通阶段与 atomic 记录，
// submit-pmu 只保留 PMU 窗口，perf-clock 则只增加首个/末个 Submit
// 两个性能时间边界。未显式传宏的既有后端继续使用原有通用实现。
#ifndef PA_BUILD_SWIMLANE
#define PA_BUILD_SWIMLANE 0
#endif

#ifndef PA_BUILD_SUBMIT_PMU
#define PA_BUILD_SUBMIT_PMU 0
#endif

#ifndef PA_BUILD_PERF_CLOCK
#define PA_BUILD_PERF_CLOCK 0
#endif

#ifndef PA_BUILD_ATOMIC_SWIMLANE
#define PA_BUILD_ATOMIC_SWIMLANE 0
#endif

#ifndef PA_BUILD_COMPACT_GENERIC_TRACE
#define PA_BUILD_COMPACT_GENERIC_TRACE 0
#endif

#if PA_BUILD_COMPACT_GENERIC_TRACE != 0 && \
    PA_BUILD_COMPACT_GENERIC_TRACE != 1
#error "PA_BUILD_COMPACT_GENERIC_TRACE must be 0 or 1"
#endif

#if (PA_BUILD_SWIMLANE + PA_BUILD_SUBMIT_PMU + PA_BUILD_PERF_CLOCK) > 1
#error "swimlane, submit-pmu, and perf-clock builds are mutually exclusive"
#endif

#if PA_BUILD_ATOMIC_SWIMLANE && !PA_BUILD_SWIMLANE
#error "the compile-time atomic trace specialization requires a swimlane build"
#endif

#if PA_BUILD_COMPACT_GENERIC_TRACE && \
    (!PTO_FDWIC_SHARED_MAP || !PA_BUILD_SWIMLANE || \
     !PA_BUILD_ATOMIC_SWIMLANE)
#error "compact generic trace is restricted to shared full-swimlane builds"
#endif

#if PA_BUILD_COMPACT_GENERIC_TRACE && \
    PTO_FDWIC_TENSORMAP_RING_CAP != 128
#error "compact generic trace currently supports only the production CAP=128 ABI"
#endif

// submit-pmu 与 perf-clock 都不允许把阶段泳道、atomic 包围计时或
// phase-profile 模板带入最终 ELF。统一谓词避免各 helper 对“无 trace”
// 的理解逐渐分叉；它不代表 PMU 已开启。
#define PA_BUILD_TRACE_FREE (PA_BUILD_SUBMIT_PMU || PA_BUILD_PERF_CLOCK)

namespace pa_scheduler {

enum class TensorMapBuildMode : uint32_t {
    Private = 0,
    Shared = 1,
};

constexpr TensorMapBuildMode kCompiledTensorMapMode =
    static_cast<TensorMapBuildMode>(PTO_FDWIC_SHARED_MAP);
constexpr uint32_t kBuildIdentityMagic = 0x50414249U;  // "PABI"
constexpr uint32_t kBuildIdentityCompactGenericTraceBit =
    1U << 31U;
#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kBuildIdentityAbiGeneration = 13;
#else
constexpr uint32_t kBuildIdentityAbiGeneration = 4;
#endif
// 默认 CAP=128 时，private 保留历史 ABI 值；shared generation 12 另把
// active insert-turn G 编入低位。这样既避免 private AIC/AIV 入口因身份
// 元数据多一条大立即数构造，也让 manifest v4 和 host/device 握手共同
// 拒绝不同 G 的 shared 混件。非默认隔离变体继续把 CAP 编进 ABI。
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kBuildIdentityAbiVersion =
    (kBuildIdentityAbiGeneration << 8U) |
    static_cast<uint32_t>(
        PTO_FDWIC_SHARED_INSERT_TURN_GROUPS
    ) |
    (PA_BUILD_COMPACT_GENERIC_TRACE
         ? kBuildIdentityCompactGenericTraceBit
         : 0U);
#else
constexpr uint32_t kBuildIdentityAbiVersion =
    kBuildIdentityAbiGeneration;
#endif
#else
#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kBuildIdentityAbiVersion =
    (kBuildIdentityAbiGeneration << 24U) |
    (static_cast<uint32_t>(
         PTO_FDWIC_SHARED_INSERT_TURN_GROUPS
     ) << 16U) |
    static_cast<uint32_t>(PTO_FDWIC_TENSORMAP_RING_CAP);
#else
constexpr uint32_t kBuildIdentityAbiVersion =
    (kBuildIdentityAbiGeneration << 16U) |
    static_cast<uint32_t>(PTO_FDWIC_TENSORMAP_RING_CAP);
#endif
#endif
static_assert(
    (kBuildIdentityAbiVersion &
     kBuildIdentityCompactGenericTraceBit) ==
        (PA_BUILD_COMPACT_GENERIC_TRACE
             ? kBuildIdentityCompactGenericTraceBit
             : 0U),
    "build identity does not encode the compact trace format"
);

// private 与现有 shared 单组 Case1 每 batch 都回放五个 task。shared
// 多组请求复用一次 Alloc，随后每个 block group 增加 QK/SF/PV/UP 四
// task；ticket 的两个 group bit 与 PA 256/64=4 组上限一致。实际组数
// 仍由每个 batch 的 context_len 决定，不是编译期固定为四组。
constexpr uint32_t kDefaultBatches = 256;
#if PTO_FDWIC_SHARED_MAP
// shared standalone 扩展到 512 batch，用于保持每 batch 的默认 PA-G1
// 业务不变并把总 task 从 1,280 增至 2,560。shared 多 group 仍受独立
// kMaxTasks 总容量约束，不能把 batch 上限误解为任意 context 都可达；
// private 继续保持原有 256-batch ABI 和测试边界。
constexpr uint32_t kMaxBatches = 512;
#else
constexpr uint32_t kMaxBatches = kDefaultBatches;
#endif
constexpr uint32_t kTasksPerBatch = 5;
#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kSharedPaMaxBlockGroups = 4;
constexpr uint32_t kSharedPaMaxTasksPerBatch =
    1U + 4U * kSharedPaMaxBlockGroups;
// 保留现有 4,352-task output/history 物理布局；它既覆盖原 256 batch
// 的 PA-G4 最坏计划，也覆盖新增 512 batch 的默认 PA-G1 计划。
// 512 batch 的多 group 计划若超过该总量，会由 host/device plan
// 在触碰共享状态前 fail closed。
constexpr uint32_t kMaxTasks =
    kDefaultBatches * kSharedPaMaxTasksPerBatch;
static_assert(
    kSharedPaMaxTasksPerBatch == 17 && kMaxTasks == 4352 &&
        kMaxTasks >= kMaxBatches * kTasksPerBatch,
    "shared PA task capacity no longer covers PA-G4/B256 and PA-G1/B512"
);
#else
constexpr uint32_t kMaxTasks = kMaxBatches * kTasksPerBatch;
#endif
constexpr uint32_t kTaskCellCapacity = 1U << 16;

constexpr uint32_t kAicWorkers = 32;
constexpr uint32_t kAivWorkers = 64;
constexpr uint32_t kWorkers = kAicWorkers + kAivWorkers;
constexpr uint32_t kRuntimeMaxWorkers = 108;
constexpr uint32_t kCursorShards = 4;
// S4.14a 已建立 shared Vector 四分片迁址对照；S4.14b 继续使用相同的
// sidecar 地址、物理容量和代码骨架，只启用此前预留的后四条物理线。
// device 热路径仍使用同一取模表达式，唯一数值变量是 active shards
// 从4改8，以便单独归因分片收益。
constexpr uint32_t kSharedVectorCursorCapacity = 8;
constexpr uint32_t kSharedVectorCursorShards = 8;
constexpr uint32_t kSharedVectorCursorShardMask =
    kSharedVectorCursorShards - 1;
static_assert(
    (kSharedVectorCursorShards & kSharedVectorCursorShardMask) == 0,
    "shared Vector cursor shards must be a power of two"
);
static_assert(
    kSharedVectorCursorShards <= kSharedVectorCursorCapacity,
    "active shared Vector shards exceed physical capacity"
);
// shared 输出 heap 按 task_id 固定分成 8 个物理 shard。首版只做有界
// 绝对递增分配，不在 shard 内回绕；该常量同时属于 host 地址 oracle。
constexpr uint32_t kSharedHeapShards = 8;
constexpr uint32_t kSharedInsertTurnCapacity = 128;
constexpr uint32_t kSharedInsertTurnGroups =
    static_cast<uint32_t>(
        PTO_FDWIC_SHARED_INSERT_TURN_GROUPS
    );
constexpr uint32_t kSharedInsertTurnMask =
    kSharedInsertTurnGroups - 1U;
static_assert(
    (kSharedInsertTurnGroups & kSharedInsertTurnMask) == 0 &&
        kSharedInsertTurnGroups <= kSharedInsertTurnCapacity,
    "shared insert-turn groups must be a supported power of two"
);
constexpr uint32_t kFinalBarrierMaxLeafGroups = 16;
constexpr uint32_t kFinalBarrierMaxMiddleGroups = 4;
// 每个 worker 私有 ring 有 4 个物理 slot，其中 2 个为 BlockWon 协议预留；
// 单 lane Case1 虽不进入 BlockWon，普通 kernel 仍只能占用剩余 2 个 slot。
constexpr uint32_t kPrivateSlots = 4;
constexpr uint32_t kWonReserve = 2;
constexpr uint32_t kUsableSlots = kPrivateSlots - kWonReserve;
constexpr uint32_t kMaxFanin = 16;
// H=64 同时约束 heap 可回收 frontier 和 TensorMap producer 的存活下界。
// heap_next 使用单调逻辑地址；真正落到 256 MiB 环形 heap 时才取模，因而可判断覆盖风险。
constexpr uint32_t kHeapWindow = 64;
constexpr uint64_t kHeapBytes = 256ULL << 20;
// B512/PA-G1 的逻辑 reservation 为 B256 的两倍，超过默认 256 MiB
// no-wrap heap。standalone 只在 batches>kDefaultBatches 时使用该扩展
// 逻辑容量；real-compute 仍访问独立 workspace，不分配或解引用这段
// synthetic heap。
constexpr uint64_t kExtendedBatchHeapBytes = 512ULL << 20;
constexpr uint64_t kSyntheticHeapBase = 0x100000000ULL;
constexpr uint64_t kOutputAlignment = 1024;
constexpr uint32_t kMaxTensorDims = 5;
constexpr uint32_t kMaxTaskTensors = 32;
constexpr uint32_t kMaxTaskScalars = 16;
constexpr uint32_t kSharedOutputMaxPerTask = 8;
constexpr uint32_t kPayloadSlots = 2048;
constexpr uint32_t kPayloadMask = kPayloadSlots - 1;
constexpr uint32_t kPayloadStride = 4096;
// private/shared 统一为 ring-per-bucket。物理槽总数继续固定为旧 map 的
// 16K；构建期 CAP 决定每桶连续槽数，桶数由 16K/CAP 推导。正式默认仍是
// 128×128，隔离门槛另外覆盖 32×512、256×64 与 16384×1 等形态。
// CAP 变大意味着桶更少、单桶扫描更长；CAP 变小则更易触发显式满环。
constexpr uint32_t kMapCapacity = 16384;
constexpr uint32_t kMapBucketCapacity =
    static_cast<uint32_t>(PTO_FDWIC_TENSORMAP_RING_CAP);
constexpr uint32_t kMapBuckets = kMapCapacity / kMapBucketCapacity;

constexpr uint32_t ConstexprLog2(uint32_t value) {
    return value <= 1U ? 0U : 1U + ConstexprLog2(value >> 1U);
}

constexpr uint32_t kMapBucketShift = ConstexprLog2(kMapBuckets);
constexpr uint32_t kMapBucketMask = kMapBuckets - 1;
constexpr uint32_t kMapBucketSlotMask = kMapBucketCapacity - 1;
constexpr uint32_t kDefaultMapBucketCapacity = 128;
constexpr uint32_t kPaCase1MapEntriesPerBatch = 4;
constexpr uint32_t kPaCase1MaxLiveMapBatches =
    (kHeapWindow + 1 + kTasksPerBatch - 1) / kTasksPerBatch;
constexpr uint32_t kPaCase1MaxLiveMapEntries =
    kPaCase1MapEntriesPerBatch * kPaCase1MaxLiveMapBatches;
constexpr uint32_t kTaskWindow = 1 << 10;
constexpr uint32_t kTaskWindowMask = kTaskWindow - 1;
constexpr uint64_t kSystemCounterHz = 1000000000ULL;
constexpr uint64_t kWatchdogTicks = 2 * kSystemCounterHz;
// trace_enabled 是位图而不是 bool：bit0 保持既有阶段泳道，bit1 额外开启
// 逐条 atomic 源码括号记录。atomic 记录依赖同一份 trace buffer，因此 bit1
// 只能与 bit0 一起配置。
constexpr uint32_t kTracePhasesEnabled = 1U << 0;
constexpr uint32_t kTraceAtomicsEnabled = 1U << 1;
// Claim trace flags 是独立 raw ABI：bit0 表示获胜，bit1 表示已经通过
// AIC/AIV role 路由并真正执行 atomicMax。未 attempted 的 Claim 仍保留
// role-selection 开销，但转换器会明确标成 claim.not_attempted。
constexpr uint32_t kClaimWon = 1U << 0;
constexpr uint32_t kClaimAttempted = 1U << 1;
// 下列 offset/size 来自真实 DistGlobal/DistCore ABI。standalone 保留被测关键字段的
// offset、DistCore ABI 和 kRealDistGlobalBytes 总跨度；其余区域可用 opaque padding，
// 并不是对生产结构全部字段的逐一镜像。
constexpr size_t kRealDistCoreOffset = 10043904;
constexpr size_t kRealDistGlobalBytes = 1007026048;
constexpr size_t kRealFinalBarrierBytes = 2176;
constexpr size_t kRealTasksOffset = 896;
constexpr size_t kRealFatalOffset = 4195264;
constexpr size_t kRealReplayDoneOffset = 10043776;
constexpr size_t kRealStartedCountOffset = 10043840;
// shared 把每个 task/core 固定存在的 Claim+Submit 端点移到 32B 专用区。
// 通用区容量固定为 28,416 条：默认 32B 格式仍恰好占满 1 MiB/worker；
// CCEC full-swimlane 的 16B 格式只缩短物理 stride，不扩大事件容量。
constexpr uint32_t kTraceLogicalRecordSizeBytes = 32;
#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kTraceSubmitClaimRecordSizeBytes = 32;
constexpr uint32_t kTraceRecordsPerCore = 28416;
constexpr uint32_t kTraceRecordSizeBytes =
    PA_BUILD_COMPACT_GENERIC_TRACE ? 16U : 32U;
constexpr size_t kTraceWorkerBytes =
    static_cast<size_t>(kMaxTasks) *
        kTraceSubmitClaimRecordSizeBytes +
    static_cast<size_t>(kTraceRecordsPerCore) *
        kTraceRecordSizeBytes;
static_assert(
    kTraceWorkerBytes ==
        (PA_BUILD_COMPACT_GENERIC_TRACE ? 593920U : (1U << 20)),
    "shared trace worker stride changed"
);
#else
constexpr uint32_t kTraceSubmitClaimRecordSizeBytes = 0;
constexpr uint32_t kTraceRecordsPerCore = 1U << 16;
constexpr uint32_t kTraceRecordSizeBytes = 32;
constexpr size_t kTraceWorkerBytes =
    static_cast<size_t>(kTraceRecordsPerCore) * 32U;
#endif
static_assert(
    kTraceWorkerBytes % 64U == 0,
    "each trace worker partition must remain cache-line aligned"
);
static_assert((kPayloadSlots & kPayloadMask) == 0, "payload slots must be a power of two");
static_assert(
    kMapBucketCapacity >= 32 && kMapBucketCapacity <= kMapCapacity,
    "standalone ring CAP must be in [32, 16384]"
);
static_assert(
    (kMapBucketCapacity & (kMapBucketCapacity - 1U)) == 0,
    "standalone ring CAP must be a power of two"
);
static_assert(
    kMapCapacity % kMapBucketCapacity == 0,
    "standalone ring CAP must divide the fixed 16K slot pool"
);
static_assert((kMapBuckets & kMapBucketMask) == 0, "map bucket count must be a power of two");
static_assert(
    (kMapBucketCapacity & kMapBucketSlotMask) == 0,
    "map bucket capacity must be a power of two"
);
static_assert(kMapCapacity == 16384, "private ring must preserve the old map capacity");
static_assert(
    kPaCase1MaxLiveMapEntries <= kDefaultMapBucketCapacity,
    "the default PA Case1 ring capacity no longer covers its conservative live bound"
);
static_assert(kTaskWindow > kHeapWindow, "task counters must retire before their slot is reused");
static_assert(
    kMaxTasks < kTaskCellCapacity,
    "task table must cover every PA task and private frontier sentinel"
);

// These are the measured means from the best PA A5 trace, in 1 GHz ticks.
// The scalar-NOP compatibility baseline calibrates its counts against these targets.
// 无参数默认使用 real-compute：CCEC/AscendC 执行完整 Cube/Vector 流水，
// CPU 执行对等算术；下列 NOP 常量只供显式 scalar-nop 校准。两种模式的
// Submit、依赖、heap 与 completion 路径都不靠补时修改。target 是真实泳道
// 均值，不是调度阶段预算。
constexpr uint32_t kTargetQkTicks = 44170;
constexpr uint32_t kTargetSfTicks = 53729;
constexpr uint32_t kTargetPvTicks = 27626;
constexpr uint32_t kTargetUpTicks = 1565;

// Calibrated on the local A5 with the CCEC RuntimeNop implementation. These
// counts resolve to the measured targets above; they are not cycle guesses.
constexpr uint32_t kDefaultQkNops = 129600;
constexpr uint32_t kDefaultSfNops = 157900;
constexpr uint32_t kDefaultPvNops = 79950;
constexpr uint32_t kDefaultUpNops = 2400;

enum class CoreRole : uint32_t {
    Aic = 0,
    Aiv = 1,
};

enum class FinalBarrierShape : uint32_t {
    Flat = 0,
    TwoLevel4 = 1,
    TwoLevel8 = 2,
    TwoLevel16 = 3,
    ThreeLevel6x4x4 = 4,
};

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
// split caller 与 finish 通过同一份 block-local 状态协作。cookie 只用于
// 正确性闭环：它同时编码 worker 与核型，防止 AIC/AIV 或相邻 worker 串用状态。
constexpr uint64_t kCompeteFirstSplitStateCookieBase = 0x434653504c495400ULL;
#endif

// private 的 task_id % 5 即 kind。shared 单组虽有相同数值布局，也统一
// 从 ticket 元数据恢复 kind；多组布局为 Alloc + N×(QK/SF/PV/UP)，
// 不能再用 %5 推导。
enum class TaskKind : uint32_t {
    Alloc = 0,
    Qk = 1,
    Sf = 2,
    Pv = 3,
    Up = 4,
    Count = 5,
};

#if PTO_FDWIC_SHARED_MAP
// shared Claim 继续使用现有 4/4/8 条跨 task 高水位 cursor，但每条
// cursor 只由映射到同一 shard 的 worker 子集竞争。shared 中闲置的
// 四路 legacy vector cursor 与四路 alloc cursor 合成八路 Alloc
// 高水位，使 Alloc/QK-PV/SF-UP 分别保留 12/8/8 个动态候选。与
// 固定单 owner 不同，每个 shard 内仍有多个候选；与 per-task atomic
// 不同，后续 task 仍受同一高水位链约束，不会放大无界 run-ahead。
constexpr uint32_t kSharedAllocCursorShards =
    2U * kCursorShards;
constexpr uint32_t kSharedAllocClaimParticipants =
    kWorkers / kSharedAllocCursorShards;
constexpr uint32_t kSharedAicClaimParticipants =
    kAicWorkers / kCursorShards;
constexpr uint32_t kSharedAivClaimParticipants =
    kAivWorkers / kSharedVectorCursorShards;
static_assert(
    kWorkers % kSharedAllocCursorShards == 0 &&
        kAicWorkers % kCursorShards == 0 &&
        kAivWorkers % kSharedVectorCursorShards == 0,
    "shared Claim participant groups must divide the worker topology"
);
static_assert(
    kSharedAllocClaimParticipants == 12U &&
        kSharedAicClaimParticipants == 8U &&
        kSharedAivClaimParticipants == 8U,
    "shared Claim participant widths changed unexpectedly"
);

#ifdef PA_DEVICE
#define PA_SHARED_CLAIM_INLINE PA_DEVICE
#else
#define PA_SHARED_CLAIM_INLINE inline
#endif

PA_SHARED_CLAIM_INLINE constexpr uint32_t
SharedClaimParticipantCount(TaskKind kind) {
    switch (kind) {
        case TaskKind::Alloc:
            return kSharedAllocClaimParticipants;
        case TaskKind::Qk:
        case TaskKind::Pv:
            return kSharedAicClaimParticipants;
        case TaskKind::Sf:
        case TaskKind::Up:
            return kSharedAivClaimParticipants;
        case TaskKind::Count:
            return 0;
    }
    return 0;
}

PA_SHARED_CLAIM_INLINE constexpr bool IsSharedClaimParticipant(
    uint32_t worker_id, uint32_t task_id, TaskKind kind
) {
    if (worker_id >= kWorkers ||
        task_id >= kTaskCellCapacity) {
        return false;
    }
    if (kind == TaskKind::Alloc) {
        return worker_id % kSharedAllocCursorShards ==
               task_id % kSharedAllocCursorShards;
    }
    if (kind == TaskKind::Qk ||
        kind == TaskKind::Pv) {
        return worker_id < kAicWorkers &&
               worker_id % kCursorShards ==
                   task_id % kCursorShards;
    }
    if (kind == TaskKind::Sf ||
        kind == TaskKind::Up) {
        return worker_id >= kAicWorkers &&
               (worker_id - kAicWorkers) %
                       kSharedVectorCursorShards ==
                   task_id % kSharedVectorCursorShards;
    }
    return false;
}

#undef PA_SHARED_CLAIM_INLINE
#endif

// 记录 kernel 最终在哪次 drain 中落地：Submit 开头、slot/heap 背压期间，或
// 所有 worker 回放结束后的最终清空。三者之和必须等于实际 kernel 数。
enum class DrainPlace : uint32_t {
    EfDrain = 0,
    RingBackpressure = 1,
    FinalDrain = 2,
    Count = 3,
};

enum class TensorArgType : int32_t {
    Input = 0,
    Output = 1,
    Inout = 2,
    OutputExisting = 3,
    NoDependency = 4,
};
// Input 作为 kernel 输入并参与依赖、但不登记为写者；Output 由本次 Submit 在 heap 中物化；
// Inout 与 OutputExisting 还需登记进每 worker 私有 TensorMap，供后续重叠区间查询 producer。
static_assert(sizeof(TensorArgType) == sizeof(int32_t), "TensorArgType must match the PA tag ABI");

enum class DataType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    Int16 = 3,
    Int8 = 4,
    Uint8 = 5,
    Bfloat16 = 6,
    Int64 = 7,
    Uint64 = 8,
    Uint16 = 9,
    Uint32 = 10,
    Bool = 11,
    Count = 12,
};

// ProfilePhase 是聚合计数下标，TracePhase 是原始泳道事件 ABI；二者故意分离，
// 不能假设枚举值相同。一次 trace 写入可同时归入一个不同命名的 profile 阶段。
enum class ProfilePhase : uint32_t {
    Orchestration = 0,
    Submit = 1,
    EfDrain = 2,
    Materialize = 3,
    PrepareMap = 4,
    Claim = 5,
    Fanin = 6,
    Register = 7,
    WaitForSlot = 8,
    HeapGuard = 9,
    Build = 10,
    ReplayTail = 11,
    Count = 12,
};

// submit-pmu 每个 ELF 只编译一个局部归因阶段。none 不做中途 counter
// 读取，是完整 Submit 的正式基线；其余阶段都在每个 worker 的五次 Submit
// 上各执行一次，因此统一按固定 5*batches 次数闭合。历史 ID=3 曾用于
// winner-only WaitForSlot，现已退役且不复用，避免旧 raw 被误认成新阶段。
enum class SubmitPmuPhase : uint32_t {
    None = 0,
    Claim = 1,
    EfDrain = 2,
    Materialize = 4,
    Register = 5,
    Count = 6,
};

#ifndef PA_SUBMIT_PMU_PHASE_ID
#define PA_SUBMIT_PMU_PHASE_ID 0
#endif

constexpr SubmitPmuPhase kCompiledSubmitPmuPhase =
    static_cast<SubmitPmuPhase>(PA_SUBMIT_PMU_PHASE_ID);
constexpr uint32_t kBuildVariantSwimlane = 1U;
constexpr uint32_t kBuildVariantSubmitPmu = 2U;
constexpr uint32_t kBuildVariantPerfClock = 3U;
constexpr uint32_t kCompiledBuildVariant =
#if PA_BUILD_SUBMIT_PMU
    kBuildVariantSubmitPmu;
#elif PA_BUILD_PERF_CLOCK
    kBuildVariantPerfClock;
#else
    kBuildVariantSwimlane;
#endif
static_assert(
    PA_SUBMIT_PMU_PHASE_ID == static_cast<int>(SubmitPmuPhase::None) ||
        PA_SUBMIT_PMU_PHASE_ID == static_cast<int>(SubmitPmuPhase::Claim) ||
        PA_SUBMIT_PMU_PHASE_ID == static_cast<int>(SubmitPmuPhase::EfDrain) ||
        PA_SUBMIT_PMU_PHASE_ID == static_cast<int>(SubmitPmuPhase::Materialize) ||
        PA_SUBMIT_PMU_PHASE_ID == static_cast<int>(SubmitPmuPhase::Register),
    "invalid compiled submit-pmu phase"
);

struct NopCounts {
    uint32_t qk;
    uint32_t sf;
    uint32_t pv;
    uint32_t up;
};

// winner 的计算负载与 NOP 校准量使用两套独立计数，禁止把同一个数字同时解释成
// scalar 指令条数和 vector/cube 工作迭代数。首阶段只有 CCEC 实现 RealCompute；
// 该 ABI 放在公共模型中，便于后续按相同配置逐步迁移 AscendC 与 CPU。
struct WorkloadCounts {
    uint32_t qk;
    uint32_t sf;
    uint32_t pv;
    uint32_t up;
};
static_assert(sizeof(WorkloadCounts) == 16, "workload counts ABI changed");

enum class WinnerWorkloadMode : uint32_t {
    ScalarNop = 0,
    RealCompute = 1,
};

constexpr uint32_t kWinnerWorkloadConfigVersion = 1;

// 真实计算工作区是 standalone sidecar，不属于生产 DistGlobal/DistCore ABI。
// workspace_base 指向 host 单独申请并初始化的 GM；每个 worker 只写自己的输出片段。
struct alignas(64) WinnerWorkloadConfig {
    uint32_t mode;
    uint32_t version;
    WorkloadCounts repeats;
    uint64_t workspace_base;
    uint64_t workspace_bytes;
    uint32_t reserved[6];
};
static_assert(sizeof(WinnerWorkloadConfig) == 64, "winner workload config must occupy one cache line");
static_assert(offsetof(WinnerWorkloadConfig, mode) == 0, "winner workload mode offset changed");
static_assert(offsetof(WinnerWorkloadConfig, version) == 4, "winner workload version offset changed");
static_assert(offsetof(WinnerWorkloadConfig, repeats) == 8, "winner workload counts offset changed");
static_assert(offsetof(WinnerWorkloadConfig, workspace_base) == 24, "winner workload base offset changed");
static_assert(offsetof(WinnerWorkloadConfig, workspace_bytes) == 32, "winner workload bytes offset changed");

// RunConfig 是 host 在 launch 前写、worker 启动时只读的控制 cache line。
// 输入为 batch/NOP/诊断开关；输出不回写这里，而发布到独立 WorkerResult。
struct alignas(64) RunConfig {
    uint32_t batches;
    uint32_t workers;
    NopCounts nops;
    uint32_t profile_phases;
    uint32_t trace_enabled;
    uint64_t trace_base;
    uint32_t trace_records_per_core;
    uint32_t final_barrier_shape;
    // host 与 device 分别按自己的编译常量写入/核对这四个字段。它们占用
    // RunConfig 原有的 16B padding，不扩大热控制行，也不移动生产状态。
    uint32_t build_identity_magic;
    uint32_t build_identity_abi_version;
    uint32_t tensor_map_mode;
    uint32_t scheduler_state_size;
};
static_assert(sizeof(RunConfig) == 64, "RunConfig must occupy one cache line");
static_assert(offsetof(RunConfig, build_identity_magic) == 48, "build identity offset changed");

// CCEC PMU 的窗口选择与寄存器表是 standalone 诊断 sidecar，不属于
// RunConfig，也不应占用 winner workload 的字段。独占 cache line 后，
// host/kernel 可显式搬运和失效整个配置，而不会再发生 reserved[4] 越界。
struct alignas(64) PmuProbeConfig {
    uint32_t mode;
    uint32_t work_amount;
    uint64_t register_table;
    uint32_t magic;
    // 该 cache line 已随 RunConfig 一起由 host 写入并由 device 失效读取。
    // 复用首个保留槽做构建变体握手，防止绕过 manifest 后把
    // swimlane/submit-pmu/perf-clock 的 host 与 kernel 交叉运行。
    uint32_t build_variant;
    uint32_t reserved[10];
};
static_assert(sizeof(PmuProbeConfig) == 64, "PMU probe config must occupy one cache line");
static_assert(offsetof(PmuProbeConfig, register_table) == 8, "PMU register-table offset changed");
static_assert(offsetof(PmuProbeConfig, magic) == 16, "PMU magic offset changed");
static_assert(offsetof(PmuProbeConfig, build_variant) == 20, "build variant offset changed");

enum class TracePhase : int32_t {
    Kernel = 0,
    Alloc = 1,
    Build = 2,
    DrainWon = 3,
    Replay = 4,
    RingBp = 5,
    EfDrain = 6,
    Commit = 7,
    Submit = 8,
    Materialize = 9,
    PrepareMap = 10,
    Claim = 11,
    Fanin = 12,
    Register = 13,
    Atomic = 14,
    // 逐 atomic 诊断构建中，每个 worker 只记录一次连续两次 SYS_CNT 的
    // 空括号，用来给出同一二进制、同一物理核上的计时分辨率下限。
    ClockBaseline = 15,
    // schema-v5 的父区间与真实动作区间。loser 没有可单列的真实动作，
    // 其时间直接归入离线计算的 Submit residual，不占用 raw 记录。
    OrchestrationReplay = 16,
    FinalDrain = 17,
    WinnerBuild = 18,
    AllocComplete = 19,
    // shared Register 父区间内只增加这一条真实 metadata 发布边界。
    // 等待 insert turn 和把 turn 交给 N+1 的两段由父/子端点离线还原，
    // 避免为每个 winner 再扩张两条 raw 记录，更不能逐 poll 记录。
    SharedRegisterPublishMetadata = 20,
    // Materialize 尾部精确包住 fresh shared-output cell 的预检、writer
    // 预留、descriptor flush 与 published 发布。该 cell 按 task_id
    // 独占，不进入后续 ordinary/symbol 的全局串行插入区。
    SharedMaterializePublishTaskOutputs = 21,
    // PublishTaskOutputs 内再拆两层：先整批 copy descriptor，再整批
    // FlushRegion。两端点仍由正式 Materialize 调用点写 raw，通用 helper
    // 只回传时间戳，不自行 WriteTrace。
    SharedMaterializePublishTaskOutputsCopy = 22,
    SharedMaterializePublishTaskOutputsFlush = 23,
    // 区域级 DCCI 记录是 scalar 调度泳道的 overlay；它不参与 Submit
    // 排他分段，也不复用 Atomic 的 flags/auxiliary 编号。
    Dcci = 24,
    Count = 25,
};

// AtomicSite 按 standalone PA 中真实出现的源码调用点分类。编号写入 TraceRecord::auxiliary，
// 是离线泳道 schema 的一部分；追加新位置只能在 Count 前扩展，不能重排既有值。
enum class AtomicSite : uint32_t {
    StartupIncrement = 0,
    StartupPoll = 1,
    FatalPoll = 2,
    FatalSet = 3,
    ClaimMax = 4,
    FaninFlagLoad = 5,
    CompletionVendExchange = 6,
    CompletionFlagExchange = 7,
    FrontierInitialLoad = 8,
    FrontierFlagLoad = 9,
    FrontierMax = 10,
    HeapFrontierLoad = 11,
    HeapVendLoad = 12,
    ReplayDoneIncrement = 13,
    ReplayDonePoll = 14,
    // shared heap 的预检 load 与两个返回型 FetchAdd 必须分开：前者只读
    // 全局/分片控制字，后者的旧值直接决定本 task 的物理区间。
    SharedHeapVendLoad = 15,
    SharedHeapCursorLoad = 16,
    SharedHeapCursorReserve = 17,
    SharedHeapVendAdvance = 18,
    // shared Register 的 insert-turn 等待只按一次 Wait episode 聚合，
    // 不为循环内每次 Load 写 raw；handoff CAS 则保留一条 return-ready。
    SharedInsertTurnPoll = 19,
    SharedInsertTurnHandoff = 20,
    // shared 正式 Submit 中原先绕过 TraceAtomic* 的固定调用点。读取类按
    // 业务位置拆开，避免把 Fanin、metadata 串行区和 output 发布混成一项；
    // 编号继续 append-only，旧 raw 的 0..20 语义不变。
    SharedWinnerFatalGuardLoad = 21,
    SharedMetadataFatalGuardLoad = 22,
    SharedFaninOutputPublishedLoad = 23,
    SharedMetadataOutputPublishedLoad = 24,
    SharedFaninLastWriterLoad = 25,
    SharedMetadataLastWriterLoad = 26,
    SharedMetadataLastWriterCommit = 27,
    SharedOutputWriterReserve = 28,
    SharedOutputPublishedExchange = 29,
    SharedMapLookupHeadLoad = 30,
    SharedMapLookupTailLoad = 31,
    // ordinary-region ring 的 lookup/preflight/append 控制字。lookup 的
    // 两次 seq 双检复用同一站点；append 的 reset/publish/tail 分开，
    // 便于直接看出串行 Register 中是哪一步在等待共享 cache line。
    SharedMapLookupSeqLoad = 32,
    SharedMapAppendHeadLoad = 33,
    SharedMapAppendTailLoad = 34,
    SharedMapAppendSeqLoad = 35,
    SharedMapAppendSeqResetExchange = 36,
    SharedMapAppendSeqPublishExchange = 37,
    SharedMapAppendTailExchange = 38,
    // 仅在失败回滚中出现，返回旧值不参与协议判断。
    SharedOutputRollbackExchange = 39,
    Count = 40,
};

// Atomic 记录 flags 的低四位保存操作种类；bit4 表示返回值参与后续判断，
// bit5 表示 Load 观察到零，bit6 表示结束时间已由返回值依赖推进到
// return-ready 边界。schema-v3 中 bit7 区分等待区 PollBatch：此时
// bits[31:8] 是精确调用次数；直接 FetchMax 中同一区域仍表示软件重试数。
enum class AtomicOp : uint32_t {
    Load = 0,
    Exchange = 1,
    FetchAdd = 2,
    FetchMax = 3,
    CompareExchange = 4,
};
constexpr uint32_t kAtomicOpMask = 0x0fU;
constexpr uint32_t kAtomicResultUsed = 1U << 4;
constexpr uint32_t kAtomicValueZero = 1U << 5;
constexpr uint32_t kAtomicReturnReady = 1U << 6;
constexpr uint32_t kAtomicPollBatch = 1U << 7;
constexpr uint32_t kAtomicRetriesShift = 8;
constexpr uint32_t kAtomicPollCountShift = 8;
constexpr uint32_t kAtomicPollCountMax = 0x00ffffffU;
constexpr uint32_t kAtomicPollBatchSiteCount = 6;
static_assert(kAtomicPollBatchSiteCount <= 32, "PollBatch enable mask supports at most 32 compact indices");

// DCCI 与 Atomic 使用同一个 32B TraceRecord，但拥有完全独立的 raw ABI。
// 每条记录描述一个区域原语或一组同 op 的聚合区域原语：
// - bits[1:0]：DcciOp；
// - bit2：每个区域原语末尾都执行 trailing DSB；
// - bits[6:3]：logical call_count（1..15）；
// - bit7：保留，必须为 0；
// - bits[31:8]：实际覆盖的 64B cache-line 总数。
enum class DcciOp : uint32_t {
    Invalidate = 0,
    CleanOut = 1,
    Count = 2,
};

enum class DcciSite : uint32_t {
    SharedFaninHistoryInvalidate = 0,
    SharedWriterHistoryFlush = 1,
    SharedOutputRollbackFlush = 2,
    SharedOutputDescriptorFlush = 3,
    SharedRegionReadInvalidate = 4,
    SharedRegionAppendInvalidate = 5,
    SharedRegionAppendFlush = 6,
    SharedWinnerBuildDescriptorInvalidate = 7,
    // observer 自身的 records clean 与 core-state clean 必须聚合为一条
    // terminal row，不能在 FlushRegion 内递归写 trace。
    ObserverTraceExport = 8,
    // RunConfig DCCI 发生在 AttachTrace 之前；正常握手成功后用已保存的
    // begin/end 补记这一条。它同时存在于 private/shared 构建。
    StartupConfigInvalidate = 9,
    Count = 10,
};

constexpr uint32_t kDcciOpMask = 0x03U;
constexpr uint32_t kDcciTrailingDsb = 1U << 2;
constexpr uint32_t kDcciCallCountShift = 3;
constexpr uint32_t kDcciCallCountMask = 0x0fU;
constexpr uint32_t kDcciReservedBit = 1U << 7;
constexpr uint32_t kDcciLineCountShift = 8;
constexpr uint32_t kDcciLineCountMax = 0x00ffffffU;

// 这些映射是 raw ABI 的一部分，同时被 device 聚合器与 host 闭环校验使用。
// 0..14 与真实 PA 保持稳定；BlockWon 尚未在 standalone 中实现，不能只为
// 编号齐全而追加没有真实调用路径的 site。
#ifdef PA_DEVICE
#define PA_MODEL_INLINE PA_DEVICE
#else
#define PA_MODEL_INLINE inline
#endif

PA_MODEL_INLINE constexpr AtomicOp AtomicSiteExpectedOp(AtomicSite site) {
    switch (site) {
        case AtomicSite::StartupIncrement:
        case AtomicSite::ReplayDoneIncrement:
        case AtomicSite::SharedHeapCursorReserve:
        case AtomicSite::SharedHeapVendAdvance:
            return AtomicOp::FetchAdd;
        case AtomicSite::FatalSet:
        case AtomicSite::CompletionVendExchange:
        case AtomicSite::CompletionFlagExchange:
            return AtomicOp::Exchange;
        case AtomicSite::ClaimMax:
        case AtomicSite::FrontierMax:
            return AtomicOp::FetchMax;
        case AtomicSite::SharedInsertTurnHandoff:
        case AtomicSite::SharedMetadataLastWriterCommit:
            return AtomicOp::CompareExchange;
        case AtomicSite::SharedOutputWriterReserve:
            return AtomicOp::FetchMax;
        case AtomicSite::SharedOutputPublishedExchange:
        case AtomicSite::SharedMapAppendSeqResetExchange:
        case AtomicSite::SharedMapAppendSeqPublishExchange:
        case AtomicSite::SharedMapAppendTailExchange:
        case AtomicSite::SharedOutputRollbackExchange:
            return AtomicOp::Exchange;
        default:
            return AtomicOp::Load;
    }
}

PA_MODEL_INLINE constexpr bool AtomicSiteResultUsed(AtomicSite site) {
    switch (site) {
        case AtomicSite::StartupIncrement:
        case AtomicSite::FatalSet:
        case AtomicSite::CompletionVendExchange:
        case AtomicSite::CompletionFlagExchange:
        case AtomicSite::ReplayDoneIncrement:
        case AtomicSite::SharedOutputRollbackExchange:
            return false;
        case AtomicSite::StartupPoll:
        case AtomicSite::FatalPoll:
        case AtomicSite::ClaimMax:
        case AtomicSite::FaninFlagLoad:
        case AtomicSite::FrontierInitialLoad:
        case AtomicSite::FrontierFlagLoad:
        case AtomicSite::FrontierMax:
        case AtomicSite::HeapFrontierLoad:
        case AtomicSite::HeapVendLoad:
        case AtomicSite::ReplayDonePoll:
        case AtomicSite::SharedHeapVendLoad:
        case AtomicSite::SharedHeapCursorLoad:
        case AtomicSite::SharedHeapCursorReserve:
        case AtomicSite::SharedHeapVendAdvance:
        case AtomicSite::SharedInsertTurnPoll:
        case AtomicSite::SharedInsertTurnHandoff:
        case AtomicSite::SharedWinnerFatalGuardLoad:
        case AtomicSite::SharedMetadataFatalGuardLoad:
        case AtomicSite::SharedFaninOutputPublishedLoad:
        case AtomicSite::SharedMetadataOutputPublishedLoad:
        case AtomicSite::SharedFaninLastWriterLoad:
        case AtomicSite::SharedMetadataLastWriterLoad:
        case AtomicSite::SharedMetadataLastWriterCommit:
        case AtomicSite::SharedOutputWriterReserve:
        case AtomicSite::SharedOutputPublishedExchange:
        case AtomicSite::SharedMapLookupHeadLoad:
        case AtomicSite::SharedMapLookupTailLoad:
        case AtomicSite::SharedMapLookupSeqLoad:
        case AtomicSite::SharedMapAppendHeadLoad:
        case AtomicSite::SharedMapAppendTailLoad:
        case AtomicSite::SharedMapAppendSeqLoad:
        case AtomicSite::SharedMapAppendSeqResetExchange:
        case AtomicSite::SharedMapAppendSeqPublishExchange:
        case AtomicSite::SharedMapAppendTailExchange:
            return true;
        case AtomicSite::Count:
            return false;
    }
    return false;
}

PA_MODEL_INLINE constexpr int32_t AtomicPollBatchIndex(AtomicSite site) {
    switch (site) {
        case AtomicSite::StartupPoll:
            return 0;
        case AtomicSite::FatalPoll:
            return 1;
        case AtomicSite::FaninFlagLoad:
            return 2;
        case AtomicSite::HeapFrontierLoad:
            return 3;
        case AtomicSite::HeapVendLoad:
            return 4;
        case AtomicSite::ReplayDonePoll:
            return 5;
        default:
            return -1;
    }
}

PA_MODEL_INLINE constexpr AtomicSite AtomicPollBatchSite(uint32_t index) {
    switch (index) {
        case 0:
            return AtomicSite::StartupPoll;
        case 1:
            return AtomicSite::FatalPoll;
        case 2:
            return AtomicSite::FaninFlagLoad;
        case 3:
            return AtomicSite::HeapFrontierLoad;
        case 4:
            return AtomicSite::HeapVendLoad;
        case 5:
            return AtomicSite::ReplayDonePoll;
        default:
            return AtomicSite::Count;
    }
}

PA_MODEL_INLINE constexpr bool AtomicSiteIsPollBatchable(AtomicSite site) {
    // insert-turn 使用 Wait 已有 polls 一次性写聚合记录，不占用
    // AtomicPollBurst compact slot，也绝不进入逐调用 TraceAtomicLoad。
    return AtomicPollBatchIndex(site) >= 0 ||
           site == AtomicSite::SharedInsertTurnPoll;
}

PA_MODEL_INLINE constexpr bool AtomicSiteIsSharedOnly(AtomicSite site) {
    return static_cast<uint32_t>(site) >=
               static_cast<uint32_t>(AtomicSite::SharedHeapVendLoad) &&
           static_cast<uint32_t>(site) <
               static_cast<uint32_t>(AtomicSite::Count);
}

PA_MODEL_INLINE constexpr uint32_t AtomicPollBatchMask(AtomicSite site) {
    const int32_t index = AtomicPollBatchIndex(site);
    return index >= 0 && index < 32 ? 1U << static_cast<uint32_t>(index) : 0U;
}

PA_MODEL_INLINE constexpr DcciOp DcciSiteExpectedOp(DcciSite site) {
    switch (site) {
        case DcciSite::SharedWriterHistoryFlush:
        case DcciSite::SharedOutputRollbackFlush:
        case DcciSite::SharedOutputDescriptorFlush:
        case DcciSite::SharedRegionAppendFlush:
        case DcciSite::ObserverTraceExport:
            return DcciOp::CleanOut;
        default:
            return DcciOp::Invalidate;
    }
}

PA_MODEL_INLINE constexpr bool DcciSiteIsSharedOnly(DcciSite site) {
    return static_cast<uint32_t>(site) <
           static_cast<uint32_t>(DcciSite::ObserverTraceExport);
}

static_assert(
    AtomicSiteIsSharedOnly(AtomicSite::SharedInsertTurnPoll) &&
        AtomicSiteIsSharedOnly(
            AtomicSite::SharedInsertTurnHandoff
        ),
    "insert-turn atomic sites must remain shared-only"
);
static_assert(
    !DcciSiteIsSharedOnly(DcciSite::ObserverTraceExport),
    "observer trace export must remain available in both TensorMap modes"
);
static_assert(
    !DcciSiteIsSharedOnly(DcciSite::StartupConfigInvalidate),
    "startup config invalidate must remain available in both TensorMap modes"
);

#undef PA_MODEL_INLINE

// ClockBaseline 的 bit0 区分普通连续 SYS_CNT 与后端的 atomic 返回依赖
// 计时钩子；后者用于量化那一条依赖 MOV 自身带来的固定底噪。
constexpr uint32_t kClockAtomicDependency = 1U << 0;
constexpr uint32_t kClockAtomicDependencyApplied = 1U << 1;

struct alignas(64) TraceCoreState {
    volatile uint32_t count;
    volatile uint32_t dropped;
    // logical atomic 调用数与物理记录数分开闭合：PollBatch 的一条记录可以
    // 表示多次只读轮询，physical = atomic_calls - poll_calls + batch_records。
    volatile uint32_t atomic_calls;
    volatile uint32_t poll_calls;
    volatile uint32_t poll_batch_records;
    // 拓扑在一个 worker 分区内恒定；只在 core state 保存一份权威身份，
    // 32B TraceRecord 不再为每条事件重复写入这 12B。
    volatile int32_t core_idx;
    volatile int32_t block_id;
    volatile int32_t lane;
    // 复用 core-state 既有 32B 尾部，不扩大 header。calls/lines 是
    // logical 总量，records 是实际落盘的 Dcci row 数。
    volatile uint32_t dcci_calls;
    volatile uint32_t dcci_lines;
    volatile uint32_t dcci_records;
    uint32_t padding[5];
};
// 每个 worker 独占一个计数 cache line 和一段定长 records，不需要为了写 trace
// 再引入跨核 atomic；满容量后只增加本 worker 的 dropped。
static_assert(sizeof(TraceCoreState) == 64, "trace core state must occupy one cache line");
static_assert(offsetof(TraceCoreState, dcci_calls) == 32, "DCCI counters must reuse the core-state tail");
static_assert(offsetof(TraceCoreState, dcci_records) == 40, "DCCI counter layout changed");

struct alignas(64) TraceHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_cores;
    uint32_t records_per_core;
    uint64_t frequency_hz;
    // 占用首条 header cache line 的既有 padding，不移动 cores。host/device
    // 都据此拒绝把旧 64B record 缓冲误解为当前构建的物理 generic ABI。
    uint32_t record_size_bytes;
    TraceCoreState cores[kRuntimeMaxWorkers];
};
// 本 benchmark 固定物理分配 kWorkers=96 个定长 record 分区，合法 header 也要求
// num_cores==96；其 header/record 布局和 phase 编号可转换为真实泳道使用的 JSON。
static_assert(offsetof(TraceHeader, record_size_bytes) == 24, "trace record-size offset changed");
static_assert(offsetof(TraceHeader, cores) == 64, "trace core states must start at the second cache line");
static_assert(sizeof(TraceHeader) == 6976, "trace header must match PA swimlane layout");

struct alignas(32) TraceRecord {
    uint64_t start_cycle;
    uint64_t end_cycle;
    int32_t task_id;
    int32_t function_id;
    uint32_t flags;
    uint16_t phase;
    uint16_t auxiliary;
};
// 与真实 FDWIC 的生产 record 同形：每条只保留事件自身字段，物理
// core/block/lane 由所属分区的 TraceCoreState 在 host 导出时回填。
static_assert(offsetof(TraceRecord, flags) == 24, "trace flags offset changed");
static_assert(offsetof(TraceRecord, phase) == 28, "trace phase offset changed");
static_assert(offsetof(TraceRecord, auxiliary) == 30, "trace auxiliary offset changed");
static_assert(
    sizeof(TraceRecord) == kTraceLogicalRecordSizeBytes,
    "logical trace record must occupy half a cache line"
);
static_assert(alignof(TraceRecord) == 32, "trace record alignment changed");
static_assert(
    static_cast<uint32_t>(TracePhase::Count) <= UINT16_MAX,
    "trace phase does not fit the 16-bit raw ABI"
);
static_assert(
    static_cast<uint32_t>(AtomicSite::Count) <= UINT16_MAX,
    "atomic site does not fit the 16-bit raw ABI"
);
static_assert(
    static_cast<uint32_t>(DcciSite::Count) <= UINT16_MAX,
    "DCCI site does not fit the 16-bit raw ABI"
);

// shared CCEC full-swimlane 的通用物理记录只保留低 32-bit 时钟与一个
// 紧凑业务字；host 回读后恢复成上面的 32B 逻辑 TraceRecord。flags
// 完整保留 32 bit，Atomic/DCCI 的既有 raw ABI 不发生裁剪。
struct alignas(16) CompactTraceRecord16 {
    uint32_t start_cycle_low;
    uint32_t end_cycle_low;
    uint32_t flags;
    uint32_t packed;
};
static_assert(
    sizeof(CompactTraceRecord16) == 16 &&
        alignof(CompactTraceRecord16) == 16,
    "compact generic trace record must occupy 16 bytes"
);
static_assert(
    offsetof(CompactTraceRecord16, start_cycle_low) == 0 &&
        offsetof(CompactTraceRecord16, end_cycle_low) == 4 &&
        offsetof(CompactTraceRecord16, flags) == 8 &&
        offsetof(CompactTraceRecord16, packed) == 12,
    "compact generic trace offsets changed"
);

constexpr uint32_t kCompactTraceTaskBits = 13;
constexpr uint32_t kCompactTraceTaskMask =
    (1U << kCompactTraceTaskBits) - 1U;
constexpr uint32_t kCompactTraceTaskSentinel =
    kCompactTraceTaskMask;
constexpr uint32_t kCompactTraceFunctionShift = 13;
constexpr uint32_t kCompactTraceFunctionBits = 3;
constexpr uint32_t kCompactTraceFunctionMask =
    (1U << kCompactTraceFunctionBits) - 1U;
constexpr uint32_t kCompactTraceFunctionSentinel =
    kCompactTraceFunctionMask;
constexpr uint32_t kCompactTracePhaseShift = 16;
constexpr uint32_t kCompactTracePhaseBits = 5;
constexpr uint32_t kCompactTracePhaseMask =
    (1U << kCompactTracePhaseBits) - 1U;
constexpr uint32_t kCompactTraceAuxiliaryShift = 21;
constexpr uint32_t kCompactTraceAuxiliaryBits = 11;
constexpr uint32_t kCompactTraceAuxiliaryMask =
    (1U << kCompactTraceAuxiliaryBits) - 1U;

static_assert(
    kMaxTasks <= kCompactTraceTaskSentinel,
    "compact trace task field cannot encode every PA task"
);
static_assert(
    static_cast<uint32_t>(TracePhase::Count) <=
        kCompactTracePhaseMask + 1U,
    "compact trace phase field is too narrow"
);
static_assert(
    static_cast<uint32_t>(AtomicSite::Count) <=
            kCompactTraceAuxiliaryMask + 1U &&
        static_cast<uint32_t>(DcciSite::Count) <=
            kCompactTraceAuxiliaryMask + 1U &&
        kMaxTaskTensors <= kCompactTraceAuxiliaryMask &&
        kMaxFanin <= kCompactTraceAuxiliaryMask,
    "compact trace auxiliary field is too narrow"
);

#ifdef PA_DEVICE
#define PA_TRACE_ABI_INLINE PA_DEVICE
#else
#define PA_TRACE_ABI_INLINE inline
#endif

PA_TRACE_ABI_INLINE bool CompactTraceFieldsFit(
    int32_t task_id, int32_t function_id,
    TracePhase phase, uint32_t auxiliary
) {
    return task_id >= -1 &&
           task_id < static_cast<int32_t>(kMaxTasks) &&
           function_id >= -1 && function_id <= 3 &&
           static_cast<uint32_t>(phase) <
               static_cast<uint32_t>(TracePhase::Count) &&
           auxiliary <= kCompactTraceAuxiliaryMask;
}

PA_TRACE_ABI_INLINE uint32_t PackCompactTraceFields(
    int32_t task_id, int32_t function_id,
    TracePhase phase, uint32_t auxiliary
) {
    return
        (static_cast<uint32_t>(task_id) &
         kCompactTraceTaskMask) |
        ((static_cast<uint32_t>(function_id) &
          kCompactTraceFunctionMask)
         << kCompactTraceFunctionShift) |
        (static_cast<uint32_t>(phase) <<
         kCompactTracePhaseShift) |
        (auxiliary << kCompactTraceAuxiliaryShift);
}

#undef PA_TRACE_ABI_INLINE

#if PA_BUILD_COMPACT_GENERIC_TRACE
using TraceStorageRecord = CompactTraceRecord16;
#else
using TraceStorageRecord = TraceRecord;
#endif
static_assert(
    sizeof(TraceStorageRecord) == kTraceRecordSizeBytes,
    "generic trace storage size disagrees with the build identity"
);

struct alignas(32) SharedSubmitClaimTraceRecord {
    uint64_t claim_begin;
    // SYS_CNT 在 watchdog 窗口内不会触及 bit63；复用该位保存 winner，
    // 不为逐 task 固定存在的布尔值扩张记录。
    uint64_t claim_end_and_winner;
    uint64_t submit_begin;
    uint64_t submit_end;
};
constexpr uint64_t kSharedClaimWinnerBit = 1ULL << 63;
static_assert(
    sizeof(SharedSubmitClaimTraceRecord) == 32 &&
        alignof(SharedSubmitClaimTraceRecord) == 32,
    "shared Submit/Claim record must remain 32 bytes"
);
static_assert(
    offsetof(SharedSubmitClaimTraceRecord, claim_begin) == 0 &&
        offsetof(SharedSubmitClaimTraceRecord, claim_end_and_winner) == 8 &&
        offsetof(SharedSubmitClaimTraceRecord, submit_begin) == 16 &&
        offsetof(SharedSubmitClaimTraceRecord, submit_end) == 24,
    "shared compact Submit/Claim offsets changed"
);
#if PTO_FDWIC_SHARED_MAP
static_assert(
    kTraceSubmitClaimRecordSizeBytes ==
        sizeof(SharedSubmitClaimTraceRecord),
    "shared Submit/Claim layout constant changed"
);
#endif

constexpr size_t kTraceSubmitClaimBytesPerCore =
    static_cast<size_t>(kMaxTasks) *
    kTraceSubmitClaimRecordSizeBytes;
constexpr size_t kTraceGenericBytesPerCore =
    static_cast<size_t>(kTraceRecordsPerCore) *
    sizeof(TraceStorageRecord);
static_assert(
    kTraceSubmitClaimBytesPerCore +
            kTraceGenericBytesPerCore ==
        kTraceWorkerBytes,
    "per-worker trace regions must exactly fill their partition"
);
constexpr size_t TraceWorkerOffset(uint32_t worker) {
    return sizeof(TraceHeader) +
           static_cast<size_t>(worker) * kTraceWorkerBytes;
}
constexpr size_t TraceSubmitClaimOffset(uint32_t worker) {
    return TraceWorkerOffset(worker);
}
constexpr size_t TraceRecordsOffset(uint32_t worker) {
    return TraceWorkerOffset(worker) +
           kTraceSubmitClaimBytesPerCore;
}

constexpr size_t kTraceBytes =
    sizeof(TraceHeader) +
    static_cast<size_t>(kWorkers) * kTraceWorkerBytes;

struct alignas(64) AtomicLine {
    volatile int64_t value;
    uint8_t padding[64 - sizeof(int64_t)];
};
// 热点共享量各占一个 cache line，保持生产代码的地址隔离，避免 standalone
// 因伪共享额外放大 Claim/frontier/start barrier 的竞争。
static_assert(sizeof(AtomicLine) == 64, "AtomicLine must occupy one cache line");

// final 分层汇合把 arrival 与 release 分到不同 cache line：等待 release
// 的 add-zero 不会反向堵塞尚未到达的 worker。最多 16 个叶组、4 个中间组；
// 未被当前形态使用的节点必须保持零并由 host 校验。startup 仍使用生产 flat 屏障。
struct alignas(64) FinalBarrierState {
    AtomicLine leaf_arrivals[kFinalBarrierMaxLeafGroups];
    AtomicLine leaf_releases[kFinalBarrierMaxLeafGroups];
    AtomicLine middle_arrivals[kFinalBarrierMaxMiddleGroups];
    AtomicLine middle_releases[kFinalBarrierMaxMiddleGroups];
    AtomicLine root_arrival;
    AtomicLine root_release;
};
static_assert(sizeof(FinalBarrierState) == 2688, "final barrier state size changed");

struct alignas(64) AtomicFlagLine {
    volatile int32_t value;
    uint8_t padding[64 - sizeof(int32_t)];
};
// 32-bit fatal 与 64-bit cursor 使用不同封装，但都独占 cache line；成功路径中
// fatal 始终为零，任何写一都表示协议已终止，不能作为普通等待条件清除。
static_assert(sizeof(AtomicFlagLine) == 64, "AtomicFlagLine must occupy one cache line");

struct alignas(64) TaskCell {
    volatile int64_t flag;
    volatile uint64_t vend;
#if PTO_FDWIC_SHARED_MAP
    // shared 热路径把该字作为 per-task TensorMap 插入完成原子：初值
    // -1，task N 的唯一 Claim owner 完成 writer 元数据发布后用 CAS
    // 写成 N；N+1 owner 只轮询这一字。它与 flag/vend 共处 TaskCell，
    // 但当前热路径不对该 cache line 执行 DCCI。旧 writer-ready helper
    // 只供隔离协议测试，不能与本热路径混用。
    volatile int64_t deps_prepared;
    uint8_t padding[64 - 3 * sizeof(int64_t)];
#else
    uint8_t padding[64 - 2 * sizeof(int64_t)];
#endif
};
// flag 是两种模式的依赖就绪发布位；vend 是该 task 完成时 worker 的 heap
// 快照。private 还用 flag 连续推进 frontier，并由 HeapGuard 读取
// frontier-H 对应 vend 判断环形 heap 是否可覆盖；shared no-wrap 中 vend
// 只是 aggregate-vend 快照，flag 只服务 fanin/slot，均不参与 heap 回收。
static_assert(sizeof(TaskCell) == 64, "TaskCell must occupy one cache line");
#if PTO_FDWIC_SHARED_MAP
static_assert(
    offsetof(TaskCell, deps_prepared) == 16,
    "shared task dependency-intent offset mismatch"
);
#endif

// TensorDesc 保留真实 Tensor 的两条 64-byte 数据线。owner_task_id 表达显式生产者，
// buffer_addr + 字节区间用于 TensorMap 发现同一 backing buffer 上的读写依赖。
struct TensorDesc {
    uint64_t buffer_addr;
    uint64_t buffer_size;
    uint64_t owner_task_id;
    uint64_t start_offset;
    int32_t version;
    uint32_t ndims;
    DataType dtype;
    bool manual_dep;
    bool is_contiguous;
    uint8_t child_memory;
    uint32_t shapes[kMaxTensorDims];

    uint64_t extent_elem_cache;
    uint32_t strides[kMaxTensorDims];
    uint8_t padding[36];
};
static_assert(sizeof(TensorDesc) == 128, "TensorDesc must match the PA Tensor ABI size");
static_assert(offsetof(TensorDesc, buffer_addr) == 0, "TensorDesc buffer offset mismatch");
static_assert(offsetof(TensorDesc, owner_task_id) == 16, "TensorDesc owner offset mismatch");
static_assert(offsetof(TensorDesc, start_offset) == 24, "TensorDesc view offset mismatch");
static_assert(offsetof(TensorDesc, version) == 32, "TensorDesc version offset mismatch");
static_assert(offsetof(TensorDesc, shapes) == 44, "TensorDesc shape offset mismatch");
static_assert(offsetof(TensorDesc, extent_elem_cache) == 64, "TensorDesc extent offset mismatch");
static_assert(offsetof(TensorDesc, strides) == 72, "TensorDesc stride offset mismatch");

struct TensorCreateInfo {
    uint64_t initial_value;
    bool has_initial_value;
    uint8_t padding0[7];
    uint64_t reserved0;
    uint64_t start_offset;
    int32_t version;
    uint32_t ndims;
    DataType dtype;
    bool manual_dep;
    bool is_contiguous;
    uint8_t child_memory;
    uint32_t shapes[kMaxTensorDims];
};
// CreateInfo 只描述尚未分配的 Output；Materialize 根据形状和 dtype 计算大小，
// 再把它变成位于 worker 逻辑 heap 上的 TensorDesc。
static_assert(sizeof(TensorCreateInfo) == 64, "TensorCreateInfo must match the PA create-info ABI size");
static_assert(offsetof(TensorCreateInfo, start_offset) == 24, "TensorCreateInfo start offset mismatch");
static_assert(offsetof(TensorCreateInfo, version) == 32, "TensorCreateInfo version offset mismatch");
static_assert(offsetof(TensorCreateInfo, shapes) == 44, "TensorCreateInfo shape offset mismatch");

struct MapEntry {
    uint64_t buffer_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    uint32_t payload_padding;
    // 末 16B 只为保持 standalone/真实 PA 的既有 48B entry ABI。本阶段
    // private ring 不需要 seq，也不能提前把 shared 发布协议塞进保留区。
    uint8_t abi_reserved[16];
};
// bucket 与槽下标都由外层 ring 的连续布局隐式给出，不再保存 next/prev 指针。
static_assert(sizeof(MapEntry) == 48, "MapEntry must match the PA tensor-map entry ABI");
static_assert(alignof(MapEntry) == 8, "MapEntry alignment changed");
static_assert(offsetof(MapEntry, producer) == 24, "MapEntry producer offset mismatch");
static_assert(offsetof(MapEntry, abi_reserved) == 32, "MapEntry ABI reserve offset mismatch");

struct TensorMap {
    MapEntry entries[kMapCapacity];
    // 前 128 个桶沿用默认 128×128 的原始位置。CAP=32/64 时桶数增至
    // 512/256，额外游标从原 32 KiB ABI 保留区中切出；这样默认热字段
    // offset 不动，所有 CAP 下方的 task 计数与 WorkerState 总跨度也不动。
    uint64_t bucket_heads[128];
    uint64_t bucket_tails[128];
#if PTO_FDWIC_TENSORMAP_RING_CAP == 32
    uint64_t extra_bucket_heads[384];
    uint64_t extra_bucket_tails[384];
    uint8_t abi_reserved[24576];
#elif PTO_FDWIC_TENSORMAP_RING_CAP == 64
    uint64_t extra_bucket_heads[128];
    uint64_t extra_bucket_tails[128];
    uint8_t abi_reserved[28672];
#else
    // CAP>=128 时逻辑桶数不超过 128，完整保留原来的 30 KiB padding。
    uint8_t abi_reserved[30720];
#endif
    // producer 退休时据此精确扣减 logical live_count；物理 bucket head
    // 则由访问该桶时的 RetireBucket 惰性推进。
    uint32_t task_entry_counts[kTaskWindow];
    uint32_t live_count;
    uint32_t high_water;
    int32_t alive_floor;
    int32_t cleaned_upto;
};
// alive_floor 是 lookup 的权威存活下界；cleaned_upto 表示逐任务计数已
// 精确扣减到哪里。桶头可以因惰性退休暂时落后，但不会改变逻辑 live 数。
static_assert(sizeof(TensorMap) == 823312, "TensorMap must preserve the WorkerState ABI");
static_assert(alignof(TensorMap) == 8, "TensorMap alignment changed");
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(offsetof(TensorMap, bucket_heads) == 786432, "TensorMap head offset mismatch");
static_assert(offsetof(TensorMap, bucket_tails) == 787456, "TensorMap tail offset mismatch");
static_assert(offsetof(TensorMap, abi_reserved) == 788480, "TensorMap reserve offset mismatch");
static_assert(offsetof(TensorMap, task_entry_counts) == 819200, "TensorMap task-count offset mismatch");
static_assert(offsetof(TensorMap, live_count) == 823296, "TensorMap live-count offset mismatch");
static_assert(offsetof(TensorMap, high_water) == 823300, "TensorMap high-water offset mismatch");
static_assert(offsetof(TensorMap, alive_floor) == 823304, "TensorMap alive-floor offset mismatch");
static_assert(offsetof(TensorMap, cleaned_upto) == 823308, "TensorMap cleaned offset mismatch");
#elif PTO_FDWIC_TENSORMAP_RING_CAP == 64
static_assert(offsetof(TensorMap, extra_bucket_heads) == 788480, "CAP64 extra-head offset mismatch");
static_assert(offsetof(TensorMap, extra_bucket_tails) == 789504, "CAP64 extra-tail offset mismatch");
static_assert(offsetof(TensorMap, abi_reserved) == 790528, "CAP64 reserve offset mismatch");
static_assert(offsetof(TensorMap, task_entry_counts) == 819200, "CAP64 task-count offset mismatch");
#elif PTO_FDWIC_TENSORMAP_RING_CAP == 32
static_assert(offsetof(TensorMap, extra_bucket_heads) == 788480, "CAP32 extra-head offset mismatch");
static_assert(offsetof(TensorMap, extra_bucket_tails) == 791552, "CAP32 extra-tail offset mismatch");
static_assert(offsetof(TensorMap, abi_reserved) == 794624, "CAP32 reserve offset mismatch");
static_assert(offsetof(TensorMap, task_entry_counts) == 819200, "CAP32 task-count offset mismatch");
#endif

// shared 模式只共享 region→producer 索引，不复用 private MapEntry 尾部的
// ABI 保留字节。payload 与 seq 各占一条 cache line：普通字段写回完成后，
// 再用独立的绝对 seq 发布该 lap，reader 以 seq 双检防止槽复用 ABA。
struct SharedRegionValue {
    uint64_t buffer_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    uint32_t reserved;
};
static_assert(sizeof(SharedRegionValue) == 32, "shared TensorMap logical value size changed");
static_assert(offsetof(SharedRegionValue, producer) == 24, "shared producer offset mismatch");

struct alignas(64) SharedRegionPayload {
    SharedRegionValue value;
    uint8_t cacheline_padding[32];
};
static_assert(sizeof(SharedRegionPayload) == 64, "shared TensorMap payload must occupy one cache line");
static_assert(alignof(SharedRegionPayload) == 64, "shared TensorMap payload alignment changed");

struct alignas(64) SharedRegionSlot {
    SharedRegionPayload payload;
    AtomicLine seq;
};
static_assert(sizeof(SharedRegionSlot) == 128, "shared TensorMap slot must occupy two cache lines");
static_assert(offsetof(SharedRegionSlot, seq) == 64, "shared TensorMap seq must own the second cache line");

struct alignas(64) SharedBucketState {
    AtomicLine head;
    AtomicLine tail;
};
static_assert(sizeof(SharedBucketState) == 128, "shared TensorMap bucket controls changed");
static_assert(offsetof(SharedBucketState, tail) == 64, "shared head/tail must not share a cache line");

#if PTO_FDWIC_SHARED_MAP
// fresh Output 不再借助 region map 按地址查找，而是由
// (producer_task_id, output_slot) 直接定位。descriptor 发布位、writer 链
// 和不可变 descriptor 分属独立 cache line 区域，避免三种访问彼此伪共享。
// generic shared 仍逐 slot 使用 last_writer；正式 PA generation 12 证明
// 三个 accumulator lockstep 后，仅把 Alloc slot0 解释为 group latest。
// shared 最大覆盖 256 batch × 4 block group；本轮 task_id 不复用，
// 因此外层表直接寻址，不做取模，也不在这里提前引入 generation。
struct alignas(64) SharedOutputCell {
    AtomicLine published[kSharedOutputMaxPerTask];
    AtomicLine last_writer[kSharedOutputMaxPerTask];
    TensorDesc tensors[kSharedOutputMaxPerTask];
};
static_assert(sizeof(SharedOutputCell) == 2048, "shared output cell size changed");
static_assert(alignof(SharedOutputCell) == 64, "shared output cell alignment changed");
static_assert(offsetof(SharedOutputCell, published) == 0, "shared output publish offset mismatch");
static_assert(offsetof(SharedOutputCell, last_writer) == 512, "shared output writer offset mismatch");
static_assert(offsetof(SharedOutputCell, tensors) == 1024, "shared output tensor offset mismatch");

// latest writer 只是正常顺序 reader 的快取；慢 reader 若在查询前遇到
// future writer，必须沿不可变前驱链回到严格早于自己的版本。每个 writer
// task 独占一个 history cell，record 的 writer id 由 cell 下标隐含；
// packed key 无哈希碰撞，可还原为 fresh descriptor 的 (producer, slot)。
constexpr uint32_t kSharedWriterHistoryMaxPerTask = kMaxTaskTensors;
constexpr uint32_t kSharedWriterHistoryMagic = 0x57484953U;  // "WHIS"
struct SharedWriterHistoryRecord {
    uint32_t symbol_key;
    int32_t previous_writer;
};
static_assert(sizeof(SharedWriterHistoryRecord) == 8, "shared writer-history record size changed");
static_assert(alignof(SharedWriterHistoryRecord) == 4, "shared writer-history record alignment changed");

struct alignas(64) SharedWriterHistoryCell {
    // header 与常见的三个 PA writer record 共处首条 cache line；唯一
    // winner 一次写回这段不可变 payload，随后各 symbol 的 last_writer
    // CAS 才是 reader 的发布边界，不额外增加 history atomic。
    uint32_t magic;
    int32_t writer_task;
    uint32_t count;
    uint32_t reserved;
    SharedWriterHistoryRecord entries[kSharedWriterHistoryMaxPerTask];
    uint8_t padding[48];
};
static_assert(sizeof(SharedWriterHistoryCell) == 320, "shared writer-history cell size changed");
static_assert(alignof(SharedWriterHistoryCell) == 64, "shared writer-history cell alignment changed");
static_assert(
    offsetof(SharedWriterHistoryCell, entries) == 16,
    "shared writer-history entries must follow their immutable header"
);
static_assert(
    kSharedOutputMaxPerTask <= 8 &&
        kMaxTasks <= UINT32_MAX / kSharedOutputMaxPerTask,
    "packed shared symbol key no longer fits uint32"
);
#endif

struct alignas(64) SharedTensorMapSidecar {
    // lane 0 保留既有 committed_tasks 地址：G=1 时它仍表示下一个允许
    // 插入 writer 元数据的 task id。G>1 时它只是交错 token 的 lane 0；
    // 其余七条物理线追加在 sidecar 尾部，所有既有热点字段 offset 不动。
    AtomicLine committed_tasks;
    // reclaim_upto 是可回收 producer 的 inclusive 上界，初始 -1。
    // insert-before-lookup 基线固定不回收 ordinary ring，因此保持 -1；
    // reader-progress/reclaim 只由隔离测试覆盖，尚未接回当前热路径。
    AtomicLine reclaim_upto;
    SharedBucketState buckets[kMapBuckets];
    SharedRegionSlot slots[kMapCapacity];
#if PTO_FDWIC_SHARED_MAP
    // 追加在既有 S2.5 region ring 之后，保持 committed/reclaim、bucket 和
    // slot 的全部 offset 不变。容量按 shared 最坏 17 task/batch 分配；
    // 现有 shared sidecar H2D/D2H 按 sizeof 搬运。
    SharedOutputCell shared_outputs[kMaxTasks];
    // shared heap 控制字继续追加在 S3.1 output table 之后。每个 shard cursor
    // 与全局 aggregate vend 独占 cache line，避免不同 winner 的原子更新伪共享。
    AtomicLine shared_heap_cursor[kSharedHeapShards];
    AtomicLine shared_heap_vend;
    // S4.14a 的 shared-only Vector Claim cursor 追加在既有 sidecar 尾部；
    // S4.14b 只启用此前已经预留的后四条物理线。production prefix 和
    // 已验证的 region/output/heap 字段仍不移动，且不宣称该地址与参考
    // DistGlobal 具有相同字节 offset。
    AtomicLine shared_vector_cursor[kSharedVectorCursorCapacity];
    // 追加在全部既有字段之后，避免为通用 writer history 移动 PA 已测
    // 热点控制字。当前 task id 在一轮内不复用，因此 history 不取模；
    // 1.33 MiB 增量只影响启动期整块搬运，不改变 Submit 内旧字段地址。
    SharedWriterHistoryCell writer_history[kMaxTasks];
    // ordinary reader 的完成前沿不能复用 WorkerState::local_index：后者在
    // 读取前就会推进，且 96 个字段分散在近 1 GiB 的 per-worker arena。
    // 每个 worker 独占一条连续 cache line；初值 -1，值 N 只表示该 worker
    // 已关闭 task [0,N] 的全部 ordinary-ring 读取。R4e-a 只建立状态与
    // 纯公式门槛，尚不从 PA 或通用 Submit 热路径发布该字段。
    AtomicLine reader_done[kWorkers];
    // insert_turn_extra[0..126] 对应逻辑 lane 1..127。初始值全部为 -1；
    // active G 只决定前 G 条逻辑 lane 的寻址，inactive 物理线始终保持
    // -1。每个 owner（包括空写集合）发布完整元数据后，只把 baton 从
    // task N 轮换为 N+1；fanin、Build 与执行不属于这条顺序链。
    AtomicLine insert_turn_extra[
        kSharedInsertTurnCapacity - 1
    ];
#endif
};
#if PTO_FDWIC_SHARED_MAP
static_assert(
    offsetof(SharedTensorMapSidecar, reader_done) ==
        offsetof(SharedTensorMapSidecar, writer_history) +
            sizeof(SharedWriterHistoryCell) * kMaxTasks,
    "shared reader progress must immediately follow writer history"
);
static_assert(
    sizeof(SharedTensorMapSidecar) ==
        offsetof(SharedTensorMapSidecar, reader_done) +
            sizeof(AtomicLine) * kWorkers +
            sizeof(AtomicLine) *
                (kSharedInsertTurnCapacity - 1),
    "shared insert-turn lines must remain the sidecar tail"
);
static_assert(
    offsetof(SharedTensorMapSidecar, insert_turn_extra) ==
        offsetof(SharedTensorMapSidecar, reader_done) +
            sizeof(AtomicLine) * kWorkers,
    "extra shared insert-turn lines must follow reader progress"
);
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(sizeof(SharedTensorMapSidecar) == 12434560, "shared TensorMap sidecar size changed");
static_assert(
    offsetof(SharedTensorMapSidecar, shared_outputs) == 2113664,
    "shared output table offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_cursor) == 11026560,
    "shared heap cursor offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_vend) == 11027072,
    "shared heap vend offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_vector_cursor) == 11027136,
    "shared Vector cursor offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, writer_history) == 11027648,
    "shared writer-history tail offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, reader_done) == 12420288,
    "shared reader-progress tail offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, insert_turn_extra) ==
        12426432,
    "shared insert-turn tail offset mismatch"
);
#endif
#else
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(sizeof(SharedTensorMapSidecar) == 2113664, "private sidecar layout changed");
#endif
#endif
static_assert(alignof(SharedTensorMapSidecar) == 64, "shared TensorMap sidecar alignment changed");
static_assert(
    offsetof(SharedTensorMapSidecar, buckets) == 128,
    "shared TensorMap bucket offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, slots) ==
        128 + sizeof(SharedBucketState) * kMapBuckets,
    "shared TensorMap slots must immediately follow all bucket controls"
);
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(offsetof(SharedTensorMapSidecar, slots) == 16512, "shared TensorMap slot offset changed");
#endif

struct TaskPayload {
    TensorDesc tensors[kMaxTaskTensors];
};
// task_id 通过 kPayloadMask 映射到 2048 个 4 KiB payload。现有单组 B256
// 只有 1280 个 task，不会回绕；未来 shared 多组允许回绕，但 payload
// 只活到本次 Finish 完成 descriptor 发布/slot 拷贝，后续依赖读取的是
// shared cell 或 LocalSlot 中的副本，不能跨 Submit 保存 payload 指针。
static_assert(sizeof(TaskPayload) == kPayloadStride, "TaskPayload must preserve the real 4 KiB task stride");
static_assert(alignof(TaskPayload) == 8, "TaskPayload alignment must match DistTaskPayload");
static_assert(offsetof(TaskPayload, tensors) == 0, "TaskPayload tensor offset mismatch");

struct LocalSlot {
    // occupied 先保留容量，built 表示 payload 已按生产顺序构建；task/function
    // 标识决定执行哪个 NOP 体，后续大数组则是 kernel 真正看到的参数快照。
    bool occupied;
    bool built;
    uint8_t header_padding[2];
    uint32_t task_id;
    uint32_t kind;
    uint32_t function_padding;
    uint64_t function_address;
    uint32_t tensor_count;
    uint32_t scalar_count;
    uint8_t tensor_padding[32];

    TensorDesc tensors[kMaxTaskTensors];
    uint64_t scalars[kMaxTaskScalars];
    uint64_t args[kMaxTaskTensors + kMaxTaskScalars + 2];
    union {
        struct {
            uint8_t local_context[48];
            uint32_t global_context;
            int32_t fanin[kMaxFanin];
            uint32_t fanin_count;
        };
        // Compatibility view used by the standalone NOP payload builder. The
        // first six words are the real 48-byte LocalContext; the remaining
        // words overlap GlobalContext and the beginning of fanin, exactly as
        // dictated by the real RingSlot offsets.
        // 该视图只用于按真实 offset 填充 dispatch context，不增加另一份
        // 存储；修改其中后两字会同步覆盖 GlobalContext/fanin 的对应 ABI 字节。
        uint64_t context_words[8];
    };
    bool is_multicore;
    int32_t won_block;
    int32_t won_slot;
};
// LocalSlot 是每个 winner 写入自己私有 ring 的完整 dispatch 包。fanin 在执行前
// 逐项检查 task.flag；occupied/built 与计数共同约束最多两个普通 kernel 在途。
static_assert(sizeof(LocalSlot) == 4824, "LocalSlot must match the PA RingSlot ABI size");
static_assert(alignof(LocalSlot) == 8, "LocalSlot alignment must match RingSlot");
static_assert(offsetof(LocalSlot, occupied) == 0, "LocalSlot occupied offset mismatch");
static_assert(offsetof(LocalSlot, built) == 1, "LocalSlot built offset mismatch");
static_assert(offsetof(LocalSlot, task_id) == 4, "LocalSlot task offset mismatch");
static_assert(offsetof(LocalSlot, kind) == 8, "LocalSlot function-id offset mismatch");
static_assert(offsetof(LocalSlot, function_address) == 16, "LocalSlot function address offset mismatch");
static_assert(offsetof(LocalSlot, tensor_count) == 24, "LocalSlot tensor-count offset mismatch");
static_assert(offsetof(LocalSlot, tensors) == 64, "LocalSlot tensor payload offset mismatch");
static_assert(offsetof(LocalSlot, scalars) == 4160, "LocalSlot scalar payload offset mismatch");
static_assert(offsetof(LocalSlot, args) == 4288, "LocalSlot dispatch-args offset mismatch");
static_assert(offsetof(LocalSlot, local_context) == 4688, "LocalSlot local-context offset mismatch");
static_assert(offsetof(LocalSlot, global_context) == 4736, "LocalSlot global-context offset mismatch");
static_assert(offsetof(LocalSlot, fanin) == 4740, "LocalSlot fanin offset mismatch");
static_assert(offsetof(LocalSlot, fanin_count) == 4804, "LocalSlot fanin-count offset mismatch");
static_assert(offsetof(LocalSlot, is_multicore) == 4808, "LocalSlot multicore offset mismatch");
static_assert(offsetof(LocalSlot, won_block) == 4812, "LocalSlot won-block offset mismatch");
static_assert(offsetof(LocalSlot, won_slot) == 4816, "LocalSlot won-slot offset mismatch");

struct WorkerState {
    CoreRole role;
    int32_t core_idx;
    int32_t block_id;
    int32_t lane;
    int32_t sub_block_id;
    int32_t local_index;
    uint64_t heap_next;
    TensorMap map;
    uint8_t slot_padding[16];
    LocalSlot slots[kPrivateSlots];
    uint32_t occupied_count;
    uint32_t owned_total;
    uint64_t swimlane_last_cycle;
    uint8_t payload_padding[16];
    TaskPayload payloads[kPayloadSlots];
};
// 每个物理 worker 都持有独立 heap cursor、TensorMap、ring 与 task payload arena；
// 多核共享的只有 SchedulerState 前缀中的 cursor/task/frontier 等协议状态。
static_assert(sizeof(WorkerState) == 9231296, "WorkerState must match the PA DistCore ABI size");
static_assert(alignof(WorkerState) == 8, "WorkerState alignment must match DistCore");
static_assert(offsetof(WorkerState, role) == 0, "WorkerState role offset mismatch");
static_assert(offsetof(WorkerState, local_index) == 20, "WorkerState replay-index offset mismatch");
static_assert(offsetof(WorkerState, heap_next) == 24, "WorkerState heap cursor offset mismatch");
static_assert(offsetof(WorkerState, map) == 32, "WorkerState tensor-map offset mismatch");
static_assert(offsetof(WorkerState, slots) == 823360, "WorkerState ring-slot offset mismatch");
static_assert(offsetof(WorkerState, occupied_count) == 842656, "WorkerState occupancy offset mismatch");
static_assert(offsetof(WorkerState, owned_total) == 842660, "WorkerState owned-count offset mismatch");
static_assert(offsetof(WorkerState, swimlane_last_cycle) == 842664, "WorkerState trace clock offset mismatch");
static_assert(offsetof(WorkerState, payloads) == 842688, "WorkerState task-payload offset mismatch");

struct alignas(64) WorkerResult {
    // 时间边界：Submit 口径不含启动屏障和最终 drain，finish_cycle 则覆盖完整 worker 生命周期。
    uint64_t submit_begin;
    uint64_t submit_end;
    uint64_t finish_cycle;
    uint64_t checksum;

    // 协议计数用于验证固定 Claim 拓扑及等待/依赖动态次数；joint_polls 是为未来
    // BlockWon 模拟保留的兼容字段，当前实现没有递增点，不能据其检测 joint 分支。
    uint64_t submits;
    uint64_t claim_attempts;
    uint64_t claim_wins;
    uint64_t heap_guards;
    uint64_t fanin_ready_loads;
    uint64_t completion_duplicates;
    uint64_t cas_retries;
    uint64_t joint_polls;

    // 默认 256 batch 时 winner、kernel 分别闭合到 1280 task 和 1024 kernel；
    // 非默认配置按 5*batches、4*batches 计算，placement 仍闭合到全部 kernel。
    uint64_t wins[static_cast<uint32_t>(TaskKind::Count)];
    uint64_t kernel_counts[4];
    uint64_t kernel_cycles[4];
    uint64_t kernel_min_cycles[4];
    uint64_t kernel_max_cycles[4];
    uint64_t placement[static_cast<uint32_t>(DrainPlace::Count)];
    uint64_t phase_cycles[static_cast<uint32_t>(ProfilePhase::Count)];
    uint64_t phase_calls[static_cast<uint32_t>(ProfilePhase::Count)];
    uint64_t wait_events[2];
    uint64_t wait_iterations[2];

    // 前端工作量计数不是性能填充：private 核对全员构参/物化；shared
    // 的五类 task 都核对 owner-only 重构参与 owner-only 物化。lookup、
    // slot copy 和 fanin 也按 owner 业务量闭合。
    uint64_t context_reads;
    uint64_t views_created;
    uint64_t dynamic_create_infos;
    uint64_t arg_resets;
    uint64_t tensor_args_added;
    uint64_t scalar_args_added;
    uint64_t materialized_outputs;
    uint64_t map_inserts;
    uint64_t map_lookups;
    uint64_t slot_tensor_copies;
    uint64_t slot_scalar_copies;
    uint64_t fanin_edges;

    // private 保存逐 worker 逻辑 heap/map 终态；shared 的 heap_next 只是本核
    // 最近一次 winner 看到的 aggregate prefix，权威终态位于 shared sidecar。
    uint64_t final_heap_next;
    uint64_t map_high_water;
    uint64_t map_alive_floor;
    uint64_t map_cleaned_upto;
    uint64_t map_live_entries;

    uint64_t worker_id;
    uint64_t role;
    uint64_t max_occupied;
    uint64_t final_occupied;

    // CCEC 标量 PMU 取证使用 WorkerResult 的诊断 sidecar，不改变生产 DistCore ABI。
    // 该诊断只在显式开启时有效；CNT2/CNT6/CNT7 分别对应 scalar busy、I-cache req/miss。
    uint64_t pmu_total_cycles;
    uint32_t pmu_scalar_busy;
    uint32_t pmu_icache_requests;
    uint32_t pmu_icache_misses;
    uint32_t pmu_status;

    // 这些计数只在 worker 私有 LocalStats 中递增，结束时一次性发布；它们把
    // 动态 fanin 重试和 private frontier helping 展开为准确次数。shared
    // no-wrap 构建要求三个 frontier 计数全零；取数本身不增加共享 atomic。
    uint64_t fanin_not_ready_loads;
    uint64_t frontier_initial_loads;
    uint64_t frontier_updates;
    uint64_t frontier_terminal_loads;

    // 仅在 trace_enabled bit1 开启时递增；每次源码 atomic 调用恰好增加一，
    // host 用它与 Atomic span 数逐 worker 闭合，禁止把丢记录的泳道当成完整结果。
    uint64_t atomic_trace_calls;

    // I-cache 单 miss 探针用该槽保存 cold 窗口的 1 GHz SYS_CNT；submit-pmu
    // 复用同一 64-bit 槽保存所选局部阶段的逐调用累计时间。两种构建互斥，
    // 因而无需扩大当前 896B WorkerResult，也不会改变相邻 worker 的 cache-line 布局。
    union {
        uint64_t pmu_window_ticks;
        uint64_t pmu_phase_elapsed_ticks;
    };
    uint64_t pmu_warm_total_cycles;
    uint64_t pmu_warm_window_ticks;
    union {
        uint32_t pmu_warm_icache_requests;
        uint32_t pmu_phase_begin_reads;
    };
    union {
        uint32_t pmu_warm_icache_misses;
        uint32_t pmu_phase_end_reads;
    };

    // PIPE_UTILIZATION 已同时配置 CNT0/1/3/4/5/8；与上面的 scalar/I-cache
    // 一样只保存每核原始累计值，AIC/AIV 汇总与比率统一在 host sidecar 中计算。
    // 六个 32-bit 值复用本结构既有 PMU 诊断区，不再增加 cache line。
    uint32_t pmu_vector_busy;
    uint32_t pmu_cube_busy;
    uint32_t pmu_mte1_busy;
    uint32_t pmu_mte2_busy;
    // swimlane ABI 保留该槽；submit-pmu 将物理 CNT5 改作 shadow miss，
    // 因而显式发布 0，并在当前 submit-pmu schema-v5 标记 mte3_busy 不可用。
    uint32_t pmu_mte3_busy;
    uint32_t pmu_fix_busy;

    // 复用 WorkerResult 原有的 32B cache-line 尾洞，不扩大 896B stride。
    // CNT6/7 是从不中途读取的权威整窗，CNT8/CNT5 是 read-to-clear shadow；
    // none 在 stop 后要求逐核精确相等；运行中切片的 phase 只允许 shadow
    // 单向小于 primary，并显式导出差值形成局部观测区间。
    uint32_t pmu_build_variant;
    uint32_t pmu_phase_id;
    uint32_t pmu_phase_calls;
    uint32_t pmu_phase_status;
    uint32_t pmu_phase_icache_requests;
    uint32_t pmu_phase_icache_misses;
    uint32_t pmu_shadow_icache_requests;
    uint32_t pmu_shadow_icache_misses;

    // 生命周期屏障的 SYS_CNT 边界属于 standalone 诊断 sidecar，不进入生产
    // DistCore ABI。final release 与 final end 分开，保留“全局停止生产”和
    // “本核 drain 完成”两个不同事件。
    uint64_t startup_barrier_begin;
    uint64_t startup_barrier_end;
    uint64_t final_barrier_begin;
    uint64_t final_barrier_release;
    uint64_t final_barrier_end;
    // 复用原 barrier_reserved[3] 的 24B，不扩大 WorkerResult。dependency
    // signature 继续闭合 fanin 拓扑；后两项统计 shared symbol 的
    // last_writer INPUT load 和构建后逻辑 INOUT symbol commit，private
    // 构建必须保持零。正式 PA generation 12 的三个 lockstep symbol
    // 共用一次物理 group CAS，但逻辑提交数仍为三。
    uint64_t dependency_signature;
    uint64_t shared_symbol_input_loads;
    uint64_t shared_symbol_inout_commits;
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    // split 协议诊断独占一条 cache line。普通 CPU/AscendC 与局部 PMU
    // 构建不带这些字段，不改变它们的 WorkerResult ABI。shared 下
    // finish_calls 只统计跨 TU 的 winner Finish；task_id_sum 则由 caller
    // 统计完整逻辑 replay，二者有意采用不同粒度。
    uint64_t compete_first_split_caller_state_address;
    uint64_t compete_first_split_finish_state_address;
    uint64_t compete_first_split_finish_calls;
    uint64_t compete_first_split_protocol_errors;
    uint64_t compete_first_split_state_cookie;
    uint64_t compete_first_split_task_id_sum;
    uint64_t compete_first_split_owner_worker_id;
    uint64_t compete_first_split_reserved;
#endif
};
// WorkerResult 是 standalone 尾部的诊断 sidecar，不属于真实 DistCore ABI；按
// cache line 隔离后，各 worker 发布统计不会相互覆盖或污染被测共享状态。
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
static_assert(sizeof(WorkerResult) == 960, "split WorkerResult diagnostics must occupy whole cache lines");
static_assert(offsetof(WorkerResult, compete_first_split_caller_state_address) == 896,
              "split WorkerResult oracle offset mismatch");
static_assert(offsetof(WorkerResult, compete_first_split_reserved) == 952,
              "split WorkerResult oracle tail mismatch");
#else
static_assert(sizeof(WorkerResult) == 896, "WorkerResult diagnostics must occupy whole cache lines");
#endif
static_assert(offsetof(WorkerResult, pmu_total_cycles) == 680, "WorkerResult PMU offset mismatch");
static_assert(offsetof(WorkerResult, pmu_status) == 700, "WorkerResult PMU status offset mismatch");
static_assert(offsetof(WorkerResult, fanin_not_ready_loads) == 704, "WorkerResult atomic diagnostic offset mismatch");
static_assert(offsetof(WorkerResult, atomic_trace_calls) == 736, "WorkerResult atomic trace offset mismatch");
static_assert(offsetof(WorkerResult, pmu_window_ticks) == 744, "WorkerResult PMU timing offset mismatch");
static_assert(offsetof(WorkerResult, pmu_vector_busy) == 776, "WorkerResult extended PMU offset mismatch");
static_assert(offsetof(WorkerResult, pmu_build_variant) == 800, "WorkerResult submit-PMU offset mismatch");
static_assert(offsetof(WorkerResult, pmu_shadow_icache_misses) == 828, "WorkerResult submit-PMU tail mismatch");
static_assert(offsetof(WorkerResult, dependency_signature) == 872, "WorkerResult dependency signature offset mismatch");
static_assert(offsetof(WorkerResult, shared_symbol_input_loads) == 880,
              "WorkerResult shared symbol-load offset mismatch");
static_assert(
    offsetof(WorkerResult, shared_symbol_inout_commits) == 888,
    "WorkerResult shared symbol-commit offset mismatch"
);

// 从 cube_cursor 到 workers 结束保留关键字段 offset、DistCore ABI 和生产总字节跨度，
// 并非字段级完整镜像。RunConfig、输入 context_lens 与校验结果追加在该跨度之后，
// 因此测试控制信息不会改变被测字段 offset。
struct alignas(64) SchedulerState {
    // production prefix 的三组四分片 cursor 服务 AIC、private AIV 与
    // Alloc；shared AIV 继续使用 sidecar 尾部的 Vector cursor，
    // S4.14b 启用全部八条物理线。
    // 同 task 的 eligible workers 仍竞争同一 shard，只有旧值小于
    // task_id 的调用成为 winner。
    AtomicLine cube_cursor[kCursorShards];
    AtomicLine vector_cursor[kCursorShards];
    AtomicLine alloc_cursor[kCursorShards];
    AtomicLine frontier;
    int32_t heap_window;
    uint8_t tasks_padding[60];
    TaskCell tasks[kTaskCellCapacity];
    // heap_base/size 描述共享物理区间。private 的 worker.heap_next 是各核
    // 独立回放的 ring 逻辑游标；shared 下它只镜像本核最近 winner 的 vend。
    uint64_t heap_base;
    uint64_t heap_size;
    uint64_t orchestration_args;
    uint64_t runtime_state;
    uint64_t runtime;
    uint8_t fatal_padding[24];
    AtomicFlagLine fatal;
    int32_t num_workers;
    int32_t num_blocks;
    // Case1 never enters BlockWon, but the inactive layout and BlockWon arena
    // remains byte-for-byte reserved so every subsequent PA atomic line keeps
    // its production offset.
    // 此处不能因 Case1 动态次数为零而删减，否则 replay_done、started_count
    // 和 DistCore 数组整体前移，便不再是对真实 PA 地址布局的等价测试。
    uint8_t layout_and_block_won[5848440];
    AtomicLine replay_done;
    AtomicLine started_count;
    // started_count 形成 launch 屏障。生产路径已将 final 汇合迁移到
    // DistGlobal 尾部的固定 G=16 树，replay_done 原位保留以维持后续字段 ABI。
    WorkerState workers[kRuntimeMaxWorkers];
    // 生产 DistGlobal 在 cores 后追加的固定 G=16 final 树。standalone
    // 自己的五形态实验状态仍放在 controls 之后，两者不混用。
    uint8_t production_final_barrier[kRealFinalBarrierBytes];
    // Standalone-only controls live after the complete DistGlobal image. They
    // therefore do not shift any cursor/task/fatal/worker address under test.
    RunConfig config;
    PmuProbeConfig pmu_probe;
    WinnerWorkloadConfig winner_workload;
    // Context lengths are the only PA input elements read by orchestration;
    // keeping them in GM preserves the per-batch descriptor-based load.
    // 除这些 batch 长度值外，其余 tensor 仅需稳定的合成地址来复现
    // descriptor、依赖和 heap 行为，不会解引用成真实计算数据。
    volatile int32_t context_lens[kMaxBatches];
    // standalone 的 final 分层实验状态位于完整生产 DistGlobal 镜像之后，
    // 不移动任何被测生产字段；startup 继续使用生产 started_count。
    FinalBarrierState final_barrier;
    WorkerResult results[kWorkers];
#if PTO_FDWIC_SHARED_MAP
    // shared 后端状态只追加在完整 production prefix、standalone controls
    // 和 results 之后，不移动 RunConfig、WorkerState 或任何被测生产字段。
    SharedTensorMapSidecar shared_map;
#endif
};
static_assert(offsetof(SchedulerState, cube_cursor) == 0, "cube cursor offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, vector_cursor) == 256, "vector cursor offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, alloc_cursor) == 512, "alloc cursor offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, frontier) == 768, "frontier offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, heap_window) == 832, "H offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, tasks) == kRealTasksOffset, "task table offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, heap_base) == 4195200, "heap base offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, heap_size) == 4195208, "heap size offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, fatal) == kRealFatalOffset, "fatal offset must match PA DistGlobal");
static_assert(offsetof(SchedulerState, replay_done) == kRealReplayDoneOffset, "replay offset must match PA DistGlobal");
static_assert(
    offsetof(SchedulerState, started_count) == kRealStartedCountOffset,
    "started-count offset must match PA DistGlobal"
);
static_assert(offsetof(SchedulerState, tasks) % 64 == 0, "task table must be cache-line aligned");
static_assert(offsetof(SchedulerState, workers) % 64 == 0, "worker table must be cache-line aligned");
static_assert(offsetof(SchedulerState, results) % 64 == 0, "result table must be cache-line aligned");
static_assert(offsetof(SchedulerState, workers) == kRealDistCoreOffset, "DistCore table offset must match PA");
static_assert(offsetof(SchedulerState, config) == kRealDistGlobalBytes, "DistGlobal byte size must match PA");
static_assert(
    offsetof(SchedulerState, pmu_probe) == kRealDistGlobalBytes + sizeof(RunConfig),
    "PMU probe sidecar must follow RunConfig"
);
static_assert(
    offsetof(SchedulerState, winner_workload) ==
        kRealDistGlobalBytes + sizeof(RunConfig) + sizeof(PmuProbeConfig),
    "winner workload sidecar offset mismatch"
);
static_assert(
    offsetof(SchedulerState, context_lens) ==
        kRealDistGlobalBytes + sizeof(RunConfig) + sizeof(PmuProbeConfig) +
            sizeof(WinnerWorkloadConfig),
    "context lengths must follow standalone controls"
);
static_assert(offsetof(SchedulerState, final_barrier) % 64 == 0, "final barrier must be cache-line aligned");
#if PTO_FDWIC_SHARED_MAP
static_assert(
    offsetof(SchedulerState, shared_map) ==
        offsetof(SchedulerState, results) + sizeof(WorkerResult) * kWorkers,
    "shared TensorMap sidecar must follow the complete result array"
);
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
static_assert(
    sizeof(SchedulerState) == 1019557696,
    "shared split SchedulerState ABI changed"
);
#else
static_assert(
    sizeof(SchedulerState) == 1019551552,
    "shared non-split SchedulerState ABI changed"
);
#endif
#endif
#endif
static_assert(sizeof(SchedulerState) <= UINT32_MAX, "SchedulerState size must fit build identity");

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_MODEL_H
