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

#if (PA_BUILD_SWIMLANE + PA_BUILD_SUBMIT_PMU + PA_BUILD_PERF_CLOCK) > 1
#error "swimlane, submit-pmu, and perf-clock builds are mutually exclusive"
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
constexpr uint32_t kBuildIdentityAbiVersion = 4;

// 这里固定的是 PA Case1 的调度拓扑，而不是为了缩小 standalone 人为选择的规模：
// 每个 batch 依次回放 Alloc/QK/SF/PV/UP 五个 task，32 个 AIC 与 64 个 AIV
// 都执行同一条 orchestration 流，只在 Claim 时按 task 的 active role 分流。
constexpr uint32_t kDefaultBatches = 256;
constexpr uint32_t kMaxBatches = 256;
constexpr uint32_t kTasksPerBatch = 5;
constexpr uint32_t kMaxTasks = kMaxBatches * kTasksPerBatch;
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
constexpr uint64_t kSyntheticHeapBase = 0x100000000ULL;
constexpr uint64_t kOutputAlignment = 1024;
constexpr uint32_t kMaxTensorDims = 5;
constexpr uint32_t kMaxTaskTensors = 32;
constexpr uint32_t kMaxTaskScalars = 16;
constexpr uint32_t kSharedOutputMaxPerTask = 8;
constexpr uint32_t kPayloadSlots = 2048;
constexpr uint32_t kPayloadMask = kPayloadSlots - 1;
constexpr uint32_t kPayloadStride = 4096;
// private/shared 最终统一为 ring-per-bucket。S1 先在 private standalone
// 验证无原子版本：128 个桶，每桶 128 个连续槽，容量仍与旧 linked map
// 完全相同。PA Case1 在 H=64 的窗口内全 map 最多只有 52 个 live entry，
// 因而单桶最坏聚集也小于 128；这个证明只适用于当前 Case1，不能据此
// 把 128 当成任意任务图的通用容量。
constexpr uint32_t kMapBuckets = 128;
constexpr uint32_t kMapBucketCapacity = 128;
constexpr uint32_t kMapCapacity = kMapBuckets * kMapBucketCapacity;
constexpr uint32_t kMapBucketShift = 7;
constexpr uint32_t kMapBucketMask = kMapBuckets - 1;
constexpr uint32_t kMapBucketSlotMask = kMapBucketCapacity - 1;
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
constexpr uint32_t kTraceRecordsPerCore = 1U << 16;
static_assert((kPayloadSlots & kPayloadMask) == 0, "payload slots must be a power of two");
static_assert((kMapBuckets & kMapBucketMask) == 0, "map bucket count must be a power of two");
static_assert(
    (kMapBucketCapacity & kMapBucketSlotMask) == 0,
    "map bucket capacity must be a power of two"
);
static_assert(kMapCapacity == 16384, "private ring must preserve the old map capacity");
static_assert(
    kPaCase1MaxLiveMapEntries <= kMapBucketCapacity,
    "PA Case1 worst-case bucket occupancy exceeds private ring capacity"
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

// task_id % 5 即 kind；该周期性是不变量，既决定 Claim cursor/active role，
// 也决定输出大小、fanin 拓扑和 winner workload 的选择。
enum class TaskKind : uint32_t {
    Alloc = 0,
    Qk = 1,
    Sf = 2,
    Pv = 3,
    Up = 4,
    Count = 5,
};

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
    // schema-v4 追加的父区间与真实动作区间。loser 没有可单列的真实动作，
    // 其时间直接归入离线计算的 Submit residual，不占用 raw 记录。
    OrchestrationReplay = 16,
    FinalDrain = 17,
    WinnerBuild = 18,
    AllocComplete = 19,
    Count = 20,
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
    Count = 19,
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
    return AtomicPollBatchIndex(site) >= 0;
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
    // 拓扑在一个 worker 分区内恒定；当前仍保留 64B TraceRecord 兼容布局，
    // 但在 core state 再保存一份权威身份，host 会验证每条记录与之相符。
    volatile int32_t core_idx;
    volatile int32_t block_id;
    volatile int32_t lane;
    uint32_t padding[8];
};
// 每个 worker 独占一个计数 cache line 和一段定长 records，不需要为了写 trace
// 再引入跨核 atomic；满容量后只增加本 worker 的 dropped。
static_assert(sizeof(TraceCoreState) == 64, "trace core state must occupy one cache line");

struct alignas(64) TraceHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_cores;
    uint32_t records_per_core;
    uint64_t frequency_hz;
    TraceCoreState cores[kRuntimeMaxWorkers];
};
// 本 benchmark 固定物理分配 kWorkers=96 个定长 record 分区，合法 header 也要求
// num_cores==96；其 header/record 布局和 phase 编号可转换为真实泳道使用的 JSON。
static_assert(sizeof(TraceHeader) == 6976, "trace header must match PA swimlane layout");

struct alignas(64) TraceRecord {
    uint64_t start_cycle;
    uint64_t end_cycle;
    int32_t task_id;
    int32_t function_id;
    int32_t phase;
    int32_t lane;
    int32_t block_id;
    int32_t core_idx;
    uint32_t flags;
    uint32_t auxiliary;
};
// start/end 保留原始 1 GHz counter；task/function/物理 lane 用于离线还原轨道。
// flags/aux 的含义由 phase 决定，例如 winner、Alloc 或 RingBp 类型，不参与调度决策。
static_assert(sizeof(TraceRecord) == 64, "trace record must occupy one cache line");

constexpr size_t kTraceBytes =
    sizeof(TraceHeader) + static_cast<size_t>(kWorkers) * kTraceRecordsPerCore * sizeof(TraceRecord);

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
    uint8_t padding[64 - 2 * sizeof(int64_t)];
};
// flag 是两种模式的依赖就绪发布位；vend 是该 task 完成时 worker 的 heap
// 快照。private 还用 flag 连续推进 frontier，并由 HeapGuard 读取
// frontier-H 对应 vend 判断环形 heap 是否可覆盖；shared no-wrap 中 vend
// 只是 aggregate-vend 快照，flag 只服务 fanin/slot，均不参与 heap 回收。
static_assert(sizeof(TaskCell) == 64, "TaskCell must occupy one cache line");

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
    // private map 由单 worker 访问，head/tail 都是普通单调整数；槽地址为
    // bucket*kMapBucketCapacity + (cursor & kMapBucketSlotMask)。
    uint64_t bucket_heads[kMapBuckets];
    uint64_t bucket_tails[kMapBuckets];
    // 复用旧 buckets[8192] 所占的 32 KiB，使下方逐 task 计数与控制字
    // 继续落在旧 task_heads/尾部控制区，减小 standalone ABI 扰动。
    uint8_t abi_reserved[30720];
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
static_assert(sizeof(TensorMap) == 823312, "TensorMap must match the PA fixed-capacity layout");
static_assert(alignof(TensorMap) == 8, "TensorMap alignment changed");
static_assert(offsetof(TensorMap, bucket_heads) == 786432, "TensorMap head offset mismatch");
static_assert(offsetof(TensorMap, bucket_tails) == 787456, "TensorMap tail offset mismatch");
static_assert(offsetof(TensorMap, abi_reserved) == 788480, "TensorMap reserve offset mismatch");
static_assert(offsetof(TensorMap, task_entry_counts) == 819200, "TensorMap task-count offset mismatch");
static_assert(offsetof(TensorMap, live_count) == 823296, "TensorMap live-count offset mismatch");
static_assert(offsetof(TensorMap, high_water) == 823300, "TensorMap high-water offset mismatch");
static_assert(offsetof(TensorMap, alive_floor) == 823304, "TensorMap alive-floor offset mismatch");
static_assert(offsetof(TensorMap, cleaned_upto) == 823308, "TensorMap cleaned offset mismatch");

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
// standalone 本阶段最多 1280 个 task，且不复用 task id，因此外层表直接按
// task_id 寻址，不做取模，也不在这里提前引入 generation。
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
#endif

struct alignas(64) SharedTensorMapSidecar {
    // committed_tasks/reclaim_upto 只属于未来非空 ordinary-region 的有序
    // ring 协议。PA Case1 的输入全部是 fresh symbol，唯一普通 output_view
    // 又标记 manual_dep，因此运行期不再触碰这两个控制字。
    AtomicLine committed_tasks;
    // reclaim_upto 是可回收 producer 的 inclusive 上界，初始 -1。
    // 通用 ring 定向测试仍验证 exact-turn/reclaim helper，但不能把该测试
    // 解释成 PA Case1 热路径仍经过全局 sequencer。
    AtomicLine reclaim_upto;
    SharedBucketState buckets[kMapBuckets];
    SharedRegionSlot slots[kMapCapacity];
#if PTO_FDWIC_SHARED_MAP
    // 追加在既有 S2.5 region ring 之后，保持 committed/reclaim、bucket 和
    // slot 的全部 offset 不变；现有 shared sidecar H2D/D2H 按 sizeof 搬运。
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
#endif
};
#if PTO_FDWIC_SHARED_MAP
static_assert(sizeof(SharedTensorMapSidecar) == 4736192, "shared TensorMap sidecar size changed");
static_assert(
    offsetof(SharedTensorMapSidecar, shared_outputs) == 2113664,
    "shared output table offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_cursor) == 4735104,
    "shared heap cursor offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_vend) == 4735616,
    "shared heap vend offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_vector_cursor) == 4735680,
    "shared Vector cursor offset mismatch"
);
#else
static_assert(sizeof(SharedTensorMapSidecar) == 2113664, "private sidecar layout changed");
#endif
static_assert(alignof(SharedTensorMapSidecar) == 64, "shared TensorMap sidecar alignment changed");
static_assert(
    offsetof(SharedTensorMapSidecar, buckets) == 128,
    "shared TensorMap bucket offset mismatch"
);
static_assert(
    offsetof(SharedTensorMapSidecar, slots) == 16512,
    "shared TensorMap slot offset mismatch"
);

struct TaskPayload {
    TensorDesc tensors[kMaxTaskTensors];
};
// task_id 通过 kPayloadMask 映射到 2048 个 4 KiB payload；Case1 只有 1280 个 task，
// 本轮不会回绕，但仍保留生产容量、寻址方式和 4 KiB stride。
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
    // 区分全员 Alloc 轻构参、四个 task 的 winner-only 重构参与
    // winner-only 物化。lookup、slot copy 和 fanin 也按 winner 业务量闭合。
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
    // last_writer INPUT load 和构建后 INOUT writer commit，private 构建
    // 必须保持零。
    uint64_t dependency_signature;
    uint64_t shared_symbol_input_loads;
    uint64_t shared_symbol_inout_commits;
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    // split 协议诊断独占一条 cache line。普通 CPU/AscendC 与局部 PMU
    // 构建不带这些字段，不改变它们的 WorkerResult ABI。
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
    // 除这 256 个长度值外，其余 tensor 仅需稳定的合成地址来复现
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
#endif
static_assert(sizeof(SchedulerState) <= UINT32_MAX, "SchedulerState size must fit build identity");

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_MODEL_H
