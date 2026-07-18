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

namespace pa_scheduler {

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
constexpr uint32_t kPayloadSlots = 2048;
constexpr uint32_t kPayloadMask = kPayloadSlots - 1;
constexpr uint32_t kPayloadStride = 4096;
constexpr uint32_t kMapCapacity = 16384;
constexpr uint32_t kMapBuckets = 1 << 13;
constexpr uint32_t kMapBucketShift = 13;
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
constexpr size_t kRealDistGlobalBytes = 1007023872;
constexpr size_t kRealTasksOffset = 896;
constexpr size_t kRealFatalOffset = 4195264;
constexpr size_t kRealReplayDoneOffset = 10043776;
constexpr size_t kRealStartedCountOffset = 10043840;
constexpr uint32_t kTraceRecordsPerCore = 1U << 16;
static_assert((kPayloadSlots & kPayloadMask) == 0, "payload slots must be a power of two");
static_assert(kMaxTasks < kTaskCellCapacity, "every frontier scan must terminate on an in-range not-ready flag");

// These are the measured means from the best PA A5 trace, in 1 GHz ticks.
// The CCEC stage calibrates the NOP counts against these targets before the
// defaults are considered final.
// 这里只用 NOP 代替四个计算 kernel 的执行体；Submit、依赖、heap 与
// completion 路径均不靠 NOP 补时。target 是真实泳道均值，不是调度阶段预算。
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

// task_id % 5 即 kind；该周期性是不变量，既决定 Claim cursor/active role，
// 也决定输出大小、fanin 拓扑和 NOP kernel 的选择。
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

struct NopCounts {
    uint32_t qk;
    uint32_t sf;
    uint32_t pv;
    uint32_t up;
};

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
    uint32_t reserved[5];
};
static_assert(sizeof(RunConfig) == 64, "RunConfig must occupy one cache line");

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
    Count = 15,
};

// Atomic 记录 flags 的低四位保存操作种类；bit4 表示返回值参与后续判断，
// bit5 表示 Load 观察到零，bit6 表示结束时间已由返回值依赖推进到
// return-ready 边界，bits[31:8] 保存 FetchMax 的软件重试数（饱和）。
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
constexpr uint32_t kAtomicRetriesShift = 8;

// ClockBaseline 的 bit0 区分普通连续 SYS_CNT 与后端的 atomic 返回依赖
// 计时钩子；后者用于量化那一条依赖 MOV 自身带来的固定底噪。
constexpr uint32_t kClockAtomicDependency = 1U << 0;
constexpr uint32_t kClockAtomicDependencyApplied = 1U << 1;

struct alignas(64) TraceCoreState {
    volatile uint32_t count;
    volatile uint32_t dropped;
    uint32_t padding[14];
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
// flag 是依赖就绪与 frontier 连续前推的发布位；vend 是该 task 完成时 worker 的
// 单调 heap_next 快照。HeapGuard 读取 frontier-H 对应 vend，判断环形 heap 是否可覆盖。
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
    int32_t bucket;
    int32_t next_in_bucket;
    int32_t prev_in_bucket;
    int32_t next_in_task;
};
// 同一 entry 同时挂在两条链上：bucket 链按 buffer 地址查询重叠区间，task 链按
// producer 批量退休。next_in_bucket 在空闲状态下复用为 free-list 链接。
static_assert(sizeof(MapEntry) == 48, "MapEntry must match the PA tensor-map entry ABI");
static_assert(offsetof(MapEntry, producer) == 24, "MapEntry producer offset mismatch");
static_assert(offsetof(MapEntry, next_in_task) == 40, "MapEntry task-link offset mismatch");

struct TensorMap {
    MapEntry entries[kMapCapacity];
    int32_t buckets[kMapBuckets];
    int32_t task_heads[kTaskWindow];
    int32_t free_head;
    int32_t high_water;
    int32_t alive_floor;
    int32_t cleaned_upto;
};
// TensorMap 是 worker 私有状态，不在多核间共享。alive_floor 表达查询存活下界，
// cleaned_upto 表达已物理摘链的进度；即使 Case1 中通常同步推进，也不能合并其 ABI 字段。
static_assert(sizeof(TensorMap) == 823312, "TensorMap must match the PA fixed-capacity layout");
static_assert(offsetof(TensorMap, buckets) == 786432, "TensorMap bucket offset mismatch");
static_assert(offsetof(TensorMap, task_heads) == 819200, "TensorMap task-head offset mismatch");
static_assert(offsetof(TensorMap, free_head) == 823296, "TensorMap control offset mismatch");

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

    // 前端工作量计数不是性能填充：构参、materialize 和 map insert 用于核对全部
    // 96 个 worker 的回放；map lookup、slot copy 与 fanin 则是 winner-only 全局计数。
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

    // 最终快照用于跨 worker 比较逻辑 heap 与 TensorMap 回收状态是否完全一致。
    uint64_t final_heap_next;
    uint64_t map_high_water;
    uint64_t map_alive_floor;
    uint64_t map_cleaned_upto;
    uint64_t map_live_entries;

    uint64_t worker_id;
    uint64_t role;
    uint64_t max_occupied;
    uint64_t final_occupied;

    // CCEC 标量 PMU 取证复用 WorkerResult 原有的 24B 尾部 padding，不增加结果区大小。
    // 该诊断只在显式开启时有效；CNT2/CNT6/CNT7 分别对应 scalar busy、I-cache req/miss。
    uint64_t pmu_total_cycles;
    uint32_t pmu_scalar_busy;
    uint32_t pmu_icache_requests;
    uint32_t pmu_icache_misses;
    uint32_t pmu_status;

    // 这些计数只在 worker 私有 LocalStats 中递增，结束时一次性发布；它们把动态
    // fanin 重试和 frontier helping 展开为准确次数，不为取数再增加共享 atomic。
    uint64_t fanin_not_ready_loads;
    uint64_t frontier_initial_loads;
    uint64_t frontier_updates;
    uint64_t frontier_terminal_loads;

    // 仅在 trace_enabled bit1 开启时递增；每次源码 atomic 调用恰好增加一，
    // host 用它与 Atomic span 数逐 worker 闭合，禁止把丢记录的泳道当成完整结果。
    uint64_t atomic_trace_calls;
};
// WorkerResult 是 standalone 尾部的诊断 sidecar，不属于真实 DistCore ABI；按
// cache line 隔离后，各 worker 发布统计不会相互覆盖或污染被测共享状态。
static_assert(sizeof(WorkerResult) == 768, "WorkerResult diagnostics must occupy whole cache lines");
static_assert(offsetof(WorkerResult, pmu_total_cycles) == 680, "WorkerResult PMU offset mismatch");
static_assert(offsetof(WorkerResult, pmu_status) == 700, "WorkerResult PMU status offset mismatch");
static_assert(offsetof(WorkerResult, fanin_not_ready_loads) == 704, "WorkerResult atomic diagnostic offset mismatch");
static_assert(offsetof(WorkerResult, atomic_trace_calls) == 736, "WorkerResult atomic trace offset mismatch");

// 从 cube_cursor 到 workers 结束保留关键字段 offset、DistCore ABI 和生产总字节跨度，
// 并非字段级完整镜像。RunConfig、输入 context_lens 与校验结果追加在该跨度之后，
// 因此测试控制信息不会改变被测字段 offset。
struct alignas(64) SchedulerState {
    // 三组四分片 cursor 分别服务 AIC kernel、AIV kernel 和 Alloc；同 task 的
    // eligible workers 竞争同一 shard，只有旧值小于 task_id 的调用成为 winner。
    AtomicLine cube_cursor[kCursorShards];
    AtomicLine vector_cursor[kCursorShards];
    AtomicLine alloc_cursor[kCursorShards];
    AtomicLine frontier;
    int32_t heap_window;
    uint8_t tasks_padding[60];
    TaskCell tasks[kTaskCellCapacity];
    // heap_base/size 描述共享物理环，worker.heap_next 则是各 worker 一致推进的逻辑游标。
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
    // started_count 形成 launch 屏障；replay_done 只用于最终 drain 判定所有 worker
    // 已不再产生新 slot。两者都位于 Submit 性能口径之外，但属于完整协议。
    WorkerState workers[kRuntimeMaxWorkers];
    // Standalone-only controls live after the complete DistGlobal image. They
    // therefore do not shift any cursor/task/fatal/worker address under test.
    RunConfig config;
    // Context lengths are the only PA input elements read by orchestration;
    // keeping them in GM preserves the per-batch descriptor-based load.
    // 除这 256 个长度值外，其余 tensor 仅需稳定的合成地址来复现
    // descriptor、依赖和 heap 行为，不会解引用成真实计算数据。
    volatile int32_t context_lens[kMaxBatches];
    WorkerResult results[kWorkers];
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

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_MODEL_H
