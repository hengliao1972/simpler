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

#pragma once

#include <cstddef>
#include <cstdint>

#include "dist_engine/common/target.h"

#include "dist_engine/dist_engine.h"
#include "common/core_type.h"
#include "fdwic_build_identity.h"
#include "pto2_dispatch_payload.h"
#include "pto_constants.h"
#include "pto_submit_types.h"
#include "pto_types.h"
#include "tensor.h"

struct PTO2Runtime;
struct L2TaskArgs;
class Runtime;

constexpr int32_t kDistRuntimeMaxWorker = 108;
#ifdef RUNTIME_MAX_WORKER
static_assert(kDistRuntimeMaxWorker == RUNTIME_MAX_WORKER, "dist_engine worker ABI must match Runtime");
#endif

constexpr int32_t kPrivateSlots = 4;
constexpr int32_t kWonReserve = 2;
constexpr int32_t kMaxFanin = 16;
constexpr int32_t kMapCap = 16384;
constexpr uint32_t kMapBucketCapacity = kFdwicTensorMapRingCap;
constexpr uint32_t kMapBuckets = kFdwicTensorMapRingBuckets;
constexpr uint32_t kMapBaseControlBuckets = 128;
constexpr uint32_t kMapBucketSlotMask = kMapBucketCapacity - 1;
constexpr uint32_t kMapBucketMask = kMapBuckets - 1;
constexpr size_t kMapControlBytes = 32768;
constexpr int32_t kFlagCap = 1 << 16;
constexpr int32_t kTaskPayloadSlots = 2048;
constexpr int32_t kTaskPayloadMask = kTaskPayloadSlots - 1;
static_assert((kTaskPayloadSlots & kTaskPayloadMask) == 0, "task payload slots must be a power of two");
static_assert(kMapBucketCapacity * kMapBuckets == static_cast<uint32_t>(kMapCap));
static_assert((kMapBucketCapacity & kMapBucketSlotMask) == 0, "TensorMap bucket capacity must be a power of two");
static_assert((kMapBuckets & kMapBucketMask) == 0, "TensorMap bucket count must be a power of two");
static_assert(kMapBuckets <= 512, "TensorMap control area only reserves up to 512 buckets");

constexpr uint32_t dist_constexpr_log2(uint32_t value) {
    return value <= 1U ? 0U : 1U + dist_constexpr_log2(value >> 1U);
}

constexpr uint32_t kMapBucketShift = dist_constexpr_log2(kMapBuckets);

struct DistTaskPayload {
    Tensor tensors[MAX_TENSOR_ARGS];
};
static_assert(sizeof(DistTaskPayload) % 64 == 0, "DistTaskPayload must not share cachelines");
static_assert(offsetof(DistTaskPayload, tensors) % 64 == 0, "payload tensors must be cacheline-aligned");

struct DistOutputLayout {
    uint64_t buffer_sizes[MAX_TENSOR_ARGS];
    uint64_t total_output_size;
};

[[maybe_unused]] constexpr size_t kHeapRingDefault = 64ull << 20;
constexpr int32_t kHDefault = 64;

struct MapEntry {
    uint64_t buf_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    uint32_t payload_abi_reserved;
    // private ring 的 bucket/slot 由连续下标隐式给出，不再保存链指针。
    // 末 16B 为后续布局演进预留；shared 发布协议不会借用 private 热槽。
    uint8_t abi_reserved[16];
};
static_assert(sizeof(MapEntry) == 48, "FDWIC MapEntry ABI size changed");
static_assert(alignof(MapEntry) == 8, "FDWIC MapEntry ABI alignment changed");
static_assert(offsetof(MapEntry, producer) == 24, "FDWIC MapEntry producer offset changed");
static_assert(offsetof(MapEntry, abi_reserved) == 32, "FDWIC MapEntry reserve offset changed");

constexpr int32_t kTaskWindow = 1 << 10;
constexpr int32_t kTaskWindowMask = kTaskWindow - 1;

struct DistTensorMap {
    MapEntry entries[kMapCap];
    // 默认 CAP=128 时，前 128 个 head/tail 保持连续固定位置。CAP=32/64
    // 的额外桶游标从原 32KiB bucket 区域内部切出，所有模式的 map 总尺寸
    // 和 DistCore 后续字段偏移保持不动。
    uint64_t bucket_heads[kMapBaseControlBuckets];
    uint64_t bucket_tails[kMapBaseControlBuckets];
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
    uint64_t extra_bucket_heads[kMapBuckets - kMapBaseControlBuckets];
    uint64_t extra_bucket_tails[kMapBuckets - kMapBaseControlBuckets];
    uint8_t control_abi_reserved[kMapControlBytes - 2 * sizeof(uint64_t) * kMapBuckets];
#else
    uint8_t control_abi_reserved[
        kMapControlBytes - 2 * sizeof(uint64_t) * kMapBaseControlBuckets
    ];
#endif
    // 旧 task-head/free-list 区只保留物理 ABI，不在默认热路径维护全局 live
    // 计数。每桶容量由 tail-head 当场判断，auto CAP 由后续静态 planner 负责。
    uint8_t task_window_abi_reserved[sizeof(int32_t) * kTaskWindow];
    uint32_t tail_abi_reserved0;
    uint32_t tail_abi_reserved1;
    int32_t alive_floor;
    int32_t tail_abi_reserved2;
};
static_assert(sizeof(DistTensorMap) == 823312, "FDWIC TensorMap must preserve the DistCore ABI");
static_assert(alignof(DistTensorMap) == 8, "FDWIC TensorMap alignment changed");
static_assert(offsetof(DistTensorMap, bucket_heads) == 786432, "FDWIC TensorMap head offset changed");
static_assert(offsetof(DistTensorMap, bucket_tails) == 787456, "FDWIC TensorMap tail offset changed");
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
static_assert(
    offsetof(DistTensorMap, extra_bucket_heads) == 788480, "FDWIC TensorMap extra-head offset changed"
);
static_assert(
    offsetof(DistTensorMap, extra_bucket_tails) ==
        788480 + sizeof(uint64_t) * (kMapBuckets - kMapBaseControlBuckets),
    "FDWIC TensorMap extra-tail offset changed"
);
#endif
static_assert(
    offsetof(DistTensorMap, task_window_abi_reserved) == 819200,
    "FDWIC TensorMap task-window reserve offset changed"
);
static_assert(
    offsetof(DistTensorMap, control_abi_reserved) + sizeof(DistTensorMap::control_abi_reserved) == 819200,
    "FDWIC TensorMap control area size changed"
);
static_assert(offsetof(DistTensorMap, tail_abi_reserved0) == 823296, "FDWIC TensorMap tail offset changed");
static_assert(offsetof(DistTensorMap, tail_abi_reserved1) == 823300, "FDWIC TensorMap tail offset changed");
static_assert(offsetof(DistTensorMap, alive_floor) == 823304, "FDWIC TensorMap alive-floor offset changed");
static_assert(offsetof(DistTensorMap, tail_abi_reserved2) == 823308, "FDWIC TensorMap tail offset changed");

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
    ClockBaseline = 15,
    OrchestrationReplay = 16,
    FinalDrain = 17,
    WinnerBuild = 18,
    AllocComplete = 19,
    LoserReplay = 20,
};

struct RingSlot {
    bool occupied;
    bool built;
    int32_t task_id;
    int32_t func_id;
    uint64_t function_bin_addr;

    int32_t tensor_count;
    int32_t scalar_count;
    uint8_t tensors_pad[32];
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];

    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    LocalContext local_ctx;
    GlobalContext global_ctx;

    int32_t fanin[kMaxFanin];
    int32_t fanin_count;

    bool is_multicore;
    int32_t won_block;
    int32_t won_slot;
};

struct BuiltSubtask {
    bool present;
    int32_t func_id;
    uint64_t function_bin_addr;
    int32_t tensor_count;
    int32_t scalar_count;
    uint8_t tensors_pad[40];
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t sub_block_id;
};

static_assert(offsetof(RingSlot, tensors) % 64 == 0, "RingSlot tensors must be cacheline-aligned");
static_assert(offsetof(BuiltSubtask, tensors) % 64 == 0, "BuiltSubtask tensors must be cacheline-aligned");

struct DrainedCell {
    volatile int64_t v;
    uint8_t pad[64 - sizeof(int64_t)];
};
static_assert(sizeof(DrainedCell) == 64, "DrainedCell must occupy one cacheline");

struct WonAtomicCell {
    volatile int64_t v;
    uint8_t pad[64 - sizeof(int64_t)];
};
static_assert(sizeof(WonAtomicCell) == 64, "WonSlot atomic cell must occupy one cacheline");

struct WonMetaLine {
    int32_t task_id;
    uint8_t pad[64 - sizeof(int32_t)];
};
static_assert(sizeof(WonMetaLine) == 64, "WonSlot metadata must occupy one cacheline");

constexpr int64_t kDrainedFree = 0;
constexpr int64_t kDrainedClaimed = 1;

constexpr int64_t kWonStateFree = 0;
constexpr int64_t kWonStateClaimed = 1;
constexpr int64_t kWonStatePublished = 2;

struct WonSlot {
    WonAtomicCell state;
    WonMetaLine meta;
    WonAtomicCell remaining;
    DrainedCell drained[PTO2_SUBTASK_SLOT_COUNT];
    BuiltSubtask lane[PTO2_SUBTASK_SLOT_COUNT];
    uint8_t tail_pad[40];
};
static_assert(offsetof(WonSlot, state) % 64 == 0, "WonSlot state must be cacheline-aligned");
static_assert(offsetof(WonSlot, meta) % 64 == 0, "WonSlot metadata must be cacheline-aligned");
static_assert(offsetof(WonSlot, remaining) % 64 == 0, "WonSlot remaining must be cacheline-aligned");
static_assert(offsetof(WonSlot, drained) % 64 == 0, "WonSlot drained cells must be cacheline-aligned");
static_assert(offsetof(WonSlot, lane) % 64 == 0, "WonSlot lanes must be cacheline-aligned");
static_assert(sizeof(WonSlot) % 64 == 0, "WonSlot must not share cachelines");

struct BlockWon {
    WonSlot slots[kPrivateSlots];
    uint8_t any_pub_pad[64 - ((sizeof(WonSlot) * kPrivateSlots) % 64)];
    volatile int32_t any_pub;
    uint8_t any_pub_tail_pad[64 - sizeof(int32_t)];
};
static_assert(offsetof(BlockWon, slots) % 64 == 0, "BlockWon slots must be cacheline-aligned");
static_assert(offsetof(BlockWon, any_pub) % 64 == 0, "BlockWon any_pub must be cacheline-aligned");
static_assert(sizeof(BlockWon) % 64 == 0, "BlockWon must not share cachelines");

enum LaneId : int32_t { LANE_AIC = 0, LANE_AIV0 = 1, LANE_AIV1 = 2, LANE_NONE = -1 };

struct CoreLayout {
    int32_t block_id;
    int32_t lane;
};

struct DistCore {
    CoreType role;
    int32_t core_idx;
    int32_t block_id;
    int32_t lane;
    int32_t sub_block_id;
    int32_t local_index;
    uint64_t heap_next;

    DistTensorMap map;

    uint8_t slots_pad[16];
    RingSlot slots[kPrivateSlots];
    int32_t occupied_count;
    int32_t owned_total;
    uint64_t swimlane_last_cycle;

    uint8_t task_payloads_pad[16];
    DistTaskPayload task_payloads[kTaskPayloadSlots];
};
static_assert(offsetof(DistCore, map) == 32, "FDWIC DistCore TensorMap offset changed");
static_assert(offsetof(DistCore, slots_pad) == 823344, "FDWIC DistCore slot padding offset changed");
static_assert(offsetof(DistCore, slots) == 823360, "FDWIC DistCore ring-slot offset changed");
static_assert(offsetof(DistCore, occupied_count) == 842656, "FDWIC DistCore occupancy offset changed");
static_assert(offsetof(DistCore, owned_total) == 842660, "FDWIC DistCore owned-total offset changed");
static_assert(
    offsetof(DistCore, swimlane_last_cycle) == 842664, "FDWIC DistCore swimlane-clock offset changed"
);
static_assert(
    offsetof(DistCore, task_payloads_pad) == 842672, "FDWIC DistCore payload padding offset changed"
);
static_assert(offsetof(DistCore, task_payloads) == 842688, "FDWIC DistCore task-payload offset changed");
static_assert(sizeof(DistCore) == 9231296, "FDWIC DistCore ABI size changed");
static_assert(offsetof(DistCore, slots) % 64 == 0, "DistCore slots must be cacheline-aligned");
static_assert(offsetof(DistCore, task_payloads) % 64 == 0, "DistCore task_payloads must be cacheline-aligned");

constexpr int32_t kCursorShards = 4;
constexpr int32_t kFinalBarrierGroups = 16;
constexpr size_t kCacheLine = 64;
static_assert(PTO2_PACKED_OUTPUT_ALIGN >= kCacheLine);
static_assert((PTO2_PACKED_OUTPUT_ALIGN % kCacheLine) == 0);

struct PaddedCursor {
    volatile int64_t v;
    uint8_t pad[kCacheLine - sizeof(int64_t)];
};
static_assert(sizeof(PaddedCursor) == kCacheLine, "PaddedCursor must occupy one cacheline");

// shared TensorMap 与 private 共用 bucket/CAP/hash/逻辑 region 语义，但不
// 借用 private MapEntry 的 16B ABI reserve。专属 32B value 的 reserved
// 必须由 writer 写 0、reader 校验 0，协议字段边界与 standalone 一致。
struct SharedTensorMapValue {
    uint64_t buf_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    uint32_t reserved;
};
static_assert(sizeof(SharedTensorMapValue) == 32, "shared TensorMap logical value size changed");
static_assert(offsetof(SharedTensorMapValue, buf_addr) == 0, "shared TensorMap buffer offset changed");
static_assert(offsetof(SharedTensorMapValue, lo) == 8, "shared TensorMap lower-bound offset changed");
static_assert(offsetof(SharedTensorMapValue, hi) == 16, "shared TensorMap upper-bound offset changed");
static_assert(offsetof(SharedTensorMapValue, producer) == 24, "shared TensorMap producer offset changed");
static_assert(offsetof(SharedTensorMapValue, reserved) == 28, "shared TensorMap reserve offset changed");

// payload 与发布 seq 必须分处独占 cache line。A5 的 atomic seq 访问和
// 普通 payload cache writeback 若落在同一行，可能互相覆盖；两行分离后，
// writer 可按“payload flush -> seq publish”建立明确的跨核可见性边界。
struct alignas(kCacheLine) SharedTensorMapPayloadLine {
    SharedTensorMapValue value;
    uint8_t pad[kCacheLine - sizeof(SharedTensorMapValue)];
};
static_assert(sizeof(SharedTensorMapPayloadLine) == kCacheLine, "shared TensorMap payload must occupy one cacheline");
static_assert(offsetof(SharedTensorMapPayloadLine, value) == 0, "shared TensorMap payload value offset changed");

struct alignas(kCacheLine) SharedTensorMapSequenceLine {
    volatile int64_t v;
    uint8_t pad[kCacheLine - sizeof(int64_t)];
};
static_assert(offsetof(SharedTensorMapSequenceLine, v) == 0, "shared TensorMap sequence value offset changed");
static_assert(sizeof(SharedTensorMapSequenceLine) == kCacheLine, "shared TensorMap sequence must occupy one cacheline");

struct alignas(kCacheLine) SharedTensorMapSlot {
    SharedTensorMapPayloadLine payload;
    SharedTensorMapSequenceLine sequence;
};
static_assert(sizeof(SharedTensorMapSlot) == 2 * kCacheLine, "shared TensorMap slot size changed");
static_assert(
    offsetof(SharedTensorMapSlot, sequence) == kCacheLine,
    "shared TensorMap sequence must not share the payload cacheline"
);

constexpr int64_t kSharedTensorMapInvalidSequence = -1;
constexpr int64_t kSharedTensorMapInitialCommit = 0;
constexpr int64_t kSharedTensorMapInitialReclaim = -1;

struct alignas(kCacheLine) SharedTensorMapBucketState {
    PaddedCursor head;
    PaddedCursor tail;
};
static_assert(sizeof(SharedTensorMapBucketState) == 2 * kCacheLine, "shared TensorMap bucket controls changed");
static_assert(
    offsetof(SharedTensorMapBucketState, tail) == kCacheLine,
    "shared TensorMap head and tail must not share a cacheline"
);

struct alignas(kCacheLine) SharedTensorMapState {
    // committed_tasks 是下一个允许发布的 task id；即使任务没有 region，
    // ordered commit 也必须从 N 推进到 N+1。
    PaddedCursor committed_tasks;
    // 已可回收的最大 producer id，初值 -1。只有 exact-turn winner 会
    // 访问 map，因此在完成 task N lookup 后可直接用 N-H-1 单调推进；
    // loser 不读 map，也不需要 per-core progress。
    PaddedCursor reclaim_upto;
    // 每桶 head/tail 各占一行且彼此相邻。第一版采用 task-id 有序单
    // 追加者，因而不需要 MPSC reserve 游标或全局 free-list。
    SharedTensorMapBucketState buckets[kMapBuckets];
    // 与 private 完全相同的连续分桶下标：
    // bucket * CAP + (absolute_cursor & (CAP - 1))。
    SharedTensorMapSlot slots[kMapCap];
};
static_assert(offsetof(SharedTensorMapState, committed_tasks) == 0);
static_assert(offsetof(SharedTensorMapState, reclaim_upto) == kCacheLine);
static_assert(offsetof(SharedTensorMapState, buckets) == 2 * kCacheLine);
static_assert(
    offsetof(SharedTensorMapState, slots) == 2 * kCacheLine + sizeof(SharedTensorMapBucketState) * kMapBuckets,
    "shared TensorMap slots must immediately follow bucket controls"
);
static_assert(sizeof(SharedTensorMapState) % kCacheLine == 0);
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(offsetof(SharedTensorMapState, buckets) == 128);
static_assert(offsetof(SharedTensorMapState, slots) == 16512);
static_assert(sizeof(SharedTensorMapState) == 2113664);
#endif

struct DistTaskCell {
    volatile int64_t flag;
    volatile uint64_t vend;
    uint8_t pad[kCacheLine - sizeof(int64_t) - sizeof(uint64_t)];
};
static_assert(sizeof(DistTaskCell) == kCacheLine);

struct alignas(kCacheLine) FinalBarrierArrival {
    volatile int64_t v;
    volatile int32_t expected;
    uint8_t pad[kCacheLine - sizeof(int64_t) - sizeof(int32_t)];
};
static_assert(sizeof(FinalBarrierArrival) == kCacheLine, "final barrier arrival must occupy one cacheline");

struct alignas(kCacheLine) FinalBarrierRelease {
    volatile int64_t v;
    uint8_t pad[kCacheLine - sizeof(int64_t)];
};
static_assert(sizeof(FinalBarrierRelease) == kCacheLine, "final barrier release must occupy one cacheline");

struct alignas(kCacheLine) FinalBarrierState {
    FinalBarrierArrival leaf_arrivals[kFinalBarrierGroups];
    FinalBarrierRelease leaf_releases[kFinalBarrierGroups];
    FinalBarrierArrival root_arrival;
    FinalBarrierRelease root_release;
};
static_assert(sizeof(FinalBarrierState) == 34 * kCacheLine, "final barrier state size changed");

struct DistGlobal {
    PaddedCursor cube_cursor[kCursorShards];
    PaddedCursor vector_cursor[kCursorShards];
    PaddedCursor alloc_cursor[kCursorShards];

    volatile int64_t frontier;
    uint8_t frontier_pad[kCacheLine - sizeof(int64_t)];
    int32_t H;
    uint8_t tasks_pad[kCacheLine - sizeof(int32_t)];
    DistTaskCell tasks[kFlagCap];

    uint8_t *heap_base;
    size_t heap_size;

    const L2TaskArgs *orch_args;
    PTO2Runtime *rt;
    Runtime *runtime;

    uint8_t fatal_pad[24];
    volatile int32_t fatal;
    // 首个非零运行时错误码获胜；fatal 仍保留原 offset，后续字段也不移动。
    // AICPU 在所有 worker 完成后失效并读取整条 cache line。
    volatile int32_t error_code;
    uint8_t fatal_tail_pad[kCacheLine - 2 * sizeof(int32_t)];

    int32_t num_workers;
    int32_t num_blocks;
    CoreLayout layout[kDistRuntimeMaxWorker];
    uint8_t blocks_pad[24];
    BlockWon blocks[kDistRuntimeMaxWorker];

    // Retained in place for the existing DistGlobal hot-field ABI. Final
    // completion now uses final_barrier after cores instead of this flat line.
    volatile int64_t replay_done;
    uint8_t replay_done_pad[kCacheLine - sizeof(int64_t)];

    volatile int64_t started_count;
    uint8_t started_count_pad[kCacheLine - sizeof(int64_t)];

    DistCore cores[kDistRuntimeMaxWorker];

    // Keep all existing hot-field and DistCore offsets stable. Only the tail
    // grows for the fixed two-level G=16 final barrier.
    FinalBarrierState final_barrier;
#if PTO_FDWIC_SHARED_MAP
    // shared 专属 sidecar 只追加在旧 DistGlobal 尾部；private artifact 不
    // 实例化这 2MiB 状态，且所有旧热字段与 per-core map offset 均不移动。
    SharedTensorMapState shared_tensor_map;
#endif
};
static_assert(offsetof(DistGlobal, frontier) % 64 == 0, "DistGlobal frontier must be cacheline-aligned");
static_assert(offsetof(DistGlobal, tasks) % 64 == 0, "DistGlobal tasks must be cacheline-aligned");
static_assert(offsetof(DistGlobal, fatal) % 64 == 0, "DistGlobal fatal must be cacheline-aligned");
static_assert(
    offsetof(DistGlobal, error_code) == offsetof(DistGlobal, fatal) + sizeof(int32_t),
    "DistGlobal runtime error must share the fatal cacheline"
);
static_assert(offsetof(DistGlobal, blocks) % 64 == 0, "DistGlobal blocks must be cacheline-aligned");
static_assert(offsetof(DistGlobal, replay_done) % 64 == 0, "DistGlobal replay_done must be cacheline-aligned");
static_assert(offsetof(DistGlobal, started_count) % 64 == 0, "DistGlobal started_count must be cacheline-aligned");
static_assert(offsetof(DistGlobal, cores) % 64 == 0, "DistGlobal cores must be cacheline-aligned");
static_assert(
    offsetof(DistGlobal, final_barrier) % 64 == 0, "DistGlobal final barrier must be cacheline-aligned"
);

// 68f51451 已冻结的 private DistGlobal 尾边界。shared sidecar 只能从该
// offset 追加，不能把 mode-specific 字段插进旧热布局。
constexpr size_t kFdwicSharedTensorMapOffset = 1007026048;
static_assert(
    offsetof(DistGlobal, final_barrier) + sizeof(FinalBarrierState) == kFdwicSharedTensorMapOffset,
    "FDWIC legacy DistGlobal tail moved"
);
#if PTO_FDWIC_SHARED_MAP
static_assert(
    offsetof(DistGlobal, shared_tensor_map) == kFdwicSharedTensorMapOffset,
    "shared TensorMap sidecar must append after the frozen DistGlobal tail"
);
static_assert(
    sizeof(DistGlobal) == kFdwicSharedTensorMapOffset + sizeof(SharedTensorMapState),
    "shared DistGlobal may only grow by its TensorMap sidecar"
);
#else
static_assert(
    sizeof(DistGlobal) == kFdwicSharedTensorMapOffset, "private DistGlobal size changed while adding shared sidecar"
);
#endif
static_assert(sizeof(DistGlobal) <= kDistEngineGlobalStateSize, "DistGlobal exceeds the reserved runtime arena size");
static_assert(
    alignof(DistGlobal) <= kDistEngineGlobalStateAlign, "DistGlobal exceeds the reserved runtime arena align"
);
