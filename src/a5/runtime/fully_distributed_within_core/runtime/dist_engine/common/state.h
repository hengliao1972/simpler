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

#include "dist_engine/common/cross_core_exec_protocol.h"
#include "dist_engine/common/cross_core_output_protocol.h"
#include "dist_engine/common/shared_pa_atomic_layout.h"
#include "dist_engine/common/target.h"
#include "dist_engine/common/swimlane_types.h"

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
// DistCore keeps its historical private-map bytes in both artifacts so the
// AICPU lifecycle and following core offsets stay frozen. Shared PA never
// reads this storage; these two constants describe only that physical ABI and
// therefore must not use the shared build-identity fields (which are zero).
constexpr uint32_t kMapBucketCapacity = static_cast<uint32_t>(PTO_FDWIC_TENSORMAP_RING_CAP);
constexpr uint32_t kMapBuckets = static_cast<uint32_t>(kMapCap) / kMapBucketCapacity;
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
    // The private ring derives bucket/slot from the contiguous index and no
    // longer stores linked-list pointers. The last 16 bytes remain reserved;
    // shared publication protocols must not borrow this private hot slot.
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
    // With the default CAP=128, the first 128 head/tail cursors stay at their
    // fixed contiguous offsets. CAP=32/64 takes the additional cursors from
    // the original 32-KiB control area so every mode preserves the total map
    // size and all following DistCore offsets.
    uint64_t bucket_heads[kMapBaseControlBuckets];
    uint64_t bucket_tails[kMapBaseControlBuckets];
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
    uint64_t extra_bucket_heads[kMapBuckets - kMapBaseControlBuckets];
    uint64_t extra_bucket_tails[kMapBuckets - kMapBaseControlBuckets];
    uint8_t control_abi_reserved[kMapControlBytes - 2 * sizeof(uint64_t) * kMapBuckets];
#else
    uint8_t control_abi_reserved[kMapControlBytes - 2 * sizeof(uint64_t) * kMapBaseControlBuckets];
#endif
    // The former task-head/free-list area remains only as physical ABI space.
    // The default hot path checks each bucket with tail-head instead of
    // maintaining a global live count; a later static planner owns auto CAP.
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
static_assert(offsetof(DistTensorMap, extra_bucket_heads) == 788480, "FDWIC TensorMap extra-head offset changed");
static_assert(
    offsetof(DistTensorMap, extra_bucket_tails) == 788480 + sizeof(uint64_t) * (kMapBuckets - kMapBaseControlBuckets),
    "FDWIC TensorMap extra-tail offset changed"
);
#endif
static_assert(
    offsetof(DistTensorMap, task_window_abi_reserved) == 819200, "FDWIC TensorMap task-window reserve offset changed"
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
#if PTO_FDWIC_SHARED_MAP
    SharedRegisterPublishMetadata = 20,
    SharedMaterializePublishTaskOutputs = 21,
    SharedMaterializePublishTaskOutputsCopy = 22,
    SharedMaterializePublishTaskOutputsFlush = 23,
    Dcci = 24,
    SharedRegisterWaitInsertTurnBypassLoad = 25,
    Count = 26,
    // Kept only so stale private code remains parsable during the compile-time
    // backend split. The selected raw ABI rejects this value in shared mode.
    LoserReplay = 0x7fff,
#else
    LoserReplay = 20,
#endif
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
static_assert(offsetof(DistCore, swimlane_last_cycle) == 842664, "FDWIC DistCore swimlane-clock offset changed");
static_assert(offsetof(DistCore, task_payloads_pad) == 842672, "FDWIC DistCore payload padding offset changed");
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

// Phase-1 shared PA is deliberately narrower than the private region map:
// exactly one Alloc+QK+SF+PV+UP group per batch and at most 256 batches.
// There is no task-id generation or wrap in this contract.
constexpr uint32_t kFdwicSharedPaBatches = 256;
constexpr uint32_t kFdwicSharedPaTasksPerBatch = 5;
constexpr uint32_t kFdwicSharedPaTaskCapacity = kFdwicSharedPaBatches * kFdwicSharedPaTasksPerBatch;
constexpr uint32_t kFdwicSharedOutputMaxPerTask = 8;
constexpr uint32_t kFdwicSharedHeapShards = 8;
constexpr uint32_t kFdwicSharedVectorCursorShards = 8;
constexpr uint64_t kFdwicSharedHeapBytes = 256ULL << 20;
constexpr uint64_t kFdwicSharedHeapShardBytes = kFdwicSharedHeapBytes / kFdwicSharedHeapShards;
constexpr uint32_t kFdwicSharedWriterHistoryMagic = 0x57484953U;  // "WHIS"

static_assert(kFdwicSharedPaTaskCapacity == 1280, "phase-1 shared PA task capacity changed");
static_assert(kFdwicSharedWorkers == 96, "phase-1 shared PA worker topology changed");
static_assert(
    kFdwicSharedAllocClaimTournamentGroups <= kFdwicSharedWorkers &&
        kFdwicSharedAicClaimTournamentGroups <= kFdwicSharedAicWorkers &&
        kFdwicSharedAivClaimTournamentGroups <= kFdwicSharedAivWorkers,
    "every shared Claim tournament group must have a candidate"
);
static_assert(kFdwicSharedHeapShardBytes == (32ULL << 20), "shared PA heap shard size changed");
static_assert(
    (kFdwicSharedHeapShards & (kFdwicSharedHeapShards - 1U)) == 0, "shared PA heap shards must be a power of two"
);
static_assert(
    (kFdwicSharedVectorCursorShards & (kFdwicSharedVectorCursorShards - 1U)) == 0,
    "shared PA Vector cursor shards must be a power of two"
);

// Fresh outputs are addressed directly by (producer_task_id, output_slot).
// Publication flags, writer caches, and immutable Tensor descriptors occupy
// disjoint cache-line regions.
struct alignas(kCacheLine) SharedOutputCell {
    PaddedCursor published[kFdwicSharedOutputMaxPerTask];
    PaddedCursor last_writer[kFdwicSharedOutputMaxPerTask];
    Tensor tensors[kFdwicSharedOutputMaxPerTask];
};
static_assert(sizeof(SharedOutputCell) == 2048, "shared output cell size changed");
static_assert(alignof(SharedOutputCell) == kCacheLine, "shared output cell alignment changed");
static_assert(offsetof(SharedOutputCell, published) == 0, "shared output publish offset changed");
static_assert(offsetof(SharedOutputCell, last_writer) == 512, "shared output writer offset changed");
static_assert(offsetof(SharedOutputCell, tensors) == 1024, "shared output descriptor offset changed");

struct SharedWriterHistoryRecord {
    uint32_t symbol_key;
    int32_t previous_writer;
};
static_assert(sizeof(SharedWriterHistoryRecord) == 8, "shared writer-history record size changed");

// One immutable payload belongs to one writer task. PA-UP writes three
// records; the physical capacity remains MAX_TENSOR_ARGS for strict checking.
struct alignas(kCacheLine) SharedWriterHistoryCell {
    uint32_t magic;
    int32_t writer_task;
    uint32_t count;
    uint32_t reserved;
    SharedWriterHistoryRecord entries[MAX_TENSOR_ARGS];
    uint8_t padding[48];
};
static_assert(sizeof(SharedWriterHistoryCell) == 320, "shared writer-history cell size changed");
static_assert(alignof(SharedWriterHistoryCell) == kCacheLine, "shared writer-history cell alignment changed");
static_assert(
    offsetof(SharedWriterHistoryCell, entries) == 16,
    "shared writer-history records must immediately follow their header"
);

// Every task owns fresh, never-reused Claim nodes. All legal candidates first
// contend on one of G local nodes; only each local winner reaches the root.
// The 512B stride is the A5 probe-selected address spacing and keeps the hot
// atomic word isolated from descriptor/history cache lines.
struct alignas(kCacheLine) SharedClaimTournamentNode {
    PaddedCursor owner;
    uint8_t padding[kFdwicSharedClaimTournamentNodeStride - sizeof(PaddedCursor)];
};
static_assert(
    sizeof(SharedClaimTournamentNode) == kFdwicSharedClaimTournamentNodeStride,
    "shared Claim tournament node stride changed"
);
static_assert(offsetof(SharedClaimTournamentNode, owner) == 0);

struct alignas(kCacheLine) SharedClaimTournamentTask {
    SharedClaimTournamentNode root;
    SharedClaimTournamentNode local[kFdwicSharedClaimTournamentMaxGroups];
};
static_assert(
    sizeof(SharedClaimTournamentTask) ==
        kFdwicSharedClaimTournamentNodeStride * (1U + kFdwicSharedClaimTournamentMaxGroups),
    "shared per-task Claim tournament layout changed"
);
static_assert(
    offsetof(SharedClaimTournamentTask, local) == kFdwicSharedClaimTournamentNodeStride,
    "shared Claim local nodes must follow the root"
);

// The old shared region ring is intentionally absent. Phase 1 keeps only
// stable output symbols, no-wrap heap controls, immutable writer history, and
// the per-task two-level Claim tournament.
struct alignas(kCacheLine) SharedPaTensorMapState {
    SharedOutputCell shared_outputs[kFdwicSharedPaTaskCapacity];
    // Independent heap shards are active concurrently. Give each hot cursor
    // one measured A5 128-byte conflict unit instead of merely one cache line.
    FdwicSharedAtomicConflictCell shared_heap_cursor[kFdwicSharedHeapShards];
    PaddedCursor shared_heap_vend;
    PaddedCursor shared_vector_cursor[kFdwicSharedVectorCursorShards];
    SharedWriterHistoryCell writer_history[kFdwicSharedPaTaskCapacity];
    SharedClaimTournamentTask claim_tournament[kFdwicSharedPaTaskCapacity];
    // TensorMap insertion remains strictly ordered, but adjacent task
    // completions must not serialize in the same 128-byte atomic unit. This
    // shared-only tail keeps the production TaskCell prefix unchanged.
    uint8_t insert_completion_alignment_pad[64];
    FdwicSharedAtomicConflictTable<kFdwicSharedPaTaskCapacity> insert_completion;
};
static_assert(offsetof(SharedPaTensorMapState, shared_outputs) == 0);
static_assert(offsetof(SharedPaTensorMapState, shared_heap_cursor) == 2621440);
static_assert(offsetof(SharedPaTensorMapState, shared_heap_cursor) % kFdwicSharedAtomicConflictBytes == 0);
static_assert(offsetof(SharedPaTensorMapState, shared_heap_vend) == 2622464);
static_assert(offsetof(SharedPaTensorMapState, shared_vector_cursor) == 2622528);
static_assert(offsetof(SharedPaTensorMapState, writer_history) == 2623040);
static_assert(offsetof(SharedPaTensorMapState, claim_tournament) == 3032640);
static_assert(offsetof(SharedPaTensorMapState, insert_completion_alignment_pad) == 8930880);
static_assert(offsetof(SharedPaTensorMapState, insert_completion) == 8930944);
static_assert(offsetof(SharedPaTensorMapState, insert_completion) % kFdwicSharedAtomicConflictBytes == 0);
static_assert(sizeof(SharedPaTensorMapState) == 9094784, "shared PA TensorMap sidecar size changed");
static_assert(alignof(SharedPaTensorMapState) == kCacheLine, "shared PA TensorMap alignment changed");

#if PTO_FDWIC_SCHEDULER_MODE == 1
// The first production cross-core backend keeps one immutable execution cell
// per logical task for the entire invocation. It deliberately does not reuse
// cells yet, so no generation, ABA or payload-reclamation rule is implicit in
// the initial correctness contract.
constexpr uint32_t kFdwicCrossCoreOrdinaryTaskCapacity = 2048;

struct alignas(kCacheLine) CrossCoreOrdinaryState {
    fdwic::cross_core::SharedExecControl fatal;
    fdwic::cross_core::SharedExecControl heap_cursor;
    fdwic::cross_core::SharedExecCell tasks[kFdwicCrossCoreOrdinaryTaskCapacity];
    fdwic::cross_core::CrossCoreOutputCell<Tensor> outputs[kFdwicCrossCoreOrdinaryTaskCapacity];
};
static_assert(offsetof(CrossCoreOrdinaryState, fatal) == 0);
static_assert(offsetof(CrossCoreOrdinaryState, heap_cursor) == kCacheLine);
static_assert(offsetof(CrossCoreOrdinaryState, tasks) == 2 * kCacheLine);
static_assert(
    offsetof(CrossCoreOrdinaryState, outputs) ==
    2 * kCacheLine + kFdwicCrossCoreOrdinaryTaskCapacity * sizeof(fdwic::cross_core::SharedExecCell)
);
static_assert(alignof(CrossCoreOrdinaryState) == kCacheLine);
static_assert(
    sizeof(CrossCoreOrdinaryState) ==
        2 * kCacheLine + kFdwicCrossCoreOrdinaryTaskCapacity * (sizeof(fdwic::cross_core::SharedExecCell) +
                                                                sizeof(fdwic::cross_core::CrossCoreOutputCell<Tensor>)),
    "cross-core ordinary state size changed"
);
#endif

struct DistTaskCell {
    volatile int64_t flag;
    volatile uint64_t vend;
#if PTO_FDWIC_SHARED_MAP
    // Winner N publishes N only after descriptors and writer metadata are
    // visible. Winner N+1 waits on this exact cell; losers never hand off.
    volatile int64_t deps_prepared;
    uint8_t pad[kCacheLine - 3 * sizeof(int64_t)];
#else
    uint8_t pad[kCacheLine - sizeof(int64_t) - sizeof(uint64_t)];
#endif
};
static_assert(sizeof(DistTaskCell) == kCacheLine);
#if PTO_FDWIC_SHARED_MAP
static_assert(offsetof(DistTaskCell, deps_prepared) == 16, "shared dependency handoff offset changed");
#endif

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
    // The first nonzero runtime error wins. fatal keeps its original offset
    // and no later field moves. AICPU invalidates and reads the complete cache
    // line after every worker has finished.
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
    // The shared-only state starts at the frozen private tail. Existing
    // AICPU/native fields and per-core offsets remain unchanged.
    SharedPaTensorMapState shared_pa;
#if PTO_FDWIC_SCHEDULER_MODE == 1
    CrossCoreOrdinaryState cross_core_ordinary;
#endif
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
static_assert(offsetof(DistGlobal, final_barrier) % 64 == 0, "DistGlobal final barrier must be cacheline-aligned");

// 68f51451 froze the private DistGlobal tail. Mode-specific state may append
// here but must not move the AICPU/native prefix.
constexpr size_t kFdwicSharedTensorMapOffset = 1007026048;
static_assert(
    offsetof(DistGlobal, final_barrier) + sizeof(FinalBarrierState) == kFdwicSharedTensorMapOffset,
    "FDWIC legacy DistGlobal tail moved"
);
#if PTO_FDWIC_SHARED_MAP
static_assert(
    offsetof(DistGlobal, shared_pa) == kFdwicSharedTensorMapOffset,
    "shared PA TensorMap sidecar must append after the frozen DistGlobal tail"
);
#if PTO_FDWIC_SCHEDULER_MODE == 1
static_assert(
    offsetof(DistGlobal, cross_core_ordinary) == kFdwicSharedTensorMapOffset + sizeof(SharedPaTensorMapState),
    "cross-core ordinary state must append after the existing shared TensorMap state"
);
static_assert(
    sizeof(DistGlobal) == kFdwicSharedTensorMapOffset + sizeof(SharedPaTensorMapState) + sizeof(CrossCoreOrdinaryState),
    "cross-core ordinary DistGlobal size changed"
);
#else
static_assert(
    sizeof(DistGlobal) == 1016120832,
    "shared DistGlobal must contain exactly the phase-1 PA TensorMap and Claim sidecar"
);
#endif
#else
static_assert(
    sizeof(DistGlobal) == kFdwicSharedTensorMapOffset, "private DistGlobal size changed while adding shared sidecar"
);
#endif
static_assert(sizeof(DistGlobal) <= kDistEngineGlobalStateSize, "DistGlobal exceeds the reserved runtime arena size");
static_assert(
    alignof(DistGlobal) <= kDistEngineGlobalStateAlign, "DistGlobal exceeds the reserved runtime arena align"
);
