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
constexpr int32_t kFlagCap = 1 << 16;
constexpr int32_t kTaskPayloadSlots = 2048;
constexpr int32_t kTaskPayloadMask = kTaskPayloadSlots - 1;
static_assert((kTaskPayloadSlots & kTaskPayloadMask) == 0, "task payload slots must be a power of two");

struct DistTaskPayload {
    int32_t tensor_count;
    int32_t scalar_count;
    TensorArgType tags[MAX_TENSOR_ARGS];
    uint8_t tensors_pad_[56];
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
};
static_assert(sizeof(DistTaskPayload) % 64 == 0, "DistTaskPayload must not share cachelines");
static_assert(offsetof(DistTaskPayload, tensors) % 64 == 0, "payload tensors must be cacheline-aligned");
static_assert(offsetof(DistTaskPayload, scalars) % 64 == 0, "payload scalars must be cacheline-aligned");

struct DistOutputLayout {
    uint64_t offsets[MAX_TENSOR_ARGS] = {};
    uint64_t buffer_sizes[MAX_TENSOR_ARGS] = {};
    uint64_t total_output_size = 0;
};

[[maybe_unused]] constexpr size_t kHeapRingDefault = 64ull << 20;
constexpr int32_t kHDefault = 64;

struct MapEntry {
    uint64_t buf_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    int32_t bucket;
    int32_t next_in_bucket;
    int32_t prev_in_bucket;
    int32_t next_in_task;
};

constexpr int32_t kMapBuckets = 1 << 13;
constexpr int32_t kMapBucketShift = 13;
constexpr int32_t kTaskWindow = 1 << 10;
constexpr int32_t kTaskWindowMask = kTaskWindow - 1;

struct DistTensorMap {
    MapEntry entries[kMapCap];
    int32_t buckets[kMapBuckets];
    int32_t task_heads[kTaskWindow];
    int32_t free_head;
    int32_t high_water;
    int32_t alive_floor;
    int32_t cleaned_upto;
};

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

constexpr int64_t kDrainedFree = 0;
constexpr int64_t kDrainedClaimed = 1;

constexpr int64_t kWonStateFree = 0;
constexpr int64_t kWonStateClaimed = 1;
constexpr int64_t kWonStatePublished = 2;

struct WonSlot {
    volatile int64_t state;
    int32_t task_id;
    uint8_t task_id_pad[4];
    volatile int64_t remaining;
    DrainedCell drained[PTO2_SUBTASK_SLOT_COUNT];
    BuiltSubtask lane[PTO2_SUBTASK_SLOT_COUNT];
};

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
static_assert(offsetof(DistCore, slots) % 64 == 0, "DistCore slots must be cacheline-aligned");
static_assert(offsetof(DistCore, task_payloads) % 64 == 0, "DistCore task_payloads must be cacheline-aligned");

constexpr int32_t kCursorShards = 4;
constexpr size_t kCacheLine = 64;
static_assert(PTO2_PACKED_OUTPUT_ALIGN >= kCacheLine);
static_assert((PTO2_PACKED_OUTPUT_ALIGN % kCacheLine) == 0);

struct PaddedCursor {
    volatile int64_t v;
    uint8_t pad[kCacheLine - sizeof(int64_t)];
};
static_assert(sizeof(PaddedCursor) == kCacheLine, "PaddedCursor must occupy one cacheline");

struct DistTaskCell {
    volatile int64_t flag;
    volatile uint64_t vend;
    uint8_t pad[kCacheLine - sizeof(int64_t) - sizeof(uint64_t)];
};
static_assert(sizeof(DistTaskCell) == kCacheLine);

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
    uint8_t fatal_tail_pad[kCacheLine - sizeof(int32_t)];

    int32_t num_workers;
    int32_t num_blocks;
    CoreLayout layout[kDistRuntimeMaxWorker];
    uint8_t blocks_pad[24];
    BlockWon blocks[kDistRuntimeMaxWorker];

    volatile int64_t replay_done;
    uint8_t replay_done_pad[kCacheLine - sizeof(int64_t)];

    volatile int64_t started_count;
    uint8_t started_count_pad[kCacheLine - sizeof(int64_t)];

    DistCore cores[kDistRuntimeMaxWorker];
};
static_assert(offsetof(DistGlobal, frontier) % 64 == 0, "DistGlobal frontier must be cacheline-aligned");
static_assert(offsetof(DistGlobal, tasks) % 64 == 0, "DistGlobal tasks must be cacheline-aligned");
static_assert(offsetof(DistGlobal, fatal) % 64 == 0, "DistGlobal fatal must be cacheline-aligned");
static_assert(offsetof(DistGlobal, blocks) % 64 == 0, "DistGlobal blocks must be cacheline-aligned");
static_assert(offsetof(DistGlobal, replay_done) % 64 == 0, "DistGlobal replay_done must be cacheline-aligned");
static_assert(offsetof(DistGlobal, started_count) % 64 == 0, "DistGlobal started_count must be cacheline-aligned");
static_assert(offsetof(DistGlobal, cores) % 64 == 0, "DistGlobal cores must be cacheline-aligned");

static_assert(sizeof(DistGlobal) <= kDistEngineGlobalStateSize, "DistGlobal exceeds the reserved runtime arena size");
static_assert(
    alignof(DistGlobal) <= kDistEngineGlobalStateAlign, "DistGlobal exceeds the reserved runtime arena align"
);
