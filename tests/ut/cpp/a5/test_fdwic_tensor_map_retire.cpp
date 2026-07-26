/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE file.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "inner_kernel.h"
#include "dist_engine/common/state.h"
#include "dist_engine/aicore/tensor_map.h"

[[noreturn]] void assert_impl(const char *condition, const char *, int) { throw std::logic_error(condition); }

namespace {

// 保留优化前的 retire 控制流作为差分参考。entry 摘链仍复用生产 helper，
// 因而测试只对比本次候选涉及的 task-head 分支，不复制整套 TensorMap 实现。
void advance_retire_reference(DistTensorMap &map, int32_t task_id, int32_t history) {
    const int32_t new_floor = task_id - history;
    if (new_floor <= map.cleaned_upto) {
        if (new_floor > map.alive_floor) map.alive_floor = new_floor;
        return;
    }
    for (int32_t id = map.cleaned_upto; id < new_floor; ++id) {
        int32_t current = map.task_heads[id & kTaskWindowMask];
        while (current >= 0) {
            const int32_t next = map.entries[current].next_in_task;
            EXPECT_EQ(map.entries[current].producer, id);
            dist_private_tensor_map_free_entry(map, current);
            current = next;
        }
        map.task_heads[id & kTaskWindowMask] = -1;
    }
    map.cleaned_upto = new_floor;
    map.alive_floor = new_floor;
}

std::unique_ptr<DistTensorMap> make_empty_map() {
    auto map = std::make_unique<DistTensorMap>();
    dist_private_tensor_map_reset(*map);
    return map;
}

std::unique_ptr<DistTensorMap> clone_map_bytes(const DistTensorMap &source) {
    static_assert(std::is_trivially_copyable_v<DistTensorMap>);
    auto clone = std::make_unique<DistTensorMap>();
    std::memcpy(clone.get(), &source, sizeof(DistTensorMap));
    return clone;
}

void expect_exact_map_state(const DistTensorMap &actual, const DistTensorMap &expected) {
    // expected 是 actual 在调用前的逐字节副本；两边只经过确定性的生产/参考
    // retire，因此这里可以连同未触及的 entry 一起检查，防止快路误写相邻状态。
    EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(DistTensorMap)), 0);
}

void seed_entry(
    DistTensorMap &map, int32_t index, int32_t producer, int32_t bucket, int32_t previous_in_bucket,
    int32_t next_in_bucket, int32_t next_in_task
) {
    MapEntry &entry = map.entries[index];
    entry.buf_addr = static_cast<uint64_t>(0x1000 + index * 0x100);
    entry.lo = static_cast<uint64_t>(index * 16);
    entry.hi = entry.lo + 16;
    entry.producer = producer;
    entry.bucket = bucket;
    entry.prev_in_bucket = previous_in_bucket;
    entry.next_in_bucket = next_in_bucket;
    entry.next_in_task = next_in_task;
}

Tensor make_region(uint64_t address) {
    Tensor tensor{};
    tensor.buffer = {address, 4};
    tensor.start_offset = 0;
    tensor.ndims = 1;
    tensor.dtype = DataType::FLOAT32;
    tensor.is_contiguous = true;
    tensor.shapes[0] = 1;
    tensor.extent_elem_cache = 1;
    tensor.strides[0] = 1;
    return tensor;
}

Tensor make_logical_test_tensor(
    uint64_t address, uint64_t start_offset, uint32_t extent, DataType dtype = DataType::FLOAT32,
    bool contiguous = true, uint32_t stride = 1
) {
    Tensor tensor{};
    tensor.buffer = {address, 1ULL << 32};
    tensor.owner_task_id = PTO2TaskId::invalid();
    tensor.start_offset = start_offset;
    tensor.version = 0;
    tensor.ndims = 1;
    tensor.dtype = dtype;
    tensor.manual_dep = false;
    tensor.is_contiguous = contiguous;
    tensor.child_memory = 0;
    tensor.shapes[0] = extent;
    tensor.strides[0] = stride;
    tensor.extent_elem_cache = contiguous ? extent : 1 + static_cast<uint64_t>(extent - 1) * stride;
    return tensor;
}

uint64_t reference_element_bytes(DataType dtype) {
    // 独立列出 wire dtype 的元素宽度，不调用 production get_element_size()，
    // 避免被测 byte-range helper 与 reference 共享同一个错误。
    switch (dtype) {
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FLOAT16:
        case DataType::INT16:
        case DataType::BFLOAT16:
        case DataType::UINT16:
            return 2;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
            return 1;
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::DATA_TYPE_NUM:
            break;
    }
    throw std::logic_error("unexpected dtype in TensorMap logical reference");
}

struct LogicalByteRange {
    uint64_t address;
    uint64_t lo;
    uint64_t hi;
};

LogicalByteRange reference_byte_range(const Tensor &tensor) {
    uint64_t extent = tensor.extent_elem_cache;
    if (tensor.is_contiguous) {
        extent = 1;
        for (uint32_t dimension = 0; dimension < tensor.ndims; ++dimension) {
            extent *= tensor.shapes[dimension];
        }
    }
    const uint64_t element_bytes = reference_element_bytes(tensor.dtype);
    return {
        tensor.buffer.addr,
        tensor.start_offset * element_bytes,
        (tensor.start_offset + extent) * element_bytes,
    };
}

bool logical_ranges_overlap(const LogicalByteRange &left, const LogicalByteRange &right) {
    return left.address == right.address && left.lo < right.hi && right.lo < left.hi;
}

struct LogicalReferenceEntry {
    LogicalByteRange range;
    int32_t producer;
};

class LogicalReferenceMap {
public:
    void advance(int32_t task_id, int32_t history) {
        const int32_t floor = task_id - history;
        if (floor <= alive_floor_) {
            return;
        }
        alive_floor_ = floor;
        entries_.erase(
            std::remove_if(
                entries_.begin(), entries_.end(),
                [floor](const LogicalReferenceEntry &entry) { return entry.producer < floor; }
            ),
            entries_.end()
        );
    }

    void insert(const Tensor &tensor, int32_t producer) {
        entries_.push_back({reference_byte_range(tensor), producer});
    }

    int32_t lookup(const Tensor &tensor) const {
        const LogicalByteRange query = reference_byte_range(tensor);
        int32_t best = -1;
        for (const LogicalReferenceEntry &entry : entries_) {
            if (entry.producer >= alive_floor_ && logical_ranges_overlap(entry.range, query)) {
                best = std::max(best, entry.producer);
            }
        }
        return best;
    }

private:
    std::vector<LogicalReferenceEntry> entries_;
    int32_t alive_floor_ = 0;
};

void expect_logical_lookup_matches(
    const DistTensorMap &actual, const LogicalReferenceMap &reference, const Tensor &query, const char *context,
    int32_t task_id = -1
) {
    EXPECT_EQ(dist_private_tensor_map_lookup(actual, query), reference.lookup(query))
        << "context=" << context << " task_id=" << task_id;
}

Tensor make_random_logical_tensor(std::mt19937_64 &random) {
    constexpr std::array<DataType, 4> kDtypes = {
        DataType::FLOAT32,
        DataType::FLOAT16,
        DataType::UINT8,
        DataType::INT64,
    };
    const uint64_t buffer_index = random() % 48;
    const uint64_t address = 0x800000000ULL + buffer_index * 0x100000ULL;
    const uint64_t start_offset = random() % 96;
    const uint32_t extent = 1 + static_cast<uint32_t>(random() % 12);
    const DataType dtype = kDtypes[random() % kDtypes.size()];
    const bool contiguous = (random() & 3U) != 0;
    const uint32_t stride = contiguous ? 1 : 2 + static_cast<uint32_t>(random() % 4);
    return make_logical_test_tensor(address, start_offset, extent, dtype, contiguous, stride);
}

TEST(FdwicTensorMapLogical, EmptyAndHalfOpenRangesMatchIndependentReference) {
    auto actual = make_empty_map();
    LogicalReferenceMap reference;
    const Tensor left = make_logical_test_tensor(0x100000000ULL, 0, 4);
    const Tensor touching = make_logical_test_tensor(0x100000000ULL, 4, 4);
    const Tensor overlap = make_logical_test_tensor(0x100000000ULL, 3, 2);
    const Tensor different_buffer = make_logical_test_tensor(0x100100000ULL, 0, 4);

    expect_logical_lookup_matches(*actual, reference, left, "empty");
    EXPECT_EQ(reference.lookup(left), -1);

    ASSERT_TRUE(dist_private_tensor_map_insert(*actual, left, 2));
    reference.insert(left, 2);
    expect_logical_lookup_matches(*actual, reference, touching, "touching-half-open");
    expect_logical_lookup_matches(*actual, reference, overlap, "overlap");
    expect_logical_lookup_matches(*actual, reference, different_buffer, "different-buffer");
    EXPECT_EQ(reference.lookup(touching), -1);
    EXPECT_EQ(reference.lookup(overlap), 2);
    EXPECT_EQ(reference.lookup(different_buffer), -1);
}

TEST(FdwicTensorMapLogical, LookupSelectsMaximumOverlappingProducer) {
    auto actual = make_empty_map();
    LogicalReferenceMap reference;
    const uint64_t address = 0x200000000ULL;
    const Tensor producer_3 = make_logical_test_tensor(address, 0, 8);
    const Tensor producer_5 = make_logical_test_tensor(address, 2, 4);
    const Tensor producer_7 = make_logical_test_tensor(address, 3, 2);
    const Tensor query = make_logical_test_tensor(address, 3, 1);

    for (const auto &[tensor, producer] :
         std::array<std::pair<const Tensor *, int32_t>, 3>{
             std::pair{&producer_3, 3},
             std::pair{&producer_5, 5},
             std::pair{&producer_7, 7},
         }) {
        ASSERT_TRUE(dist_private_tensor_map_insert(*actual, *tensor, producer));
        reference.insert(*tensor, producer);
    }

    expect_logical_lookup_matches(*actual, reference, query, "maximum-overlapping-producer");
    EXPECT_EQ(reference.lookup(query), 7);
}

TEST(FdwicTensorMapLogical, HistoryFloorIsHalfOpenAndMonotonic) {
    auto actual = make_empty_map();
    LogicalReferenceMap reference;
    const Tensor producer_9 = make_logical_test_tensor(0x300000000ULL, 0, 1);
    const Tensor producer_10 = make_logical_test_tensor(0x300000000ULL, 2, 1);

    ASSERT_TRUE(dist_private_tensor_map_insert(*actual, producer_9, 9));
    ASSERT_TRUE(dist_private_tensor_map_insert(*actual, producer_10, 10));
    reference.insert(producer_9, 9);
    reference.insert(producer_10, 10);

    dist_private_tensor_map_advance_retire(*actual, 20, 10);
    reference.advance(20, 10);
    expect_logical_lookup_matches(*actual, reference, producer_9, "below-floor");
    expect_logical_lookup_matches(*actual, reference, producer_10, "at-floor");
    EXPECT_EQ(reference.lookup(producer_9), -1);
    EXPECT_EQ(reference.lookup(producer_10), 10);

    dist_private_tensor_map_advance_retire(*actual, 20, 10);
    reference.advance(20, 10);
    dist_private_tensor_map_advance_retire(*actual, 19, 10);
    reference.advance(19, 10);
    expect_logical_lookup_matches(*actual, reference, producer_10, "repeated-and-lower-floor");
    EXPECT_EQ(reference.lookup(producer_10), 10);

    dist_private_tensor_map_advance_retire(*actual, 21, 10);
    reference.advance(21, 10);
    expect_logical_lookup_matches(*actual, reference, producer_10, "after-floor-advance");
    EXPECT_EQ(reference.lookup(producer_10), -1);
}

TEST(FdwicTensorMapLogical, FixedSeedTwelveThousandTasksMatchIndependentReference) {
    constexpr uint64_t kSeed = 0x5041524F445631ULL;  // "PARODV1"
    constexpr int32_t kHistory = 15;
    constexpr int32_t kTasks = 12000;
    constexpr size_t kRecentCapacity = static_cast<size_t>(kHistory + 1);

    auto actual = make_empty_map();
    LogicalReferenceMap reference;
    std::mt19937_64 random(kSeed);
    std::array<Tensor, kRecentCapacity> recent{};
    size_t recent_count = 0;

    for (int32_t task_id = 0; task_id < kTasks; ++task_id) {
        dist_private_tensor_map_advance_retire(*actual, task_id, kHistory);
        reference.advance(task_id, kHistory);

        const size_t slot = static_cast<size_t>(task_id) % recent.size();
        if (task_id >= static_cast<int32_t>(recent.size())) {
            expect_logical_lookup_matches(*actual, reference, recent[slot], "retired-slot-before-reuse", task_id);
        }

        const Tensor inserted = make_random_logical_tensor(random);
        ASSERT_TRUE(dist_private_tensor_map_insert(*actual, inserted, task_id))
            << "task_id=" << task_id
            << " workload keeps at most " << kRecentCapacity << " globally live entries";
        reference.insert(inserted, task_id);
        recent[slot] = inserted;
        recent_count = std::min(recent_count + 1, recent.size());

        expect_logical_lookup_matches(*actual, reference, inserted, "just-inserted", task_id);
        const size_t recent_slot = static_cast<size_t>(random() % recent_count);
        expect_logical_lookup_matches(*actual, reference, recent[recent_slot], "recent-probe", task_id);
        const Tensor random_query = make_random_logical_tensor(random);
        expect_logical_lookup_matches(*actual, reference, random_query, "random-query", task_id);
    }
}

TEST(FdwicTensorMapRetire, EmptySentinelAndDefensiveNegativeValueMatchOriginalState) {
    auto actual = make_empty_map();
    actual->task_heads[1] = -2;
    auto expected = clone_map_bytes(*actual);

    // new_floor=3：id 0/2 是正常空链 -1；id 1 是防御性异常负值。
    // 候选只能跳过精确的 -1，仍须把其他负值归一为 -1。
    dist_private_tensor_map_advance_retire(*actual, 67, 64);
    advance_retire_reference(*expected, 67, 64);

    expect_exact_map_state(*actual, *expected);
    EXPECT_EQ(actual->task_heads[0], -1);
    EXPECT_EQ(actual->task_heads[1], -1);
    EXPECT_EQ(actual->task_heads[2], -1);
    EXPECT_EQ(actual->cleaned_upto, 3);
    EXPECT_EQ(actual->alive_floor, 3);
    EXPECT_EQ(actual->free_head, -1);
    EXPECT_EQ(actual->high_water, 0);
}

TEST(FdwicTensorMapRetire, NonEmptyTaskChainsPreserveBucketAndFreeListSemantics) {
    auto actual = make_empty_map();
    actual->high_water = 4;

    // bucket 7: retired entry 0 -> live entry 1。
    actual->buckets[7] = 0;
    seed_entry(*actual, 0, 0, 7, -1, 1, -1);
    seed_entry(*actual, 1, 5, 7, 0, -1, -1);
    actual->task_heads[0] = 0;
    actual->task_heads[5] = 1;

    // bucket 9 与 task 2 都是 entry 2 -> entry 3；两项应按原顺序进入 free list。
    actual->buckets[9] = 2;
    seed_entry(*actual, 2, 2, 9, -1, 3, 3);
    seed_entry(*actual, 3, 2, 9, 2, -1, -1);
    actual->task_heads[2] = 2;

    auto expected = clone_map_bytes(*actual);
    dist_private_tensor_map_advance_retire(*actual, 67, 64);
    advance_retire_reference(*expected, 67, 64);

    expect_exact_map_state(*actual, *expected);
    EXPECT_EQ(actual->buckets[7], 1);
    EXPECT_EQ(actual->entries[1].prev_in_bucket, -1);
    EXPECT_EQ(actual->buckets[9], -1);
    EXPECT_EQ(actual->task_heads[0], -1);
    EXPECT_EQ(actual->task_heads[2], -1);
    EXPECT_EQ(actual->task_heads[5], 1);
    EXPECT_EQ(actual->free_head, 3);
    EXPECT_EQ(actual->entries[3].next_in_bucket, 2);
    EXPECT_EQ(actual->entries[2].next_in_bucket, 0);
    EXPECT_EQ(actual->entries[0].next_in_bucket, -1);
    EXPECT_EQ(actual->high_water, 4);
    EXPECT_EQ(actual->cleaned_upto, 3);
    EXPECT_EQ(actual->alive_floor, 3);
}

TEST(FdwicTensorMapRetire, ReusedTaskWindowSlotAndRepeatedFloorsRemainDeterministic) {
    auto actual = make_empty_map();
    actual->cleaned_upto = kTaskWindow;
    actual->alive_floor = kTaskWindow;
    actual->high_water = 1;
    actual->buckets[11] = 0;
    seed_entry(*actual, 0, kTaskWindow, 11, -1, -1, -1);
    actual->task_heads[0] = 0;  // producer 1024 复用 task-window 槽 0。

    auto expected = clone_map_bytes(*actual);
    dist_private_tensor_map_advance_retire(*actual, kTaskWindow + 65, 64);
    advance_retire_reference(*expected, kTaskWindow + 65, 64);

    expect_exact_map_state(*actual, *expected);
    EXPECT_EQ(actual->task_heads[0], -1);
    EXPECT_EQ(actual->buckets[11], -1);
    EXPECT_EQ(actual->free_head, 0);
    EXPECT_EQ(actual->cleaned_upto, kTaskWindow + 1);
    EXPECT_EQ(actual->alive_floor, kTaskWindow + 1);

    const auto after_first_retire = clone_map_bytes(*actual);
    dist_private_tensor_map_advance_retire(*actual, kTaskWindow + 65, 64);
    dist_private_tensor_map_advance_retire(*actual, 100, 64);
    expect_exact_map_state(*actual, *after_first_retire);
}

TEST(FdwicTensorMapRetire, LastPhysicalSlotSucceedsAndOverflowLeavesMapUnchanged) {
    auto map = make_empty_map();
    map->high_water = kMapCap - 1;
    const Tensor last = make_region(0x100000);

    EXPECT_TRUE(dist_private_tensor_map_insert(*map, last, 7));
    EXPECT_EQ(map->high_water, kMapCap);

    const auto full_state = clone_map_bytes(*map);
    const Tensor overflow = make_region(0x200000);
    EXPECT_FALSE(dist_private_tensor_map_insert(*map, overflow, 8));
    expect_exact_map_state(*map, *full_state);
}

TEST(FdwicTensorMapRetire, RetiredFreeSlotCanBeReusedAfterHighWaterReachesCapacity) {
    auto map = make_empty_map();
    map->high_water = kMapCap;
    map->free_head = 19;
    map->entries[19].next_in_bucket = -1;
    const Tensor tensor = make_region(0x300000);

    EXPECT_TRUE(dist_private_tensor_map_insert(*map, tensor, 9));
    EXPECT_EQ(map->high_water, kMapCap);
    EXPECT_EQ(map->free_head, -1);
    EXPECT_EQ(map->entries[19].producer, 9);
}

TEST(FdwicTensorMapRetire, FacadePropagatesBackendCapacityFailureWithoutMutation) {
    auto worker = std::make_unique<DistCore>();
    dist_tensor_map_reset_worker(*worker);
    worker->map.high_water = kMapCap;
    const auto full_state = clone_map_bytes(worker->map);
    const Tensor tensor = make_region(0x400000);

    EXPECT_FALSE(dist_tensor_map_insert_for_task(*worker, tensor, 10, /*task_won=*/true));
    expect_exact_map_state(worker->map, *full_state);
}

}  // namespace
