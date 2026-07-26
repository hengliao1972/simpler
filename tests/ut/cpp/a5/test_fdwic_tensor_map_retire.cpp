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
    // expected 是调用前的逐字节副本。失败路径必须连同未触及 entry 与 ABI
    // 保留区一起保持不变，防止满环检查后仍误写 slot 或 cursor。
    EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(DistTensorMap)), 0);
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

Tensor make_region_in_another_bucket(uint64_t address) {
    const uint32_t original_bucket = dist_private_tensor_map_hash(address);
    for (uint64_t candidate = address + 64; candidate < address + (1ULL << 30); candidate += 64) {
        if (dist_private_tensor_map_hash(candidate) != original_bucket) {
            return make_region(candidate);
        }
    }
    throw std::logic_error("failed to find a TensorMap address in another bucket");
}

Tensor make_region_in_bucket(uint32_t target_bucket, uint64_t seed) {
    constexpr uint64_t kSearchSteps = 1ULL << 20;
    for (uint64_t step = 0; step < kSearchSteps; ++step) {
        const uint64_t candidate = seed + step * 64;
        if (dist_private_tensor_map_hash(candidate) == target_bucket) {
            return make_region(candidate);
        }
    }
    throw std::logic_error("failed to find a TensorMap address in the requested bucket");
}

void fill_region_bucket(DistTensorMap &map, const Tensor &tensor, uint32_t count, int32_t first_producer = 0) {
    for (uint32_t index = 0; index < count; ++index) {
        ASSERT_TRUE(dist_private_tensor_map_insert(map, tensor, first_producer + static_cast<int32_t>(index)))
            << "index=" << index << " count=" << count << " cap=" << kMapBucketCapacity;
    }
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
    DistTensorMap &actual, const LogicalReferenceMap &reference, const Tensor &query, const char *context,
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

TEST(FdwicTensorMapRing, SameBucketProducerContractAcceptsEqualAndIncreasingAndRejectsDecreasing) {
    auto map = make_empty_map();
    const Tensor region = make_region(0x380000);

    ASSERT_TRUE(dist_private_tensor_map_insert(*map, region, 3));
    ASSERT_TRUE(dist_private_tensor_map_insert(*map, region, 3));
    ASSERT_TRUE(dist_private_tensor_map_insert(*map, region, 4));
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, region), 4);

    const auto before_decreasing_insert = clone_map_bytes(*map);
    EXPECT_THROW(
        (void)dist_private_tensor_map_insert(*map, region, 2),
        std::logic_error
    );
    expect_exact_map_state(*map, *before_decreasing_insert);
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

TEST(FdwicTensorMapRing, ResetAndPhysicalBoundariesMatchTheFixedPool) {
    auto map = make_empty_map();

    EXPECT_EQ(sizeof(DistTensorMap), 823312U);
    EXPECT_EQ(sizeof(MapEntry), 48U);
    EXPECT_EQ(kMapBuckets * kMapBucketCapacity, static_cast<uint32_t>(kMapCap));
    EXPECT_EQ(dist_private_tensor_map_slot_index(0, 0), 0U);
    EXPECT_EQ(
        dist_private_tensor_map_slot_index(kMapBuckets - 1, kMapBucketCapacity - 1),
        static_cast<uint32_t>(kMapCap - 1)
    );
    EXPECT_EQ(
        dist_private_tensor_map_slot_index(kMapBuckets - 1, kMapBucketCapacity),
        (kMapBuckets - 1) * kMapBucketCapacity
    );
    EXPECT_EQ(map->alive_floor, 0);
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), 0U);
        EXPECT_EQ(dist_private_tensor_map_load_tail(*map, bucket), 0U);
    }
}

TEST(FdwicTensorMapRing, PrepareMapMovesOnlyTheFloorAndTouchedBucketsRetireLazily) {
    auto map = make_empty_map();
    const Tensor region = make_region(0x100000);
    ASSERT_TRUE(dist_private_tensor_map_insert(*map, region, 9));
    ASSERT_TRUE(dist_private_tensor_map_insert(*map, region, 10));
    const uint32_t bucket = dist_private_tensor_map_hash(region.buffer.addr);

    dist_private_tensor_map_advance_retire(*map, 20, 10);
    EXPECT_EQ(map->alive_floor, 10);
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, region), 10);
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), 1U);

    dist_private_tensor_map_advance_retire(*map, 20, 10);
    dist_private_tensor_map_advance_retire(*map, 19, 10);
    EXPECT_EQ(map->alive_floor, 10);
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), 1U);

    dist_private_tensor_map_advance_retire(*map, 21, 10);
    EXPECT_EQ(map->alive_floor, 11);
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), 1U);
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, region), -1);
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), 2U);
}

TEST(FdwicTensorMapRing, DifferentBuffersInTheSameBucketRemainLogicallyIndependent) {
    auto map = make_empty_map();
    const Tensor first = make_region(0x180000);
    const uint32_t bucket = dist_private_tensor_map_hash(first.buffer.addr);
    const Tensor second = make_region_in_bucket(bucket, 0x700000000ULL);
    ASSERT_NE(first.buffer.addr, second.buffer.addr);
    ASSERT_EQ(dist_private_tensor_map_hash(second.buffer.addr), bucket);

    ASSERT_TRUE(dist_private_tensor_map_insert(*map, first, 7));
    ASSERT_TRUE(dist_private_tensor_map_insert(*map, second, 8));
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, first), 7);
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, second), 8);

    ASSERT_TRUE(dist_private_tensor_map_insert(*map, first, 9));
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, first), 9);
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, second), 8);
}

TEST(FdwicTensorMapRing, FullyRetiredBucketReusesANonzeroCursorWithoutExposingStaleSlots) {
    auto map = make_empty_map();
    const Tensor region = make_region(0x1c0000);
    const uint32_t bucket = dist_private_tensor_map_hash(region.buffer.addr);
    fill_region_bucket(*map, region, kMapBucketCapacity);

    dist_private_tensor_map_advance_retire(
        *map, static_cast<int32_t>(kMapBucketCapacity), /*history=*/0
    );
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_lookup(*map, region), -1);
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), kMapBucketCapacity);
    EXPECT_EQ(dist_private_tensor_map_load_tail(*map, bucket), kMapBucketCapacity);

    ASSERT_TRUE(
        dist_private_tensor_map_insert(
            *map, region, static_cast<int32_t>(kMapBucketCapacity)
        )
    );
    EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), kMapBucketCapacity);
    EXPECT_EQ(dist_private_tensor_map_load_tail(*map, bucket), kMapBucketCapacity + 1);
    EXPECT_EQ(
        map->entries[dist_private_tensor_map_slot_index(bucket, kMapBucketCapacity)].producer,
        static_cast<int32_t>(kMapBucketCapacity)
    );
    EXPECT_EQ(
        dist_private_tensor_map_lookup(*map, region),
        static_cast<int32_t>(kMapBucketCapacity)
    );
}

TEST(FdwicTensorMapRing, FullBucketRejectsWithoutMutationAndDoesNotBlockAnotherBucket) {
    auto map = make_empty_map();
    const Tensor full = make_region(0x200000);
    const uint32_t full_bucket = dist_private_tensor_map_hash(full.buffer.addr);
    fill_region_bucket(*map, full, kMapBucketCapacity);
    EXPECT_EQ(dist_private_tensor_map_load_tail(*map, full_bucket), kMapBucketCapacity);

    const auto full_state = clone_map_bytes(*map);
    EXPECT_FALSE(dist_private_tensor_map_insert(*map, full, static_cast<int32_t>(kMapBucketCapacity)));
    expect_exact_map_state(*map, *full_state);

    if constexpr (kMapBuckets > 1) {
        const uint32_t full_slot_begin = full_bucket * kMapBucketCapacity;
        std::vector<MapEntry> full_entries_before(kMapBucketCapacity);
        std::memcpy(
            full_entries_before.data(), &map->entries[full_slot_begin],
            full_entries_before.size() * sizeof(MapEntry)
        );
        const Tensor independent = make_region_in_another_bucket(full.buffer.addr);
        const uint32_t independent_bucket = dist_private_tensor_map_hash(independent.buffer.addr);
        ASSERT_NE(independent_bucket, full_bucket);
        ASSERT_TRUE(
            dist_private_tensor_map_insert(*map, independent, static_cast<int32_t>(kMapBucketCapacity + 1))
        );
        EXPECT_EQ(dist_private_tensor_map_load_head(*map, full_bucket), 0U);
        EXPECT_EQ(dist_private_tensor_map_load_tail(*map, full_bucket), kMapBucketCapacity);
        EXPECT_EQ(
            std::memcmp(
                full_entries_before.data(), &map->entries[full_slot_begin],
                full_entries_before.size() * sizeof(MapEntry)
            ),
            0
        );
        EXPECT_EQ(dist_private_tensor_map_lookup(*map, independent), static_cast<int32_t>(kMapBucketCapacity + 1));
    }
}

TEST(FdwicTensorMapRing, BaseAndExtraBucketControlBoundaryIsIsolated) {
    if constexpr (kMapBuckets > kMapBaseControlBuckets) {
        constexpr uint32_t kLastBaseBucket = kMapBaseControlBuckets - 1;
        constexpr uint32_t kFirstExtraBucket = kMapBaseControlBuckets;
        const Tensor base_region = make_region_in_bucket(kLastBaseBucket, 0x500000000ULL);
        const Tensor extra_region = make_region_in_bucket(kFirstExtraBucket, 0x600000000ULL);
        auto map = make_empty_map();

        ASSERT_EQ(dist_private_tensor_map_hash(base_region.buffer.addr), kLastBaseBucket);
        ASSERT_EQ(dist_private_tensor_map_hash(extra_region.buffer.addr), kFirstExtraBucket);
        ASSERT_TRUE(dist_private_tensor_map_insert(*map, base_region, 1));
        ASSERT_TRUE(dist_private_tensor_map_insert(*map, base_region, 2));
        ASSERT_TRUE(dist_private_tensor_map_insert(*map, extra_region, 1));

        EXPECT_EQ(dist_private_tensor_map_load_head(*map, kLastBaseBucket), 0U);
        EXPECT_EQ(dist_private_tensor_map_load_tail(*map, kLastBaseBucket), 2U);
        EXPECT_EQ(dist_private_tensor_map_load_head(*map, kFirstExtraBucket), 0U);
        EXPECT_EQ(dist_private_tensor_map_load_tail(*map, kFirstExtraBucket), 1U);

        const uint32_t base_slot = dist_private_tensor_map_slot_index(kLastBaseBucket, 0);
        const uint32_t extra_slot = dist_private_tensor_map_slot_index(kFirstExtraBucket, 0);
        ASSERT_NE(base_slot, extra_slot);
        EXPECT_EQ(map->entries[base_slot].buf_addr, base_region.buffer.addr);
        EXPECT_EQ(map->entries[base_slot].producer, 1);
        EXPECT_EQ(map->entries[base_slot + 1].buf_addr, base_region.buffer.addr);
        EXPECT_EQ(map->entries[base_slot + 1].producer, 2);
        EXPECT_EQ(map->entries[extra_slot].buf_addr, extra_region.buffer.addr);
        EXPECT_EQ(map->entries[extra_slot].producer, 1);
        EXPECT_EQ(dist_private_tensor_map_lookup(*map, base_region), 2);
        EXPECT_EQ(dist_private_tensor_map_lookup(*map, extra_region), 1);
    }
}

TEST(FdwicTensorMapRing, RetiredSlotsWrapForThreeLapsWithoutExposingOldValues) {
    auto map = make_empty_map();
    const Tensor region = make_region(0x300000);
    const uint32_t bucket = dist_private_tensor_map_hash(region.buffer.addr);
    fill_region_bucket(*map, region, kMapBucketCapacity);

    for (uint32_t lap = 1; lap <= 3; ++lap) {
        const uint32_t producer_base = lap * kMapBucketCapacity;
        for (uint32_t offset = 0; offset < kMapBucketCapacity; ++offset) {
            const uint32_t producer = producer_base + offset;
            const uint32_t new_floor = producer - kMapBucketCapacity + 1;
            dist_private_tensor_map_advance_retire(*map, static_cast<int32_t>(new_floor), 0);
            ASSERT_TRUE(dist_private_tensor_map_insert(*map, region, static_cast<int32_t>(producer)))
                << "lap=" << lap << " offset=" << offset;
        }

        const uint64_t expected_head = static_cast<uint64_t>(lap) * kMapBucketCapacity;
        const uint64_t expected_tail = static_cast<uint64_t>(lap + 1) * kMapBucketCapacity;
        EXPECT_EQ(dist_private_tensor_map_load_head(*map, bucket), expected_head);
        EXPECT_EQ(dist_private_tensor_map_load_tail(*map, bucket), expected_tail);
        EXPECT_EQ(
            dist_private_tensor_map_lookup(*map, region),
            static_cast<int32_t>(expected_tail - 1)
        );
        for (uint32_t slot_offset = 0; slot_offset < kMapBucketCapacity; ++slot_offset) {
            const uint32_t physical_slot = dist_private_tensor_map_slot_index(bucket, slot_offset);
            EXPECT_EQ(map->entries[physical_slot].buf_addr, region.buffer.addr)
                << "lap=" << lap << " slot_offset=" << slot_offset;
            EXPECT_EQ(map->entries[physical_slot].producer, static_cast<int32_t>(producer_base + slot_offset))
                << "lap=" << lap << " slot_offset=" << slot_offset;
        }
    }
}

TEST(FdwicTensorMapRing, FacadePropagatesPerBucketCapacityFailureWithoutMutation) {
    auto worker = std::make_unique<DistCore>();
    dist_tensor_map_reset_worker(*worker);
    const Tensor tensor = make_region(0x400000);
    fill_region_bucket(worker->map, tensor, kMapBucketCapacity);
    const auto full_state = clone_map_bytes(worker->map);

    EXPECT_FALSE(dist_tensor_map_insert_for_task(*worker, tensor, 10, /*task_won=*/true));
    expect_exact_map_state(worker->map, *full_state);
}

}  // namespace
