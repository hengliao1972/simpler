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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

// 本测试直接实例化 standalone 的设备公共实现，只消去地址空间修饰符；
// 不复制一份被测 ring 算法，vector reference 只保存抽象的区间/producer 语义。
#define PA_DEVICE inline
#define PA_GM
#include "pa_scheduler_core.h"

namespace {

using pa_scheduler::CountLiveMapEntries;
using pa_scheduler::DataType;
using pa_scheduler::InsertTensor;
using pa_scheduler::LookupTensor;
using pa_scheduler::MapEntry;
using pa_scheduler::ResetTensorMap;
using pa_scheduler::TensorDesc;
using pa_scheduler::TensorMap;
using pa_scheduler::TensorMapSlotIndex;
using pa_scheduler::AdvanceTensorMap;
using pa_scheduler::kMapBucketCapacity;
using pa_scheduler::kMapBuckets;
using pa_scheduler::kMapCapacity;
using pa_scheduler::kMapBucketShift;
using pa_scheduler::kTaskWindow;

static_assert(PTO_FDWIC_SHARED_MAP == 0, "this test covers only the private ring discipline");
static_assert(kMapCapacity == kMapBuckets * kMapBucketCapacity, "private TensorMap capacity mismatch");
static_assert(sizeof(MapEntry) == 48, "MapEntry must preserve the standalone/production-compatible ABI");
static_assert(alignof(MapEntry) == 8, "MapEntry alignment changed");
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(kMapBuckets == 128, "default private TensorMap ABI requires 128 buckets");
static_assert(kMapBucketCapacity == 128, "default private TensorMap ABI requires 128 slots per bucket");
static_assert(kMapBucketShift == 7, "default 128-bucket hash must consume seven high bits");
static_assert(sizeof(TensorMap) == 823312, "TensorMap must preserve the WorkerState ABI");
#endif
static_assert(alignof(TensorMap) == 8, "TensorMap alignment changed");
static_assert(std::is_standard_layout<MapEntry>::value, "MapEntry must remain a standard-layout ABI type");
static_assert(std::is_trivially_copyable<MapEntry>::value, "MapEntry must remain trivially copyable");
static_assert(std::is_standard_layout<TensorMap>::value, "TensorMap must remain a standard-layout ABI type");

int g_failures = 0;

void Expect(bool condition, const char *test, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] private TensorMap ring/%s: %s\n", test, message);
    ++g_failures;
}

void ExpectEqual(int64_t actual, int64_t expected, const char *test, const char *field) {
    if (actual == expected) {
        return;
    }
    std::fprintf(
        stderr, "[FAIL] private TensorMap ring/%s: %s actual=%lld expected=%lld\n",
        test, field, static_cast<long long>(actual), static_cast<long long>(expected)
    );
    ++g_failures;
}

uint64_t ElementBytes(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
        case DataType::Int32:
        case DataType::Uint32:
            return 4;
        case DataType::Float16:
        case DataType::Int16:
        case DataType::Bfloat16:
        case DataType::Uint16:
            return 2;
        case DataType::Int8:
        case DataType::Uint8:
        case DataType::Bool:
            return 1;
        case DataType::Int64:
        case DataType::Uint64:
            return 8;
        default:
            std::abort();
    }
}

TensorDesc MakeTensor(
    uint64_t buffer_addr, uint64_t start_offset, uint32_t extent, DataType dtype = DataType::Float32,
    bool contiguous = true
) {
    TensorDesc tensor{};
    tensor.buffer_addr = buffer_addr;
    tensor.buffer_size = 1ULL << 32;
    tensor.owner_task_id = UINT64_MAX;
    tensor.start_offset = start_offset;
    tensor.version = 0;
    tensor.ndims = 1;
    tensor.dtype = dtype;
    tensor.manual_dep = false;
    tensor.is_contiguous = contiguous;
    tensor.child_memory = 0;
    tensor.shapes[0] = extent;
    tensor.extent_elem_cache = extent;
    tensor.strides[0] = 1;
    return tensor;
}

struct ByteRange {
    uint64_t buffer_addr;
    uint64_t lo;
    uint64_t hi;
};

ByteRange ReferenceByteRange(const TensorDesc &tensor) {
    uint64_t extent = tensor.extent_elem_cache;
    if (tensor.is_contiguous) {
        extent = 1;
        for (uint32_t dimension = 0; dimension < tensor.ndims; ++dimension) {
            extent *= tensor.shapes[dimension];
        }
    }
    const uint64_t element_bytes = ElementBytes(tensor.dtype);
    return {
        tensor.buffer_addr,
        tensor.start_offset * element_bytes,
        (tensor.start_offset + extent) * element_bytes,
    };
}

uint32_t ReferenceBucket(uint64_t buffer_addr) {
    // 与设计文档固定的乘法哈希一致；reference 不读取被测 ring 的 head/tail，
    // 只需知道一条抽象 entry 属于哪个有界桶，才能独立判断插入是否应当失败。
#if PTO_FDWIC_TENSORMAP_RING_CAP == 16384
    (void)buffer_addr;
    return 0;
#else
    const uint64_t mixed = buffer_addr * 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(mixed >> (64U - kMapBucketShift));
#endif
}

bool Overlaps(const ByteRange &left, const ByteRange &right) {
    return left.buffer_addr == right.buffer_addr && left.lo < right.hi && right.lo < left.hi;
}

struct ReferenceEntry {
    ByteRange range;
    int32_t producer;
    uint32_t bucket;
};

class ReferenceMap {
public:
    void Reset() {
        entries_.clear();
        alive_floor_ = 0;
    }

    void Advance(uint32_t task_id, int32_t heap_window) {
        const int32_t candidate = static_cast<int32_t>(task_id) - heap_window;
        if (candidate <= alive_floor_) {
            return;
        }
        alive_floor_ = candidate;
        entries_.erase(
            std::remove_if(
                entries_.begin(), entries_.end(),
                [&](const ReferenceEntry &entry) { return entry.producer < alive_floor_; }
            ),
            entries_.end()
        );
    }

    bool Insert(const TensorDesc &tensor, int32_t producer) {
        const ByteRange range = ReferenceByteRange(tensor);
        const uint32_t bucket = ReferenceBucket(range.buffer_addr);
        const size_t live_in_bucket = static_cast<size_t>(std::count_if(
            entries_.begin(), entries_.end(),
            [&](const ReferenceEntry &entry) { return entry.bucket == bucket; }
        ));
        if (live_in_bucket >= kMapBucketCapacity) {
            return false;
        }
        entries_.push_back({range, producer, bucket});
        return true;
    }

    int32_t Lookup(const TensorDesc &tensor) const {
        const ByteRange query = ReferenceByteRange(tensor);
        int32_t best = -1;
        for (const ReferenceEntry &entry : entries_) {
            if (entry.producer >= alive_floor_ && Overlaps(entry.range, query)) {
                best = std::max(best, entry.producer);
            }
        }
        return best;
    }

    uint32_t Count() const { return static_cast<uint32_t>(entries_.size()); }

private:
    std::vector<ReferenceEntry> entries_;
    int32_t alive_floor_ = 0;
};

std::unique_ptr<TensorMap> NewMap() {
    // TensorMap 是大对象，测试也必须遵守生产侧的 heap/GM 放置假设，避免 host
    // 线程默认栈大小掩盖或制造与 ring 语义无关的失败。
    auto map = std::make_unique<TensorMap>();
    ResetTensorMap(*map);
    return map;
}

void TestPhysicalLayoutBoundaries() {
    constexpr const char *kTest = "physical-layout-boundaries";
    auto map = NewMap();
    ExpectEqual(
        TensorMapSlotIndex(0, 0), 0,
        kTest, "first physical slot"
    );
    ExpectEqual(
        TensorMapSlotIndex(
            kMapBuckets - 1U,
            static_cast<uint64_t>(kMapBucketCapacity - 1U)
        ),
        kMapCapacity - 1U,
        kTest, "last physical slot"
    );
    ExpectEqual(
        TensorMapSlotIndex(
            kMapBuckets - 1U,
            static_cast<uint64_t>(kMapBucketCapacity)
        ),
        (kMapBuckets - 1U) * kMapBucketCapacity,
        kTest, "last bucket first wrapped slot"
    );

    // 直接检查物理数组而不是用同一个 helper 算 expected。CAP32/64 必须
    // 跨过默认前128桶与 ABI padding 中额外游标的分界。
#if PTO_FDWIC_TENSORMAP_RING_CAP <= 128
    TensorMapBucketHead(*map, 127U) = 17;
    TensorMapBucketTail(*map, 127U) = 27;
    ExpectEqual(map->bucket_heads[127], 17, kTest, "base head bucket 127");
    ExpectEqual(map->bucket_tails[127], 27, kTest, "base tail bucket 127");
#endif
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
    TensorMapBucketHead(*map, 128U) = 18;
    TensorMapBucketTail(*map, 128U) = 28;
    TensorMapBucketHead(*map, kMapBuckets - 1U) = 19;
    TensorMapBucketTail(*map, kMapBuckets - 1U) = 29;
    ExpectEqual(map->extra_bucket_heads[0], 18, kTest, "extra head bucket 128");
    ExpectEqual(map->extra_bucket_tails[0], 28, kTest, "extra tail bucket 128");
    ExpectEqual(
        map->extra_bucket_heads[kMapBuckets - 129U], 19,
        kTest, "last extra head"
    );
    ExpectEqual(
        map->extra_bucket_tails[kMapBuckets - 129U], 29,
        kTest, "last extra tail"
    );
#else
    TensorMapBucketHead(*map, kMapBuckets - 1U) = 19;
    TensorMapBucketTail(*map, kMapBuckets - 1U) = 29;
    ExpectEqual(
        map->bucket_heads[kMapBuckets - 1U], 19,
        kTest, "last base head"
    );
    ExpectEqual(
        map->bucket_tails[kMapBuckets - 1U], 29,
        kTest, "last base tail"
    );
#endif
}

void TestEmptyAndHalfOpenIntervals() {
    constexpr const char *kTest = "empty-and-half-open";
    auto map = NewMap();
    const TensorDesc left = MakeTensor(0x100000000ULL, 0, 4);
    const TensorDesc touching = MakeTensor(0x100000000ULL, 4, 4);
    const TensorDesc overlap = MakeTensor(0x100000000ULL, 3, 2);

    ExpectEqual(CountLiveMapEntries(*map), 0, kTest, "reset live count");
    ExpectEqual(LookupTensor(*map, left), -1, kTest, "empty lookup");
    Expect(InsertTensor(*map, left, 2), kTest, "first insert must succeed");
    ExpectEqual(LookupTensor(*map, touching), -1, kTest, "touching half-open ranges");
    ExpectEqual(LookupTensor(*map, overlap), 2, kTest, "overlapping half-open ranges");
}

void TestLatestProducerAndAliveFloor() {
    constexpr const char *kTest = "latest-and-alive-floor";
    auto map = NewMap();
    const TensorDesc region = MakeTensor(0x200000000ULL, 8, 8);

    Expect(InsertTensor(*map, region, 3), kTest, "producer 3 insert");
    Expect(InsertTensor(*map, region, 7), kTest, "producer 7 insert");
    Expect(InsertTensor(*map, region, 5), kTest, "producer 5 insert");
    ExpectEqual(LookupTensor(*map, region), 7, kTest, "lookup must select maximum producer");

    auto boundary_map = NewMap();
    const TensorDesc producer_9 = MakeTensor(0x300000000ULL, 0, 1);
    const TensorDesc producer_10 = MakeTensor(0x300000000ULL, 2, 1);
    Expect(InsertTensor(*boundary_map, producer_9, 9), kTest, "producer 9 insert");
    Expect(InsertTensor(*boundary_map, producer_10, 10), kTest, "producer 10 insert");

    AdvanceTensorMap(*boundary_map, 20, 10);
    ExpectEqual(LookupTensor(*boundary_map, producer_9), -1, kTest, "producer below alive_floor");
    ExpectEqual(LookupTensor(*boundary_map, producer_10), 10, kTest, "producer at alive_floor");
    AdvanceTensorMap(*boundary_map, 21, 10);
    ExpectEqual(LookupTensor(*boundary_map, producer_10), -1, kTest, "producer after floor advances");
    ExpectEqual(CountLiveMapEntries(*boundary_map), 0, kTest, "floor retirement count");
}

void TestTaskWindowWrapAndMultipleLaps() {
    constexpr const char *kTest = "task-window-wrap-and-multiple-laps";
    auto map = NewMap();
    const TensorDesc region = MakeTensor(0x400000000ULL, 0, 4);
    // 该用例要验证“窗口内始终可追加并跨多 lap”，不是验证配置过小
    // 的 FATAL；因此窗口必须严格小于当前隔离 CAP。满环失败由下一用例
    // 独立覆盖。
    constexpr int32_t kWindow =
        kMapBucketCapacity > 64U
            ? 64
            : static_cast<int32_t>(kMapBucketCapacity - 1U);
    constexpr uint32_t kLastTask = 4 * kTaskWindow + 257;

    for (uint32_t task_id = 0; task_id <= kLastTask; ++task_id) {
        AdvanceTensorMap(*map, task_id, kWindow);
        if (!InsertTensor(*map, region, static_cast<int32_t>(task_id))) {
            std::fprintf(
                stderr, "[FAIL] private TensorMap ring/%s: insert failed at task=%u\n", kTest, task_id
            );
            ++g_failures;
            return;
        }
        const int32_t producer = LookupTensor(*map, region);
        if (producer != static_cast<int32_t>(task_id)) {
            std::fprintf(
                stderr,
                "[FAIL] private TensorMap ring/%s: latest producer at task=%u actual=%d\n",
                kTest, task_id, producer
            );
            ++g_failures;
            return;
        }
        const uint32_t live = CountLiveMapEntries(*map);
        if (live > static_cast<uint32_t>(kWindow + 1)) {
            std::fprintf(
                stderr, "[FAIL] private TensorMap ring/%s: live=%u at task=%u exceeds window\n",
                kTest, live, task_id
            );
            ++g_failures;
            return;
        }
    }

    ExpectEqual(
        CountLiveMapEntries(*map), static_cast<uint32_t>(kWindow + 1), kTest, "final live window"
    );
    ExpectEqual(LookupTensor(*map, region), kLastTask, kTest, "latest producer after multiple laps");
}

void TestFullBucketDoesNotOverwrite() {
    constexpr const char *kTest = "full-bucket-no-overwrite";
    auto map = NewMap();
    std::vector<TensorDesc> existing;
    existing.reserve(kMapBucketCapacity);

    for (uint32_t index = 0; index < kMapBucketCapacity; ++index) {
        existing.push_back(MakeTensor(0x500000000ULL, 2ULL * index, 1));
        if (!InsertTensor(*map, existing.back(), static_cast<int32_t>(index))) {
            std::fprintf(
                stderr, "[FAIL] private TensorMap ring/%s: insert %u of %u failed\n",
                kTest, index + 1, kMapBucketCapacity
            );
            ++g_failures;
            return;
        }
    }

    ExpectEqual(CountLiveMapEntries(*map), kMapBucketCapacity, kTest, "full bucket live count");
    const bool inserted = InsertTensor(*map, existing.front(), 1000);
    Expect(!inserted, kTest, "CAP+1 insert must fail");
    ExpectEqual(
        CountLiveMapEntries(*map), kMapBucketCapacity, kTest, "failed insert must not change count"
    );
    for (uint32_t index = 0; index < existing.size(); ++index) {
        const int32_t producer = LookupTensor(*map, existing[index]);
        if (producer != static_cast<int32_t>(index)) {
            std::fprintf(
                stderr,
                "[FAIL] private TensorMap ring/%s: failed insert changed old result index=%u actual=%d\n",
                kTest, index, producer
            );
            ++g_failures;
            return;
        }
    }
}

TensorDesc RandomTensor(std::mt19937_64 &random) {
    constexpr DataType kDtypes[] = {
        DataType::Float32,
        DataType::Float16,
        DataType::Uint8,
        DataType::Int64,
    };
    const uint64_t buffer_index = random() % 48;
    const uint64_t buffer_addr = 0x800000000ULL + buffer_index * 0x100000ULL;
    const uint64_t start_offset = random() % 96;
    const uint32_t extent = 1 + static_cast<uint32_t>(random() % 12);
    const DataType dtype = kDtypes[random() % (sizeof(kDtypes) / sizeof(kDtypes[0]))];
    const bool contiguous = (random() & 3U) != 0;
    return MakeTensor(buffer_addr, start_offset, extent, dtype, contiguous);
}

void TestFixedSeedDifferential() {
    constexpr const char *kTest = "fixed-seed-differential";
    constexpr uint64_t kSeed = 0x504152494E475631ULL;  // "PARINGV1"
    constexpr uint32_t kOperations = 12000;
    constexpr int32_t kWindow = 64;

    auto map = NewMap();
    ReferenceMap reference;
    reference.Reset();
    std::mt19937_64 random(kSeed);
    uint32_t task_id = 0;
    std::vector<TensorDesc> probes;
    probes.reserve(kOperations);

    for (uint32_t operation = 0; operation < kOperations; ++operation) {
        const uint32_t selector = static_cast<uint32_t>(random() % 100);
        if (selector < 28) {
            task_id += 1 + static_cast<uint32_t>(random() % 5);
            AdvanceTensorMap(*map, task_id, kWindow);
            reference.Advance(task_id, kWindow);
        } else if (selector < 68) {
            const TensorDesc tensor = RandomTensor(random);
            const bool actual = InsertTensor(*map, tensor, static_cast<int32_t>(task_id));
            const bool expected = reference.Insert(tensor, static_cast<int32_t>(task_id));
            probes.push_back(tensor);
            if (actual != expected) {
                std::fprintf(
                    stderr,
                    "[FAIL] private TensorMap ring/%s: insert op=%u task=%u actual=%d expected=%d\n",
                    kTest, operation, task_id, static_cast<int>(actual), static_cast<int>(expected)
                );
                ++g_failures;
                return;
            }
        } else {
            const TensorDesc query = RandomTensor(random);
            const int32_t actual = LookupTensor(*map, query);
            const int32_t expected = reference.Lookup(query);
            if (actual != expected) {
                std::fprintf(
                    stderr,
                    "[FAIL] private TensorMap ring/%s: lookup op=%u task=%u actual=%d expected=%d\n",
                    kTest, operation, task_id, actual, expected
                );
                ++g_failures;
                return;
            }
        }

        const uint32_t actual_count = CountLiveMapEntries(*map);
        const uint32_t expected_count = reference.Count();
        if (actual_count != expected_count) {
            std::fprintf(
                stderr,
                "[FAIL] private TensorMap ring/%s: count op=%u task=%u actual=%u expected=%u\n",
                kTest, operation, task_id, actual_count, expected_count
            );
            ++g_failures;
            return;
        }

        // 每个操作后都对一个既往区间做可见状态差分，而不是只比较 live
        // 数量。这样 Advance 的错误退休、Insert 写错槽和满桶失败后误推进
        // 游标，都会在发生的同一步暴露，不依赖后续随机序列碰巧再次查询。
        if (!probes.empty()) {
            const size_t probe_index =
                (static_cast<size_t>(operation) * 0x9E3779B1ULL) % probes.size();
            const int32_t actual = LookupTensor(*map, probes[probe_index]);
            const int32_t expected = reference.Lookup(probes[probe_index]);
            if (actual != expected) {
                std::fprintf(
                    stderr,
                    "[FAIL] private TensorMap ring/%s: state op=%u task=%u probe=%zu "
                    "actual=%d expected=%d\n",
                    kTest, operation, task_id, probe_index, actual, expected
                );
                ++g_failures;
                return;
            }
        }
    }
}

}  // namespace

int main() {
    TestPhysicalLayoutBoundaries();
    TestEmptyAndHalfOpenIntervals();
    TestLatestProducerAndAliveFloor();
    TestTaskWindowWrapAndMultipleLaps();
    TestFullBucketDoesNotOverwrite();
    TestFixedSeedDifferential();

    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] private TensorMap ring: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf(
        "[PASS] private TensorMap ring CAP=%u buckets=%u: "
        "ABI, interval, reclaim, wrap, overflow, differential\n",
        kMapBucketCapacity, kMapBuckets
    );
    return EXIT_SUCCESS;
}
