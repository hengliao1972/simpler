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

#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

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

}  // namespace
