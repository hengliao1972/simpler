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

namespace {

PTO_DEVICE_FUNC inline uint64_t
dist_private_tensor_map_load_head(__gm__ const DistTensorMap &self, uint32_t bucket) {
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
    if (bucket >= kMapBaseControlBuckets) {
        return self.extra_bucket_heads[bucket - kMapBaseControlBuckets];
    }
#endif
    return self.bucket_heads[bucket];
}

PTO_DEVICE_FUNC inline void
dist_private_tensor_map_store_head(__gm__ DistTensorMap &self, uint32_t bucket, uint64_t value) {
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
    if (bucket >= kMapBaseControlBuckets) {
        self.extra_bucket_heads[bucket - kMapBaseControlBuckets] = value;
        return;
    }
#endif
    self.bucket_heads[bucket] = value;
}

PTO_DEVICE_FUNC inline uint64_t
dist_private_tensor_map_load_tail(__gm__ const DistTensorMap &self, uint32_t bucket) {
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
    if (bucket >= kMapBaseControlBuckets) {
        return self.extra_bucket_tails[bucket - kMapBaseControlBuckets];
    }
#endif
    return self.bucket_tails[bucket];
}

PTO_DEVICE_FUNC inline void
dist_private_tensor_map_store_tail(__gm__ DistTensorMap &self, uint32_t bucket, uint64_t value) {
#if PTO_FDWIC_TENSORMAP_RING_CAP < 128
    if (bucket >= kMapBaseControlBuckets) {
        self.extra_bucket_tails[bucket - kMapBaseControlBuckets] = value;
        return;
    }
#endif
    self.bucket_tails[bucket] = value;
}

PTO_DEVICE_FUNC inline void dist_private_tensor_map_reset(__gm__ DistTensorMap &self) {
    self.alive_floor = 0;
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        dist_private_tensor_map_store_head(self, bucket, 0);
        dist_private_tensor_map_store_tail(self, bucket, 0);
    }
}

PTO_DEVICE_FUNC inline uint32_t dist_private_tensor_map_hash(uint64_t addr) {
#if PTO_FDWIC_TENSORMAP_RING_CAP == 16384
    (void)addr;
    return 0;
#else
    addr *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(addr >> (64U - kMapBucketShift)) & kMapBucketMask;
#endif
}

template <typename TensorRef>
PTO_DEVICE_FUNC inline void
dist_private_tensor_map_byte_range(const TensorRef &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
    const uint64_t esz = get_element_size(t.dtype);
    addr = t.buffer.addr;
    lo = t.start_offset * esz;
    uint64_t ext;
    if (t.is_contiguous) {
        ext = 1;
        for (uint32_t i = 0; i < t.ndims; i++)
            ext *= t.shapes[i];
    } else {
        ext = t.extent_elem_cache;
    }
    hi = (t.start_offset + ext) * esz;
}

PTO_DEVICE_FUNC inline uint32_t dist_private_tensor_map_slot_index(uint32_t bucket, uint64_t cursor) {
    return bucket * kMapBucketCapacity + (static_cast<uint32_t>(cursor) & kMapBucketSlotMask);
}

PTO_DEVICE_FUNC inline void dist_private_tensor_map_retire_bucket(
    __gm__ DistTensorMap &self, uint32_t bucket, uint64_t &head, uint64_t &tail
) {
    head = dist_private_tensor_map_load_head(self, bucket);
    tail = dist_private_tensor_map_load_tail(self, bucket);
    const uint64_t original_head = head;
    while (head < tail) {
        __gm__ const MapEntry &entry = self.entries[dist_private_tensor_map_slot_index(bucket, head)];
        if (entry.producer >= self.alive_floor) {
            break;
        }
        ++head;
    }
    if (head != original_head) {
        dist_private_tensor_map_store_head(self, bucket, head);
    }
}

PTO_DEVICE_FUNC inline void
dist_private_tensor_map_advance_retire(__gm__ DistTensorMap &self, int32_t task_id, int32_t history) {
    const int32_t new_floor = task_id - history;
    if (new_floor > self.alive_floor) {
        self.alive_floor = new_floor;
    }
}

template <typename TensorRef>
PTO_DEVICE_FUNC inline bool dist_private_tensor_map_insert(
    __gm__ DistTensorMap &self, const TensorRef &t, int32_t producer
) {
    uint64_t addr, lo, hi;
    dist_private_tensor_map_byte_range(t, addr, lo, hi);
    const uint32_t bucket = dist_private_tensor_map_hash(addr);
    uint64_t head, tail;
    dist_private_tensor_map_retire_bucket(self, bucket, head, tail);
    if (tail - head >= kMapBucketCapacity) {
        return false;
    }

    // private replay 的 task_id 来自单调递增的 DistCore::local_index，同一
    // task 的多个 OUTPUT 可以相等，因此同桶 producer 必须单调不降。
    // 桶头 lazy-retire 正是建立在这一合同上；CCEC 中 always_assert
    // 为零成本，CPU/A5sim 则在所有构建类型拒绝绕过 Submit 的逆序调用。
    always_assert(
        tail == head ||
        self.entries[dist_private_tensor_map_slot_index(bucket, tail - 1)].producer <= producer
    );
    __gm__ MapEntry &entry = self.entries[dist_private_tensor_map_slot_index(bucket, tail)];
    entry.buf_addr = addr;
    entry.lo = lo;
    entry.hi = hi;
    entry.producer = producer;
    dist_private_tensor_map_store_tail(self, bucket, tail + 1);
    return true;
}

template <typename TensorRef>
PTO_DEVICE_FUNC inline int32_t dist_private_tensor_map_lookup(__gm__ DistTensorMap &self, const TensorRef &t) {
    uint64_t addr, lo, hi;
    dist_private_tensor_map_byte_range(t, addr, lo, hi);
    const uint32_t bucket = dist_private_tensor_map_hash(addr);
    uint64_t head, tail;
    dist_private_tensor_map_retire_bucket(self, bucket, head, tail);
    int32_t best = -1;
    for (uint64_t cursor = head; cursor < tail; ++cursor) {
        __gm__ const MapEntry &entry = self.entries[dist_private_tensor_map_slot_index(bucket, cursor)];
        if (entry.producer < self.alive_floor) {
            continue;
        }
        if (entry.buf_addr == addr && lo < entry.hi && entry.lo < hi && entry.producer > best) {
            best = entry.producer;
        }
    }
    return best;
}

}  // namespace
