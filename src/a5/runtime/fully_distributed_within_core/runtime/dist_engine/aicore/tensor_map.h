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

PTO_DEVICE_FUNC void dist_tensor_map_reset(__gm__ DistTensorMap &self) {
    self.free_head = -1;
    self.high_water = 0;
    self.alive_floor = 0;
    self.cleaned_upto = 0;
    for (int32_t i = 0; i < kMapBuckets; i++)
        self.buckets[i] = -1;
    for (int32_t i = 0; i < kTaskWindow; i++)
        self.task_heads[i] = -1;
}

PTO_DEVICE_FUNC uint32_t dist_tensor_map_hash(uint64_t addr) {
    addr *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(addr >> (64 - kMapBucketShift));
}

template <typename TensorRef>
PTO_DEVICE_FUNC void dist_tensor_map_byte_range(const TensorRef &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
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

PTO_DEVICE_FUNC int32_t dist_tensor_map_alloc_slot(__gm__ DistTensorMap &self) {
    if (self.free_head >= 0) {
        const int32_t s = self.free_head;
        self.free_head = self.entries[s].next_in_bucket;
        return s;
    }
    if (self.high_water < kMapCap) return self.high_water++;
    return -1;
}

PTO_DEVICE_FUNC void dist_tensor_map_free_entry(__gm__ DistTensorMap &self, int32_t idx) {
    __gm__ MapEntry &e = self.entries[idx];
    if (e.prev_in_bucket < 0) self.buckets[e.bucket] = e.next_in_bucket;
    else self.entries[e.prev_in_bucket].next_in_bucket = e.next_in_bucket;
    if (e.next_in_bucket >= 0) self.entries[e.next_in_bucket].prev_in_bucket = e.prev_in_bucket;
    e.bucket = -1;
    e.next_in_bucket = self.free_head;
    self.free_head = idx;
}

PTO_DEVICE_FUNC void dist_tensor_map_advance_retire(__gm__ DistTensorMap &self, int32_t N, int32_t H) {
    const int32_t new_floor = N - H;
    if (new_floor <= self.cleaned_upto) {
        if (new_floor > self.alive_floor) self.alive_floor = new_floor;
        return;
    }
    for (int32_t id = self.cleaned_upto; id < new_floor; id++) {
        int32_t cur = self.task_heads[id & kTaskWindowMask];
        // Empty heads are already normalized; avoid writing -1 back to GM.
        if (cur == -1) continue;
        while (cur >= 0) {
            const int32_t nxt = self.entries[cur].next_in_task;
            debug_assert(self.entries[cur].producer == id);
            dist_tensor_map_free_entry(self, cur);
            cur = nxt;
        }
        self.task_heads[id & kTaskWindowMask] = -1;
    }
    self.cleaned_upto = new_floor;
    self.alive_floor = new_floor;
}

template <typename TensorRef>
PTO_DEVICE_FUNC void dist_tensor_map_insert(__gm__ DistTensorMap &self, const TensorRef &t, int32_t producer) {
    uint64_t addr, lo, hi;
    dist_tensor_map_byte_range(t, addr, lo, hi);
    const int32_t s = dist_tensor_map_alloc_slot(self);
    if (s < 0) return;
    const uint32_t b = dist_tensor_map_hash(addr);
    __gm__ MapEntry &e = self.entries[s];
    e.buf_addr = addr;
    e.lo = lo;
    e.hi = hi;
    e.producer = producer;
    e.bucket = static_cast<int32_t>(b);
    e.prev_in_bucket = -1;
    e.next_in_bucket = self.buckets[b];
    if (self.buckets[b] >= 0) self.entries[self.buckets[b]].prev_in_bucket = s;
    self.buckets[b] = s;
    const int32_t slot = producer & kTaskWindowMask;
    e.next_in_task = self.task_heads[slot];
    self.task_heads[slot] = s;
}

template <typename TensorRef>
PTO_DEVICE_FUNC int32_t dist_tensor_map_lookup(__gm__ const DistTensorMap &self, const TensorRef &t) {
    uint64_t addr, lo, hi;
    dist_tensor_map_byte_range(t, addr, lo, hi);
    int32_t best = -1;
    for (int32_t cur = self.buckets[dist_tensor_map_hash(addr)]; cur >= 0; cur = self.entries[cur].next_in_bucket) {
        __gm__ const MapEntry &e = self.entries[cur];
        if (e.producer < self.alive_floor) continue;
        if (e.buf_addr == addr && lo < e.hi && e.lo < hi) {
            if (e.producer > best) best = e.producer;
        }
    }
    return best;
}

}  // namespace
