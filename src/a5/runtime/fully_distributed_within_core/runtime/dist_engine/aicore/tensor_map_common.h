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

// private/shared 只在副本所有权和并发发布纪律上分叉；hash、Tensor 的逻辑
// byte range、连续分桶下标与半开区间重叠必须使用同一份实现，才能保证两种
// backend 生成的依赖具有可比性。
PTO_DEVICE_FUNC inline uint32_t dist_tensor_map_hash(uint64_t addr) {
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
dist_tensor_map_byte_range(const TensorRef &tensor, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
    const uint64_t element_bytes = get_element_size(tensor.dtype);
    addr = tensor.buffer.addr;
    lo = tensor.start_offset * element_bytes;
    uint64_t extent;
    if (tensor.is_contiguous) {
        extent = 1;
        for (uint32_t dimension = 0; dimension < tensor.ndims; ++dimension) {
            extent *= tensor.shapes[dimension];
        }
    } else {
        extent = tensor.extent_elem_cache;
    }
    hi = (tensor.start_offset + extent) * element_bytes;
}

PTO_DEVICE_FUNC inline uint32_t dist_tensor_map_slot_index(uint32_t bucket, uint64_t cursor) {
    return bucket * kMapBucketCapacity + (static_cast<uint32_t>(cursor) & kMapBucketSlotMask);
}

template <typename RegionValue>
PTO_DEVICE_FUNC inline bool dist_tensor_map_regions_overlap(
    const RegionValue &left, uint64_t right_addr, uint64_t right_lo, uint64_t right_hi
) {
    return left.buf_addr == right_addr && right_lo < left.hi && left.lo < right_hi;
}

}  // namespace
