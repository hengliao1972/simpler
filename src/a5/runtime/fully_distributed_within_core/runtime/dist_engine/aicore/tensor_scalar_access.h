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

PTO_DEVICE_FUNC uint64_t dist_read_tensor_scalar_raw(const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
    uint64_t result = 0;
    const uint64_t addr = tensor.buffer.addr + flat * esz;
    if (esz == 1) {
        result = *reinterpret_cast<__gm__ const uint8_t *>(addr);
    } else if (esz == 2) {
        result = *reinterpret_cast<__gm__ const uint16_t *>(addr);
    } else if (esz == 4) {
        result = *reinterpret_cast<__gm__ const uint32_t *>(addr);
    } else {
        result = *reinterpret_cast<__gm__ const uint64_t *>(addr);
    }
    return result;
}

PTO_DEVICE_FUNC void
dist_write_tensor_scalar_raw(const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value) {
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
    const uint64_t addr = tensor.buffer.addr + flat * esz;
    if (esz == 1) {
        *reinterpret_cast<__gm__ uint8_t *>(addr) = static_cast<uint8_t>(value);
    } else if (esz == 2) {
        *reinterpret_cast<__gm__ uint16_t *>(addr) = static_cast<uint16_t>(value);
    } else if (esz == 4) {
        *reinterpret_cast<__gm__ uint32_t *>(addr) = static_cast<uint32_t>(value);
    } else {
        *reinterpret_cast<__gm__ uint64_t *>(addr) = value;
    }
}

}  // namespace
