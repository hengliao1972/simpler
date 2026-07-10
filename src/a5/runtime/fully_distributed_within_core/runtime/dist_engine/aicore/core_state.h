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

PTO_DEVICE_FUNC void dist_core_reset(__gm__ DistCore &self, CoreType r, int32_t block, int32_t lane_id) {
    self.role = r;
    self.block_id = block;
    self.lane = lane_id;
    self.sub_block_id = (lane_id == LANE_AIV1) ? 1 : 0;
    self.local_index = 0;
    self.heap_next = 0;
    dist_tensor_map_reset(self.map);
    self.occupied_count = 0;
    self.owned_total = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        self.slots[i].occupied = false;
        self.slots[i].built = false;
    }
}

}  // namespace
