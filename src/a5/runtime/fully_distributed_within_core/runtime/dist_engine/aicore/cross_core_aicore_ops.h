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

#include "dist_engine/aicore/primitive.h"
#include "dist_engine/common/atomic.h"

namespace {

struct DistCrossCoreAicoreOps {
    PTO_DEVICE_FUNC static int64_t Load(__gm__ volatile int64_t *address) { return atomic_load(*address); }

    PTO_DEVICE_FUNC static int64_t
    CompareExchange(__gm__ volatile int64_t *address, int64_t expected, int64_t desired) {
        return atomic_compare_exchange(*address, expected, desired);
    }

    PTO_DEVICE_FUNC static int64_t FetchAdd(__gm__ volatile int64_t *address, int64_t delta) {
        return atomic_fetch_add(*address, delta);
    }

    PTO_DEVICE_FUNC static void StorePayloadWord(__gm__ volatile uint64_t *address, uint64_t value) {
        *address = value;
    }

    PTO_DEVICE_FUNC static void FlushRegion(__gm__ void *address, uint64_t bytes) {
        dist_aicore_flush_region(address, bytes);
    }

    PTO_DEVICE_FUNC static void InvalidateRegion(__gm__ const void *address, uint64_t bytes) {
        dist_aicore_invalidate_region(address, bytes);
    }
};

}  // namespace
