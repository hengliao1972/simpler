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

PTO_DEVICE_FUNC bool dist_cross_core_is_simt_builder_worker(__gm__ const DistCore *self) {
#if PTO_FDWIC_SCHEDULER_MODE == 3 || PTO_FDWIC_SCHEDULER_MODE == 4
    if (self == nullptr || self->role != CoreType::AIV || self->block_id < 0 || self->lane != LANE_AIV0) return false;
#if PTO_FDWIC_SCHEDULER_MODE == 3
    // Ordinary lookup is already cheap; keep all but block0/AIV0 available for
    // replay and execution. DAG lookup is scan-heavy and uses every AIV0 to
    // spread the independent dynamic metadata work.
    return self->block_id == 0;
#else
    return true;
#endif
#else
    (void)self;
    return false;
#endif
}

}  // namespace
