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

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include "../common/u2_payload_transport.h"

namespace {

namespace u2 = pa_scheduler::simt_cross_core::u2;

constexpr uint32_t kProbeThreadCount = 32U;
constexpr uint32_t kProbeMaxWords = pa_scheduler::simt_cross_core::ubuf_staging::kMaxPayloadWords;

static __simt_vf__ __aicore__ LAUNCH_BOUND(kProbeThreadCount
) void U2PayloadTransportIrProbe(__gm__ uint64_t *destination, __ubuf__ volatile uint64_t *staging, uint64_t nonce) {
    if (static_cast<uint32_t>(threadIdx.x) != 0U) {
        return;
    }
    const uint32_t requested_words = static_cast<uint32_t>(destination[kProbeMaxWords]);
    const uint32_t written_words =
        requested_words == 0U || requested_words > kProbeMaxWords ? kProbeMaxWords : requested_words;
    for (uint32_t word = 0U; word < written_words; ++word) {
        staging[word] = nonce ^ (static_cast<uint64_t>(word) << 32U) ^ word;
    }
    u2::SimtCopyPayloadWordsToGm(staging, destination, written_words);
    asc_threadfence();
}

}  // namespace

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_u2_transport_ir_probe, 1, 2);

extern "C" __global__ __aicore__ void simt_cross_core_u2_transport_ir_probe(__gm__ uint64_t *destination) {
    __ubuf__ volatile uint64_t *staging = reinterpret_cast<__ubuf__ volatile uint64_t *>(0U);
    cce::async_invoke<U2PayloadTransportIrProbe>(
        cce::dim3{kProbeThreadCount, 1U, 1U}, destination, staging, UINT64_C(0xA5027A6E5F000001)
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
