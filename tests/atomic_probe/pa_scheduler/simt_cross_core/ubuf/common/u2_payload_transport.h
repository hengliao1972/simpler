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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_U2_PAYLOAD_TRANSPORT_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_U2_PAYLOAD_TRANSPORT_H

#include "u2_full_pa.h"

namespace pa_scheduler::simt_cross_core::u2 {

#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtCopyPayloadWordsToGm(
    __ubuf__ volatile uint64_t *source, __gm__ uint64_t *destination, uint32_t written_words
) {
    for (uint32_t word = 0U; word < written_words; ++word) {
        destination[word] = source[word];
    }
}
#endif

}  // namespace pa_scheduler::simt_cross_core::u2

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_U2_PAYLOAD_TRANSPORT_H
