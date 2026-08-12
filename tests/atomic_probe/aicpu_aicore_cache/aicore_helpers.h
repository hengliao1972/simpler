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
#ifndef ATOMIC_PROBE_AICPU_AICORE_CACHE_AICORE_HELPERS_H_
#define ATOMIC_PROBE_AICPU_AICORE_CACHE_AICORE_HELPERS_H_

#include "cce_aicore_intrinsics.h"

#include "shared.h"

#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif

namespace aicpu_aicore_cache_probe {

struct LineClass {
    bool fresh;
    bool stale;
    uint64_t first;
};

struct WaitStats {
    uint64_t attempts;
    uint64_t ticks;
};

__aicore__ inline CaseResult NewResult() {
    CaseResult result{};
    result.magic = kResultMagic;
    result.status = static_cast<uint64_t>(Status::Ok);
    result.first_bad_round = kNoBadRound;
    return result;
}

__aicore__ inline void InvalidateConfig(__gm__ const ProbeConfig *config) {
    dcci(reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ ProbeConfig *>(config)), SINGLE_CACHE_LINE);
    dcci(
        reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ ProbeConfig *>(config)) + kCacheLineBytes,
        SINGLE_CACHE_LINE
    );
    dsb(DSB_ALL);
    __asm__ volatile("" ::: "memory");
}

__aicore__ inline bool WaitAtomicExact(
    __gm__ volatile int64_t *address, int64_t expected, uint64_t timeout_ticks, WaitStats *stats = nullptr
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint64_t attempts = 0U;
    while (true) {
        ++attempts;
        if (atomicAdd(const_cast<__gm__ int64_t *>(address), int64_t{0}) == expected) {
            if (stats != nullptr) {
                stats->attempts = attempts;
                stats->ticks = static_cast<uint64_t>(get_sys_cnt()) - begin;
            }
            return true;
        }
        if ((attempts & 1023U) == 0U && static_cast<uint64_t>(get_sys_cnt()) - begin > timeout_ticks) {
            if (stats != nullptr) {
                stats->attempts = attempts;
                stats->ticks = static_cast<uint64_t>(get_sys_cnt()) - begin;
            }
            return false;
        }
    }
}

__aicore__ inline void RecordWaitStats(CaseResult &result, const WaitStats &stats) {
    result.doorbell_poll_attempts_total += stats.attempts;
    if (stats.attempts > result.doorbell_poll_attempts_max) {
        result.doorbell_poll_attempts_max = stats.attempts;
    }
    result.doorbell_wait_ticks_total += stats.ticks;
    if (stats.ticks > result.doorbell_wait_ticks_max) {
        result.doorbell_wait_ticks_max = stats.ticks;
    }
}

__aicore__ inline uint64_t LoadDevice(__gm__ volatile uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(address), 0));
}

__aicore__ inline int64_t LoadDevice(__gm__ volatile int64_t *address) {
    return static_cast<int64_t>(
        __builtin_cce_ld_dev(reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(address)), 0)
    );
}

__aicore__ inline void ReadOrdinary(__gm__ const PayloadLine *payload, uint64_t values[kPayloadWords]) {
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        values[word] = payload->words[word];
    }
}

__aicore__ inline void ReadDevice(__gm__ PayloadLine *payload, uint64_t values[kPayloadWords]) {
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        values[word] = LoadDevice(&payload->words[word]);
    }
}

__aicore__ inline LineClass ClassifyLine(
    const uint64_t observed[kPayloadWords], const uint64_t primed[kPayloadWords], Direction direction,
    uint32_t case_index, uint32_t generation
) {
    bool fresh = true;
    bool stale = true;
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        fresh = fresh && observed[word] == PayloadValue(direction, case_index, generation, word);
        stale = stale && observed[word] == primed[word];
    }
    return {fresh, stale, observed[0]};
}

__aicore__ inline void CountValue(
    int64_t observed, int64_t fresh, int64_t stale, uint64_t &fresh_count, uint64_t &stale_count, uint64_t &other_count
) {
    if (observed == fresh) {
        ++fresh_count;
    } else if (observed == stale) {
        ++stale_count;
    } else {
        ++other_count;
    }
}

__aicore__ inline void
CountLine(const LineClass &classification, uint64_t &fresh_count, uint64_t &stale_count, uint64_t &other_count) {
    if (classification.fresh) {
        ++fresh_count;
    } else if (classification.stale) {
        ++stale_count;
    } else {
        ++other_count;
    }
}

__aicore__ inline void InvalidatePayload(__gm__ PayloadLine *payload, AicorePayloadAcquire acquire) {
    if (acquire == AicorePayloadAcquire::Ordinary || acquire == AicorePayloadAcquire::LdDev) {
        return;
    }
    if (acquire == AicorePayloadAcquire::DcciDefault) {
        dcci(payload, SINGLE_CACHE_LINE);
    } else if (acquire == AicorePayloadAcquire::DcciAll) {
        dcci(payload, SINGLE_CACHE_LINE, CACHELINE_ALL);
    } else if (acquire == AicorePayloadAcquire::DcciOut) {
        dcci(payload, SINGLE_CACHE_LINE, CACHELINE_OUT);
    } else {
        dcci(payload, SINGLE_CACHE_LINE, CACHELINE_ATOMIC);
    }
    dsb(DSB_ALL);
    __asm__ volatile("" ::: "memory");
}

__aicore__ inline void PublishResult(__gm__ CaseResult *destination, const CaseResult &source) {
    __asm__ volatile("" ::: "memory");
    auto *output = reinterpret_cast<__gm__ volatile uint64_t *>(destination);
    const auto *input = reinterpret_cast<const uint64_t *>(&source);
    for (uint32_t word = 0U; word < sizeof(CaseResult) / sizeof(uint64_t); ++word) {
        output[word] = input[word];
    }
    for (uint32_t line = 0U; line < sizeof(CaseResult) / kCacheLineBytes; ++line) {
        dcci(
            reinterpret_cast<__gm__ uint8_t *>(destination) + line * kCacheLineBytes, SINGLE_CACHE_LINE, CACHELINE_OUT
        );
    }
    dsb(DSB_ALL);
}

}  // namespace aicpu_aicore_cache_probe

#endif  // ATOMIC_PROBE_AICPU_AICORE_CACHE_AICORE_HELPERS_H_
