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

#ifndef TESTS_ATOMIC_PROBE_CCEC_SCALAR_PMU_DEVICE_H_
#define TESTS_ATOMIC_PROBE_CCEC_SCALAR_PMU_DEVICE_H_

#include <cstdint>

namespace atomic_probe::scalar_pmu {

constexpr uint32_t kPhysicalSubcores = 108U;
constexpr uint64_t kCounterBlockOffset = 0x4200ULL;
constexpr uint64_t kSelectorBlockOffset = 0x2400ULL;

constexpr uint64_t kSelectorVector = 1ULL << 0;
constexpr uint64_t kSelectorCube = 1ULL << 1;
constexpr uint64_t kSelectorScalar = 1ULL << 2;
constexpr uint64_t kSelectorMte1 = 1ULL << 3;
constexpr uint64_t kSelectorMte2 = 1ULL << 4;
constexpr uint64_t kSelectorMte3 = 1ULL << 5;
constexpr uint64_t kSelectorIcacheRequest = 1ULL << 6;
constexpr uint64_t kSelectorIcacheMiss = 1ULL << 7;
constexpr uint64_t kSelectorFix = 1ULL << 8;

struct Snapshot {
    uint64_t total;
    uint64_t vector_busy;
    uint64_t cube_busy;
    uint64_t scalar_busy;
    uint64_t mte1_busy;
    uint64_t mte2_busy;
    uint64_t mte3_busy;
    uint64_t icache_request;
    uint64_t icache_miss;
    uint64_t fix_busy;
};

__aicore__ __attribute__((always_inline)) inline int32_t *CounterBase(
    uint64_t register_base
) {
    return reinterpret_cast<int32_t *>(
        register_base + kCounterBlockOffset
    );
}

__aicore__ __attribute__((always_inline)) inline int32_t *SelectorBase(
    uint64_t register_base
) {
    return reinterpret_cast<int32_t *>(
        register_base + kSelectorBlockOffset
    );
}

__aicore__ __attribute__((always_inline)) inline void ClearCounters(
    uint64_t register_base
) {
    int32_t *base = CounterBase(register_base);
    (void)ld_dev(base, 0x10);
    (void)ld_dev(base, 0x18);
    (void)ld_dev(base, 0x20);
    (void)ld_dev(base, 0x28);
    (void)ld_dev(base, 0x30);
    (void)ld_dev(base, 0x38);
    (void)ld_dev(base, 0x40);
    (void)ld_dev(base, 0x48);
    (void)ld_dev(base, 0x50);
    (void)ld_dev(base, 0x54);
    (void)ld_dev(base, 0x60);
    (void)ld_dev(base, 0x64);
}

template <int16_t Offset>
__aicore__ __attribute__((always_inline)) inline uint64_t ReadCounter(
    uint64_t register_base
) {
    return static_cast<uint32_t>(
        ld_dev(CounterBase(register_base), Offset)
    );
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadTotal(
    uint64_t register_base
) {
    int32_t *base = CounterBase(register_base);
    const uint64_t low = static_cast<uint32_t>(ld_dev(base, 0x60));
    const uint64_t high = static_cast<uint32_t>(ld_dev(base, 0x64));
    return low | (high << 32U);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadSelectorStatus(
    uint64_t register_base
) {
    int32_t *base = SelectorBase(register_base);
    uint64_t status = 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x100)) == 0x501U
        ? kSelectorVector : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x104)) == 0x301U
        ? kSelectorCube : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x108)) == 0x001U
        ? kSelectorScalar : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x10c)) == 0x701U
        ? kSelectorMte1 : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x110)) == 0x202U
        ? kSelectorMte2 : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x114)) == 0x203U
        ? kSelectorMte3 : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x118)) == 0x034U
        ? kSelectorIcacheRequest : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x11c)) == 0x035U
        ? kSelectorIcacheMiss : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x120)) == 0x714U
        ? kSelectorFix : 0U;
    return status;
}

__aicore__ __attribute__((always_inline)) inline Snapshot ReadSnapshot(
    uint64_t register_base
) {
    Snapshot snapshot{};
    snapshot.vector_busy = ReadCounter<0x10>(register_base);
    snapshot.cube_busy = ReadCounter<0x18>(register_base);
    snapshot.scalar_busy = ReadCounter<0x20>(register_base);
    snapshot.mte1_busy = ReadCounter<0x28>(register_base);
    snapshot.mte2_busy = ReadCounter<0x30>(register_base);
    snapshot.mte3_busy = ReadCounter<0x38>(register_base);
    snapshot.icache_request = ReadCounter<0x40>(register_base);
    snapshot.icache_miss = ReadCounter<0x48>(register_base);
    snapshot.fix_busy = ReadCounter<0x50>(register_base);
    snapshot.total = ReadTotal(register_base);
    return snapshot;
}

__aicore__ __attribute__((always_inline)) inline void Publish64(
    __gm__ uint64_t *address, uint64_t value
) {
    __builtin_cce_st_dev(value, address, 0);
}

}  // namespace atomic_probe::scalar_pmu

#endif  // TESTS_ATOMIC_PROBE_CCEC_SCALAR_PMU_DEVICE_H_
