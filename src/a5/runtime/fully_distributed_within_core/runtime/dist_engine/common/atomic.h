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

#include "dist_engine/common/target.h"
#include "intrinsic.h"
#include "pto_types.h"

namespace {

PTO_DEVICE_FUNC inline void store_barrier() {
#if defined(__CCE_AICORE__)
    OUT_OF_ORDER_STORE_BARRIER();
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

template <typename T>
PTO_DEVICE_FUNC inline T atomic_load(__gm__ volatile T &value, int memorder = __ATOMIC_ACQUIRE) {
#if defined(__CCE_AICORE__)
    (void)memorder;
    __gm__ T *addr = const_cast<__gm__ T *>(&value);
    return atomicAdd(addr, static_cast<T>(0));
#else
    return __atomic_load_n(&value, memorder);
#endif
}

template <typename T, typename V>
PTO_DEVICE_FUNC inline T atomic_exchange(__gm__ volatile T &value, V desired, int memorder = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)memorder;
    __gm__ T *addr = const_cast<__gm__ T *>(&value);
    return atomicExch(addr, static_cast<T>(desired));
#else
    return __atomic_exchange_n(&value, static_cast<T>(desired), memorder);
#endif
}

template <typename T>
PTO_DEVICE_FUNC inline T atomic_fetch_add(__gm__ volatile T &value, T delta, int memorder = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)memorder;
    __gm__ T *addr = const_cast<__gm__ T *>(&value);
    return atomicAdd(addr, delta);
#else
    return __atomic_fetch_add(&value, delta, memorder);
#endif
}

template <typename T>
PTO_DEVICE_FUNC inline T atomic_fetch_sub(__gm__ volatile T &value, T delta, int memorder = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)memorder;
    __gm__ T *addr = const_cast<__gm__ T *>(&value);
    return atomicSub(addr, delta);
#else
    return __atomic_fetch_sub(&value, delta, memorder);
#endif
}

template <typename T>
PTO_DEVICE_FUNC inline T atomic_fetch_max(__gm__ volatile T &value, T desired, int memorder = __ATOMIC_ACQ_REL) {
#if defined(__CCE_AICORE__)
    (void)memorder;
    __gm__ T *addr = const_cast<__gm__ T *>(&value);
    return atomicMax(addr, desired);
#else
    T cur = __atomic_load_n(&value, __ATOMIC_ACQUIRE);
    while (desired > cur) {
        if (__atomic_compare_exchange_n(&value, &cur, desired, /*weak=*/true, memorder, __ATOMIC_ACQUIRE)) return cur;
    }
    return cur;
#endif
}

// Compare `value` with expected and replace it with desired on equality.
// Returns the value observed before the operation, matching A5 atomicCAS.
template <typename T>
PTO_DEVICE_FUNC inline T atomic_compare_exchange(
    __gm__ volatile T &value, T expected, T desired, int success_memorder = __ATOMIC_ACQ_REL,
    int failure_memorder = __ATOMIC_ACQUIRE
) {
#if defined(__CCE_AICORE__)
    (void)success_memorder;
    (void)failure_memorder;
    __gm__ T *addr = const_cast<__gm__ T *>(&value);
    return atomicCAS(addr, expected, desired);
#else
    T observed = expected;
    (void)__atomic_compare_exchange_n(
        &value, &observed, desired, /*weak=*/false, success_memorder, failure_memorder
    );
    return observed;
#endif
}

}  // namespace
