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

template <typename T>
PTO_DEVICE_FUNC inline T atom_load(const volatile T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_load(const T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(volatile T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_weak(volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/true, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_strong(volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/false, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_add(volatile T &p, T d, int mo) {
    return __atomic_fetch_add(&p, d, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_sub(volatile T &p, T d, int mo) {
    return __atomic_fetch_sub(&p, d, mo);
}
PTO_DEVICE_FUNC inline void atom_thread_fence(int mo) { __atomic_thread_fence(mo); }

#if defined(__CCE_AICORE__)
template <typename T>
PTO_DEVICE_FUNC inline T atom_load(__gm__ const volatile T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_load(__gm__ const T &p, int mo) {
    return __atomic_load_n(&p, mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(__gm__ volatile T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T, typename V>
PTO_DEVICE_FUNC inline void atom_store(__gm__ T &p, V v, int mo) {
    __atomic_store_n(&p, static_cast<T>(v), mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_weak(__gm__ volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/true, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline bool atom_cas_strong(__gm__ volatile T &p, T &expected, T desired, int s_mo, int f_mo) {
    return __atomic_compare_exchange_n(&p, &expected, desired, /*weak=*/false, s_mo, f_mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_add(__gm__ volatile T &p, T d, int mo) {
    return __atomic_fetch_add(&p, d, mo);
}
template <typename T>
PTO_DEVICE_FUNC inline T atom_fetch_sub(__gm__ volatile T &p, T d, int mo) {
    return __atomic_fetch_sub(&p, d, mo);
}
#endif

}  // namespace
