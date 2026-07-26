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

#include <cstddef>
#include <cstdint>

// private/shared 是两套编译期产物，不是热路径上的运行期开关。默认值
// 保持现有 private 行为；构建系统必须显式给三类镜像传入同一个 0/1。
#ifndef PTO_FDWIC_SHARED_MAP
#define PTO_FDWIC_SHARED_MAP 0
#endif

#if PTO_FDWIC_SHARED_MAP != 0 && PTO_FDWIC_SHARED_MAP != 1
#error "PTO_FDWIC_SHARED_MAP must be 0 (private) or 1 (shared)"
#endif

#ifndef PTO_FDWIC_TENSORMAP_RING_CAP
#define PTO_FDWIC_TENSORMAP_RING_CAP 128
#endif

#if PTO_FDWIC_TENSORMAP_RING_CAP < 32 || PTO_FDWIC_TENSORMAP_RING_CAP > 16384
#error "PTO_FDWIC_TENSORMAP_RING_CAP must be in [32, 16384]"
#endif

#if (PTO_FDWIC_TENSORMAP_RING_CAP & (PTO_FDWIC_TENSORMAP_RING_CAP - 1)) != 0
#error "PTO_FDWIC_TENSORMAP_RING_CAP must be a power of two"
#endif

#if (16384 % PTO_FDWIC_TENSORMAP_RING_CAP) != 0
#error "PTO_FDWIC_TENSORMAP_RING_CAP must divide the fixed 16K physical slot pool"
#endif

enum class FdwicTensorMapMode : uint32_t {
    Private = 0,
    Shared = 1,
};

inline constexpr uint64_t kFdwicBuildIdentityMagic = 0x46445749434d4150ULL;  // "FDWICMAP"
inline constexpr uint32_t kFdwicBuildAbiVersion = 4;
inline constexpr uint32_t kFdwicDistGlobalLayoutVersion = 4;
inline constexpr FdwicTensorMapMode kFdwicCompiledTensorMapMode = static_cast<FdwicTensorMapMode>(PTO_FDWIC_SHARED_MAP);
inline constexpr uint32_t kFdwicTensorMapRingCap = PTO_FDWIC_TENSORMAP_RING_CAP;
inline constexpr uint32_t kFdwicTensorMapRingBuckets = 16384U / kFdwicTensorMapRingCap;

// 第一阶段只建立构建身份，尚未把 shared backend 接入真实调度路径。
// shared 镜像可以完整编译和参与 ABI 负向测试，但必须在零 Submit 前拒绝
// 执行，不能悄悄沿用 private TensorMap 语义。
inline constexpr bool kFdwicCompiledBackendReady = PTO_FDWIC_SHARED_MAP == 0;

enum FdwicBuildError : uint32_t {
    FdwicBuildErrorNone = 0,
    FdwicBuildErrorAicpuMismatch = 1U << 0,
    FdwicBuildErrorAicoreMismatch = 1U << 1,
    FdwicBuildErrorBackendUnavailable = 1U << 2,
};

// 该 cache line 必须保持为 Runtime 的首字段。Host、AICPU 和 AICore 在
// 解释任何模式相关状态之前先读取它；后续 shared 布局演进只能追加/修改
// 其后的状态，不能挪动这条稳定前缀。
struct alignas(64) FdwicBuildIdentity {
    uint64_t magic;
    uint32_t abi_version;
    uint32_t tensor_map_mode;
    uint32_t runtime_bytes;
    uint32_t dist_global_layout_version;
    volatile uint32_t error_bits;
    uint32_t tensor_map_ring_cap;
    uint32_t reserved[8];
};

static_assert(sizeof(FdwicBuildIdentity) == 64, "FDWIC build identity must occupy exactly one cache line");
static_assert(alignof(FdwicBuildIdentity) == 64, "FDWIC build identity must be cache-line aligned");
// 前三个控制字段已经被 v1 Host/AICPU/AICore 共同使用。新增身份字段只能
// 消耗旧 reserved，不能移动错误位；否则新旧 AICore 混件时，失败方会把
// mismatch 写到另一镜像看不到的偏移。
static_assert(offsetof(FdwicBuildIdentity, runtime_bytes) == 16, "FDWIC runtime-size identity offset changed");
static_assert(
    offsetof(FdwicBuildIdentity, dist_global_layout_version) == 20, "FDWIC dist-layout identity offset changed");
static_assert(offsetof(FdwicBuildIdentity, error_bits) == 24, "FDWIC cross-image error-bit offset changed");
static_assert(offsetof(FdwicBuildIdentity, tensor_map_ring_cap) == 28, "FDWIC ring-cap identity offset changed");

inline FdwicBuildIdentity fdwic_make_build_identity(uint32_t runtime_bytes) {
    return {
        kFdwicBuildIdentityMagic,
        kFdwicBuildAbiVersion,
        static_cast<uint32_t>(kFdwicCompiledTensorMapMode),
        runtime_bytes,
        kFdwicDistGlobalLayoutVersion,
        FdwicBuildErrorNone,
        kFdwicTensorMapRingCap,
        {},
    };
}

#if defined(__CCE_AICORE__)
__aicore__ inline bool
fdwic_build_identity_matches(__gm__ const volatile FdwicBuildIdentity &identity, uint32_t expected_runtime_bytes) {
#else
inline bool fdwic_build_identity_matches(const volatile FdwicBuildIdentity &identity, uint32_t expected_runtime_bytes) {
#endif
    return identity.magic == kFdwicBuildIdentityMagic && identity.abi_version == kFdwicBuildAbiVersion &&
           identity.tensor_map_mode == static_cast<uint32_t>(kFdwicCompiledTensorMapMode) &&
           identity.tensor_map_ring_cap == kFdwicTensorMapRingCap &&
           identity.runtime_bytes == expected_runtime_bytes &&
           identity.dist_global_layout_version == kFdwicDistGlobalLayoutVersion;
}
