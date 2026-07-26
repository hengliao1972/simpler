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

enum class FdwicTensorMapMode : uint32_t {
    Private = 0,
    Shared = 1,
};

inline constexpr uint64_t kFdwicBuildIdentityMagic = 0x46445749434d4150ULL;  // "FDWICMAP"
inline constexpr uint32_t kFdwicBuildAbiVersion = 1;
inline constexpr uint32_t kFdwicDistGlobalLayoutVersion = 1;
inline constexpr FdwicTensorMapMode kFdwicCompiledTensorMapMode = static_cast<FdwicTensorMapMode>(PTO_FDWIC_SHARED_MAP);

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
    uint32_t reserved[9];
};

static_assert(sizeof(FdwicBuildIdentity) == 64, "FDWIC build identity must occupy exactly one cache line");
static_assert(alignof(FdwicBuildIdentity) == 64, "FDWIC build identity must be cache-line aligned");
static_assert(offsetof(FdwicBuildIdentity, error_bits) < 64, "FDWIC error bits must stay in the identity line");

inline FdwicBuildIdentity fdwic_make_build_identity(uint32_t runtime_bytes) {
    return {
        kFdwicBuildIdentityMagic,
        kFdwicBuildAbiVersion,
        static_cast<uint32_t>(kFdwicCompiledTensorMapMode),
        runtime_bytes,
        kFdwicDistGlobalLayoutVersion,
        FdwicBuildErrorNone,
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
           identity.runtime_bytes == expected_runtime_bytes &&
           identity.dist_global_layout_version == kFdwicDistGlobalLayoutVersion;
}
