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

#include <gtest/gtest.h>

#include "runtime.h"

namespace {

constexpr uint32_t kRuntimeBytesForTest = sizeof(Runtime);

// private/shared 三个镜像都要先按这段公共前缀完成握手，才能读取其后的
// mode-specific 状态。这里直接约束生产 Runtime，而不是只用一个假的
// runtime_bytes 数值测试 identity header。
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
TEST(FdwicBuildIdentity, RuntimeKeepsTheStableThreeImageControlPrefix) {
    EXPECT_EQ(offsetof(Runtime, fdwic_build_identity), 0U);
    EXPECT_EQ(offsetof(Runtime, workers), 64U);
    EXPECT_EQ(
        offsetof(Runtime, worker_count),
        offsetof(Runtime, workers) + sizeof(Handshake) * RUNTIME_MAX_WORKER);
    EXPECT_EQ(offsetof(Runtime, aicpu_thread_num), offsetof(Runtime, worker_count) + sizeof(int));
    EXPECT_EQ(offsetof(Runtime, aicpu_allowed_cpus), offsetof(Runtime, aicpu_thread_num) + sizeof(int));
    EXPECT_EQ(
        offsetof(Runtime, aicpu_allowed_cpu_count),
        offsetof(Runtime, aicpu_allowed_cpus) + sizeof(int32_t) * 16);
    EXPECT_EQ(
        offsetof(Runtime, aicpu_launch_count),
        offsetof(Runtime, aicpu_allowed_cpu_count) + sizeof(int32_t));
    EXPECT_EQ(sizeof(Runtime), 70080U);
}
#pragma GCC diagnostic pop

TEST(FdwicBuildIdentity, CompiledModeBuildsAMatchingStableLine) {
    FdwicBuildIdentity identity = fdwic_make_build_identity(kRuntimeBytesForTest);

    EXPECT_EQ(sizeof(identity), 64U);
    EXPECT_EQ(offsetof(FdwicBuildIdentity, runtime_bytes), 16U);
    EXPECT_EQ(offsetof(FdwicBuildIdentity, dist_global_layout_version), 20U);
    EXPECT_EQ(offsetof(FdwicBuildIdentity, error_bits), 24U);
    EXPECT_EQ(offsetof(FdwicBuildIdentity, tensor_map_ring_cap), 28U);
    EXPECT_EQ(identity.tensor_map_mode, static_cast<uint32_t>(kFdwicCompiledTensorMapMode));
    EXPECT_EQ(identity.tensor_map_ring_cap, kFdwicTensorMapRingCap);
    EXPECT_EQ(kFdwicTensorMapRingBuckets * kFdwicTensorMapRingCap, 16384U);
    EXPECT_TRUE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest));
#if PTO_FDWIC_SHARED_MAP
    EXPECT_FALSE(kFdwicCompiledBackendReady);
#else
    EXPECT_TRUE(kFdwicCompiledBackendReady);
#endif
}

TEST(FdwicBuildIdentity, RejectsEveryCrossImageContractField) {
    FdwicBuildIdentity identity = fdwic_make_build_identity(kRuntimeBytesForTest);

    identity.magic ^= 1;
    EXPECT_FALSE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest));

    identity = fdwic_make_build_identity(kRuntimeBytesForTest);
    identity.abi_version += 1;
    EXPECT_FALSE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest));

    identity = fdwic_make_build_identity(kRuntimeBytesForTest);
    identity.tensor_map_mode ^= 1;
    EXPECT_FALSE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest));

    identity = fdwic_make_build_identity(kRuntimeBytesForTest);
    identity.tensor_map_ring_cap *= 2;
    EXPECT_FALSE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest));

    identity = fdwic_make_build_identity(kRuntimeBytesForTest);
    EXPECT_FALSE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest + 64));

    identity = fdwic_make_build_identity(kRuntimeBytesForTest);
    identity.dist_global_layout_version += 1;
    EXPECT_FALSE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest));
}

TEST(FdwicBuildIdentity, DiagnosticErrorBitsDoNotChangeCompatibility) {
    FdwicBuildIdentity identity = fdwic_make_build_identity(kRuntimeBytesForTest);
    identity.error_bits = FdwicBuildErrorAicoreMismatch | FdwicBuildErrorBackendUnavailable;

    EXPECT_TRUE(fdwic_build_identity_matches(identity, kRuntimeBytesForTest));
}

}  // namespace
