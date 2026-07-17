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
#ifndef TESTS_ATOMIC_PROBE_CCEC_NESTED_LAMBDA_CROSS_TU_LAYOUT_H
#define TESTS_ATOMIC_PROBE_CCEC_NESTED_LAMBDA_CROSS_TU_LAYOUT_H

#include <cstdint>

namespace nested_lambda_cross_tu_probe {

#if defined(__CCE_AICORE__)
#define CROSS_TU_DEVICE __aicore__
#else
#define CROSS_TU_DEVICE
#endif

enum class Variant : uint32_t {
    WeakContextMaterialize0 = 0,
    WeakContextMaterialize1,
    WeakContextMaterialize2,
    WeakContextMaterialize3,
    WeakArgsStorage,
    StrongContext,
    ArgsRuntimeRead,
    Count,
};

enum class Field : uint32_t {
    CompletedRounds = 0,
    MismatchCount,
    DispatcherCalls,
    AddressMaterializations,
    ChecksumLow,
    ChecksumHigh,
    L0TaskArgsBytes,
    VariantEcho,
    Count,
};

constexpr uint32_t kRounds = 64;
constexpr uint32_t kSubmitsPerRound = 4;
constexpr uint32_t kExpectedL0TaskArgsBytes = 1024;
constexpr uint32_t kCacheLineWords = 16;
constexpr uint32_t kStorageWords = static_cast<uint32_t>(Field::Count) * kCacheLineWords;
constexpr uint64_t kControlXor = 0x6A09E667F3BCC909ULL;

CROSS_TU_DEVICE constexpr uint32_t FieldIndex(Field field) { return static_cast<uint32_t>(field) * kCacheLineWords; }

CROSS_TU_DEVICE constexpr uint64_t TensorAddress(uint32_t round, uint32_t tensor_index) {
    return 0x100000000ULL + static_cast<uint64_t>(round) * 0x1000ULL + static_cast<uint64_t>(tensor_index) * 0x100ULL +
           0x55ULL;
}

CROSS_TU_DEVICE constexpr uint64_t TensorOffset(uint32_t round, uint32_t tensor_index) {
    return static_cast<uint64_t>(round) * 8ULL + tensor_index;
}

CROSS_TU_DEVICE constexpr int32_t TensorVersion(uint32_t round, uint32_t tensor_index) {
    return static_cast<int32_t>(100U + round * 3U + tensor_index);
}

CROSS_TU_DEVICE constexpr uint32_t TensorShape(uint32_t round, uint32_t tensor_index) {
    return 17U + round + tensor_index;
}

CROSS_TU_DEVICE constexpr uint64_t ContextSalt(uint32_t round) { return 0xBADC000000000000ULL + round; }

CROSS_TU_DEVICE constexpr uint64_t TensorDigest(uint32_t round, uint32_t tensor_index) {
    return TensorAddress(round, tensor_index) + TensorOffset(round, tensor_index) * 17ULL +
           static_cast<uint64_t>(static_cast<uint32_t>(TensorVersion(round, tensor_index))) * 257ULL +
           TensorShape(round, tensor_index) * 65537ULL;
}

CROSS_TU_DEVICE constexpr uint64_t ExpectedLazyDigest(uint32_t round) {
    return TensorDigest(round, 0) * 3ULL + TensorDigest(round, 1) * 5ULL + TensorDigest(round, 2) * 7ULL +
           ContextSalt(round);
}

CROSS_TU_DEVICE constexpr uint64_t ControlInput(uint32_t round, uint32_t submit_index) {
    return 0xC000000000000000ULL + static_cast<uint64_t>(round) * kSubmitsPerRound + submit_index;
}

CROSS_TU_DEVICE constexpr uint64_t ExpectedControlResult(uint32_t round, uint32_t submit_index) {
    return ControlInput(round, submit_index) ^ kControlXor;
}

CROSS_TU_DEVICE constexpr uint64_t ExpectedTotalChecksum() {
    uint64_t checksum = 0;
    for (uint32_t round = 0; round < kRounds; round++) {
        checksum += ExpectedLazyDigest(round);
        for (uint32_t submit = 1; submit < kSubmitsPerRound; submit++) {
            checksum += ExpectedControlResult(round, submit);
        }
    }
    return checksum;
}

constexpr uint32_t ExpectedMaterializations(Variant variant) {
    switch (variant) {
    case Variant::WeakContextMaterialize0:
    case Variant::StrongContext:
        return 0;
    case Variant::WeakContextMaterialize1:
        return 1;
    case Variant::WeakContextMaterialize2:
        return 2;
    case Variant::WeakContextMaterialize3:
    case Variant::WeakArgsStorage:
    case Variant::ArgsRuntimeRead:
        return 3;
    case Variant::Count:
        return 0;
    }
    return 0;
}

constexpr const char *VariantName(Variant variant) {
    switch (variant) {
    case Variant::WeakContextMaterialize0:
        return "weak-context-materialize-0";
    case Variant::WeakContextMaterialize1:
        return "weak-context-materialize-1";
    case Variant::WeakContextMaterialize2:
        return "weak-context-materialize-2";
    case Variant::WeakContextMaterialize3:
        return "weak-context-materialize-3";
    case Variant::WeakArgsStorage:
        return "weak-args-storage";
    case Variant::StrongContext:
        return "strong-context";
    case Variant::ArgsRuntimeRead:
        return "args-runtime-read";
    case Variant::Count:
        return "invalid";
    }
    return "invalid";
}

constexpr const char *KernelName(Variant variant) {
    switch (variant) {
    case Variant::WeakContextMaterialize0:
        return "nested_lambda_cross_tu_ctx_m0_0_mix_aic";
    case Variant::WeakContextMaterialize1:
        return "nested_lambda_cross_tu_ctx_m1_1_mix_aic";
    case Variant::WeakContextMaterialize2:
        return "nested_lambda_cross_tu_ctx_m2_2_mix_aic";
    case Variant::WeakContextMaterialize3:
        return "nested_lambda_cross_tu_ctx_m3_3_mix_aic";
    case Variant::WeakArgsStorage:
        return "nested_lambda_cross_tu_args_4_mix_aic";
    case Variant::StrongContext:
        return "nested_lambda_cross_tu_strong_5_mix_aic";
    case Variant::ArgsRuntimeRead:
        return "nested_lambda_cross_tu_runtime_args_6_mix_aic";
    case Variant::Count:
        return "";
    }
    return "";
}

// CANN defines funcEntry as the numeric suffix in kernel_foo_<entry>.
// Keep entries unique inside this multi-kernel raw ELF.
constexpr uint64_t KernelEntry(Variant variant) { return static_cast<uint64_t>(variant); }

constexpr uint32_t ExpectedDispatcherCalls(Variant variant) {
    return variant == Variant::ArgsRuntimeRead ? 0 : kRounds * 2;
}

#undef CROSS_TU_DEVICE

}  // namespace nested_lambda_cross_tu_probe

#endif  // TESTS_ATOMIC_PROBE_CCEC_NESTED_LAMBDA_CROSS_TU_LAYOUT_H
