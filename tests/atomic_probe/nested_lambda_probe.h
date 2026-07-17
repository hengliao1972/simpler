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
#ifndef TESTS_ATOMIC_PROBE_NESTED_LAMBDA_PROBE_H
#define TESTS_ATOMIC_PROBE_NESTED_LAMBDA_PROBE_H

#include <cstdint>

namespace nested_lambda_probe {

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define NESTED_LAMBDA_DEVICE __aicore__
#else
#define NESTED_LAMBDA_DEVICE
#endif

enum class Field : uint32_t {
    OuterCalls = 0,
    InputCalls,
    OutputCalls,
    ScalarCalls,
    ReferenceState,
    InputValue,
    OutputValue,
    ScalarValue,
    CombinedValue,
    Count,
};

constexpr uint32_t kCacheLineWords = 16;
constexpr uint32_t kDefaultSeed = 0x120U;
constexpr uint32_t kStorageWords =
    static_cast<uint32_t>(Field::Count) * kCacheLineWords;

constexpr uint32_t FieldIndex(Field field)
{
    return static_cast<uint32_t>(field) * kCacheLineWords;
}

struct TensorLike {
    uint32_t value = 0;
};

struct BuildResult {
    TensorLike input;
    TensorLike output;
    uint32_t scalar = 0;
};

class SubmitBuilder {
public:
    template <typename Thunk>
    NESTED_LAMBDA_DEVICE inline void AddInput(Thunk thunk)
    {
        result_.input = thunk();
    }

    template <typename Thunk>
    NESTED_LAMBDA_DEVICE inline void AddOutput(Thunk thunk)
    {
        result_.output = thunk();
    }

    template <typename Thunk>
    NESTED_LAMBDA_DEVICE inline void AddScalar(Thunk thunk)
    {
        result_.scalar = thunk();
    }

    NESTED_LAMBDA_DEVICE inline BuildResult Finish() const { return result_; }

private:
    BuildResult result_{};
};

// Free-function template under test. Its callback is the outer lambda, which
// in turn passes nested lambdas to SubmitBuilder's member-function templates.
template <typename BuildCallback>
NESTED_LAMBDA_DEVICE inline BuildResult Submit(BuildCallback callback)
{
    SubmitBuilder builder;
    callback(builder);
    return builder.Finish();
}

constexpr uint32_t Expected(Field field, uint32_t seed)
{
    const uint32_t initial_reference = seed + 5U;
    const uint32_t outer_local = seed + 3U;
    const uint32_t input = outer_local + initial_reference + 7U;
    const uint32_t output = outer_local + 0x20U;
    const uint32_t scalar = seed + 0x31U;
    switch (field) {
    case Field::OuterCalls:
        return 1U;
    case Field::InputCalls:
        return 1U;
    case Field::OutputCalls:
        return 1U;
    case Field::ScalarCalls:
        return 1U;
    case Field::ReferenceState:
        return initial_reference + 7U;
    case Field::InputValue:
        return input;
    case Field::OutputValue:
        return output;
    case Field::ScalarValue:
        return scalar;
    case Field::CombinedValue:
        return input + output + scalar;
    case Field::Count:
        return 0U;
    }
    return 0U;
}

#undef NESTED_LAMBDA_DEVICE

} // namespace nested_lambda_probe

#endif // TESTS_ATOMIC_PROBE_NESTED_LAMBDA_PROBE_H
