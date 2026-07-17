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
#include "../nested_lambda_probe.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

using nested_lambda_probe::Field;

class Result {
public:
    void Expect(bool condition, const char *label)
    {
        std::printf("[ASSERT] %-52s %s\n", label, condition ? "PASS" : "FAIL");
        if (!condition) failures_++;
    }

    int ExitCode() const
    {
        std::printf("[SUMMARY] semantic_failures=%d\n", failures_);
        return failures_ == 0 ? 0 : 1;
    }

private:
    int failures_ = 0;
};

void RunProbe(std::array<uint32_t, nested_lambda_probe::kStorageWords> &storage, uint32_t seed)
{
    uint32_t outer_calls = 0;
    uint32_t input_calls = 0;
    uint32_t output_calls = 0;
    uint32_t scalar_calls = 0;
    uint32_t reference_state = seed + 5U;

    // Outer reference-capturing lambda passed to a function template. It
    // creates three nested lambdas passed to member-function templates.
    const nested_lambda_probe::BuildResult built = nested_lambda_probe::Submit(
        [&](nested_lambda_probe::SubmitBuilder &builder) {
            outer_calls++;
            const uint32_t outer_local = seed + 3U;
            const uint32_t scalar_capture = seed + 0x30U;
            builder.AddInput([&]() -> nested_lambda_probe::TensorLike {
                input_calls++;
                reference_state += 7U;
                return {outer_local + reference_state};
            });
            builder.AddOutput([outer_local, &output_calls]() -> nested_lambda_probe::TensorLike {
                output_calls++;
                return {outer_local + 0x20U};
            });
            builder.AddScalar([scalar_capture, &scalar_calls]() -> uint32_t {
                scalar_calls++;
                return scalar_capture + 1U;
            });
        });

    storage[nested_lambda_probe::FieldIndex(Field::OuterCalls)] = outer_calls;
    storage[nested_lambda_probe::FieldIndex(Field::InputCalls)] = input_calls;
    storage[nested_lambda_probe::FieldIndex(Field::OutputCalls)] = output_calls;
    storage[nested_lambda_probe::FieldIndex(Field::ScalarCalls)] = scalar_calls;
    storage[nested_lambda_probe::FieldIndex(Field::ReferenceState)] = reference_state;
    storage[nested_lambda_probe::FieldIndex(Field::InputValue)] = built.input.value;
    storage[nested_lambda_probe::FieldIndex(Field::OutputValue)] = built.output.value;
    storage[nested_lambda_probe::FieldIndex(Field::ScalarValue)] = built.scalar;
    storage[nested_lambda_probe::FieldIndex(Field::CombinedValue)] =
        built.input.value + built.output.value + built.scalar;
}

} // namespace

int main()
{
    constexpr uint32_t seed = nested_lambda_probe::kDefaultSeed;
    std::array<uint32_t, nested_lambda_probe::kStorageWords> storage{};
    Result result;
    std::printf("=== CPU Nested-Lambda/Template Compiler Probe ===\n");
    RunProbe(storage, seed);
    bool exact = true;
    for (uint32_t raw_field = 0; raw_field < static_cast<uint32_t>(Field::Count); raw_field++) {
        const Field field = static_cast<Field>(raw_field);
        exact &= storage[nested_lambda_probe::FieldIndex(field)] ==
            nested_lambda_probe::Expected(field, seed);
    }
    result.Expect(exact, "CPU nested capture/template semantics");
    return result.ExitCode();
}
