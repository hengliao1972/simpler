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
// Pure-CCEC AICore compiler probe. This intentionally uses no AscendC header
// and no PyPTO runtime: ccec must compile the nested lambda/template shape.
#include "../nested_lambda_probe.h"
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(nested_lambda_ccec);

namespace {

using nested_lambda_probe::Field;

__aicore__ inline uint32_t DeviceFieldIndex(Field field)
{
    return static_cast<uint32_t>(field) * nested_lambda_probe::kCacheLineWords;
}

__aicore__ inline void StoreField(
    __gm__ uint32_t *storage, Field field, uint32_t value)
{
    st_dev_b32(&storage[DeviceFieldIndex(field)], value);
}

__aicore__ inline void RunProbe(__gm__ uint32_t *storage, uint32_t seed)
{
    uint32_t outer_calls = 0;
    uint32_t input_calls = 0;
    uint32_t output_calls = 0;
    uint32_t scalar_calls = 0;
    uint32_t reference_state = seed + 5U;

    const nested_lambda_probe::BuildResult built = nested_lambda_probe::Submit(
        [&](nested_lambda_probe::SubmitBuilder &builder) __aicore__ {
            outer_calls++;
            const uint32_t outer_local = seed + 3U;
            const uint32_t scalar_capture = seed + 0x30U;
            builder.AddInput([&]() __aicore__ -> nested_lambda_probe::TensorLike {
                input_calls++;
                reference_state += 7U;
                return {outer_local + reference_state};
            });
            builder.AddOutput(
                [outer_local, &output_calls]() __aicore__ -> nested_lambda_probe::TensorLike {
                    output_calls++;
                    return {outer_local + 0x20U};
                });
            builder.AddScalar([scalar_capture, &scalar_calls]() __aicore__ -> uint32_t {
                scalar_calls++;
                return scalar_capture + 1U;
            });
        });

    StoreField(storage, Field::OuterCalls, outer_calls);
    StoreField(storage, Field::InputCalls, input_calls);
    StoreField(storage, Field::OutputCalls, output_calls);
    StoreField(storage, Field::ScalarCalls, scalar_calls);
    StoreField(storage, Field::ReferenceState, reference_state);
    StoreField(storage, Field::InputValue, built.input.value);
    StoreField(storage, Field::OutputValue, built.output.value);
    StoreField(storage, Field::ScalarValue, built.scalar);
    StoreField(storage, Field::CombinedValue, built.input.value + built.output.value + built.scalar);
}

} // namespace

extern "C" __global__ __aicore__ void KERNEL_ENTRY(nested_lambda_ccec)(
    __gm__ uint32_t *storage, uint32_t seed)
{
    if (get_block_idx() != 0) return;
    RunProbe(storage, seed);
    dsb(DSB_ALL);
}
