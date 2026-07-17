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
#include "nested_lambda_cross_tu_api.h"
#include "nested_lambda_cross_tu_layout.h"

extern "C" PTO_DEVICE_FUNC void nested_probe_weak_context_dispatch(
    int32_t site_id, int32_t phase, uint64_t caller_context, L0TaskArgs *args) __attribute__((weak));
extern "C" PTO_DEVICE_FUNC void nested_probe_weak_args_dispatch(
    int32_t site_id, int32_t phase, L0TaskArgs *args) __attribute__((weak));
extern "C" PTO_DEVICE_FUNC void nested_probe_strong_context_dispatch(
    int32_t site_id, int32_t phase, uint64_t caller_context, L0TaskArgs *args);

namespace {

PTO_DEVICE_FUNC uint64_t TensorDigest(const Tensor &tensor)
{
    return tensor.buffer.addr + tensor.start_offset * 17ULL +
        static_cast<uint64_t>(static_cast<uint32_t>(tensor.version)) * 257ULL +
        tensor.shapes[0] * 65537ULL;
}

PTO_DEVICE_FUNC void ConsumeBoundArguments(L0TaskArgs *args)
{
    const Tensor &first = args->tensor(0).ref();
    const Tensor &second = args->tensor(1).ref();
    const Tensor &third = args->tensor(2).ref();
    args->scalar(0) = TensorDigest(first) * 3ULL + TensorDigest(second) * 5ULL +
        TensorDigest(third) * 7ULL + args->scalar(4);
    args->scalar(5) = 2;
    args->scalar(6) = 0;
}

} // namespace

PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_weak_context(
    int32_t site_id, uint64_t caller_context, L0TaskArgs *args)
{
    TaskOutputTensors outputs;
    if (nested_probe_weak_context_dispatch == nullptr) {
        args->scalar(6) = 1;
        return outputs;
    }
    nested_probe_weak_context_dispatch(
        site_id, static_cast<int32_t>(nested_lambda_cross_tu_probe::DispatchPhase::Prepare), caller_context, args);
    nested_probe_weak_context_dispatch(
        site_id, static_cast<int32_t>(nested_lambda_cross_tu_probe::DispatchPhase::WinnerBind), caller_context, args);
    ConsumeBoundArguments(args);
    return outputs;
}

PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_weak_args(
    int32_t site_id, L0TaskArgs *args)
{
    TaskOutputTensors outputs;
    if (nested_probe_weak_args_dispatch == nullptr) {
        args->scalar(6) = 1;
        return outputs;
    }
    nested_probe_weak_args_dispatch(
        site_id, static_cast<int32_t>(nested_lambda_cross_tu_probe::DispatchPhase::Prepare), args);
    nested_probe_weak_args_dispatch(
        site_id, static_cast<int32_t>(nested_lambda_cross_tu_probe::DispatchPhase::WinnerBind), args);
    ConsumeBoundArguments(args);
    return outputs;
}

PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_strong_context(
    int32_t site_id, uint64_t caller_context, L0TaskArgs *args)
{
    TaskOutputTensors outputs;
    nested_probe_strong_context_dispatch(
        site_id, static_cast<int32_t>(nested_lambda_cross_tu_probe::DispatchPhase::Prepare), caller_context, args);
    nested_probe_strong_context_dispatch(
        site_id, static_cast<int32_t>(nested_lambda_cross_tu_probe::DispatchPhase::WinnerBind), caller_context, args);
    ConsumeBoundArguments(args);
    return outputs;
}

PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_control(L0TaskArgs *args)
{
    TaskOutputTensors outputs;
    args->scalar(0) ^= nested_lambda_cross_tu_probe::kControlXor;
    return outputs;
}

PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_args_runtime_read(L0TaskArgs *args)
{
    TaskOutputTensors outputs;
    const auto *first = reinterpret_cast<const Tensor *>(args->scalar(8));
    const auto *second = reinterpret_cast<const Tensor *>(args->scalar(9));
    const auto *third = reinterpret_cast<const Tensor *>(args->scalar(10));
    args->scalar(4) = args->scalar(11);
    args->add_input(*first, *second, *third);
    ConsumeBoundArguments(args);
    args->scalar(5) = 0;
    return outputs;
}
