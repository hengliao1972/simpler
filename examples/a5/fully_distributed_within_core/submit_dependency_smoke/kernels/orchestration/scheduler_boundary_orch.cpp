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

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_BOUNDARY_WRITER_AIC 0
#define FUNC_MAX_ARGS_FANIN_AIV 1

PTO_DEVICE_FUNC inline void add_max_scalar_args(L0TaskArgs &args) {
    for (uint32_t scalar = 0; scalar < MAX_SCALAR_ARGS; ++scalar) {
        args.add_scalar(static_cast<uint64_t>(scalar + 1U));
    }
}

PTO_DEVICE_FUNC inline void submit_wide_reader(const Tensor &source, const Tensor &dump) {
    L0TaskArgs args;
    for (uint32_t tensor = 0; tensor < MAX_TENSOR_ARGS - 1U; ++tensor)
        args.add_input(source);
    args.add_inout(dump);
    add_max_scalar_args(args);
    rt_submit_aiv_task(FUNC_MAX_ARGS_FANIN_AIV, args);
}

PTO_DEVICE_FUNC inline void submit_view_reader(const Tensor &source, const Tensor &dump) {
    constexpr uint32_t chunk = 32;
    const uint32_t view_shape[1] = {chunk};
    Tensor views[MAX_TENSOR_ARGS - 1U];
    L0TaskArgs args;
    for (uint32_t tensor = 0; tensor < MAX_TENSOR_ARGS - 1U; ++tensor) {
        const uint32_t view_offset[1] = {tensor * chunk};
        views[tensor] = Tensor::view(source, view_shape, view_offset);
        args.add_input(views[tensor]);
    }
    args.add_inout(dump);
    add_max_scalar_args(args);
    rt_submit_aiv_task(FUNC_MAX_ARGS_FANIN_AIV, args);
}

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig
aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void
aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &input = orch_args.tensor(0).ref();
    const Tensor &output = orch_args.tensor(1).ref();
    const Tensor &dump = orch_args.tensor(2).ref();
    const uint64_t n = orch_args.scalar(0);
    const uint64_t mode = orch_args.scalar(1);

    if (mode == 0) {
        // 32 个 Tensor 和 16 个 scalar 同时取上限；外部 INPUT 不产生 fanin。
        submit_wide_reader(input, dump);
        return;
    }

    if (mode == 1) {
        // 16 个 writer 覆盖 31 个不重叠 region，宽 reader 应得到恰好 16 个
        // 去重后的 producer，覆盖跨核执行包的 fanin 上限。
        constexpr uint32_t chunk = 32;
        constexpr uint32_t writer_count = 16;
        for (uint32_t writer = 0; writer < writer_count; ++writer) {
            const uint32_t begin = writer * 2U * chunk;
            const uint32_t width = writer + 1U == writer_count ? chunk : 2U * chunk;
            const uint32_t writer_shape[1] = {width};
            const uint32_t writer_offset[1] = {begin};
            Tensor input_view = Tensor::view(input, writer_shape, writer_offset);
            Tensor output_view = Tensor::view(output, writer_shape, writer_offset);
            L0TaskArgs writer_args;
            writer_args.add_input(input_view);
            writer_args.add_inout(output_view);
            writer_args.add_scalar(static_cast<uint64_t>(width));
            rt_submit_aic_task(FUNC_BOUNDARY_WRITER_AIC, writer_args);
        }
        submit_view_reader(output, dump);
        return;
    }

    if (mode == 2) {
        // 同一个 writer 产生 31 个不同 view。调度器必须把 31 次命中去重为
        // 一个 producer，而不能把 Tensor 参数数误当作 fanin 数。
        L0TaskArgs writer_args;
        writer_args.add_input(input);
        writer_args.add_inout(output);
        writer_args.add_scalar(n);
        rt_submit_aic_task(FUNC_BOUNDARY_WRITER_AIC, writer_args);
        submit_view_reader(output, dump);
    }
}

}  // extern "C"
