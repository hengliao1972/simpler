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
#define FUNC_EXACT_FANIN16_AIV 2
#define FUNC_MAX_OUTPUTS_AIC 3
#define FUNC_BOUNDARY_WRITER_AIV 4
#define FUNC_BOUNDARY_NOOP_AIC 5
#define FUNC_BOUNDARY_NOOP_AIV 6
#define FUNC_BOUNDARY_DELAYED_WRITER_AIC 7

constexpr uint32_t kExactFanin = 16;
constexpr uint32_t kChunk = 32;
constexpr uint32_t kSkewChunk = 256;
constexpr uint32_t kMaxOutputs = MAX_TENSOR_ARGS;
constexpr uint32_t kHistoryLowerBound = 64;

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
    const uint32_t view_shape[1] = {kChunk};
    Tensor views[MAX_TENSOR_ARGS - 1U];
    L0TaskArgs args;
    for (uint32_t tensor = 0; tensor < MAX_TENSOR_ARGS - 1U; ++tensor) {
        const uint32_t view_offset[1] = {tensor * kChunk};
        views[tensor] = Tensor::view(source, view_shape, view_offset);
        args.add_input(views[tensor]);
    }
    args.add_inout(dump);
    add_max_scalar_args(args);
    rt_submit_aiv_task(FUNC_MAX_ARGS_FANIN_AIV, args);
}

PTO_DEVICE_FUNC inline void submit_exact_fanin_reader(
    const Tensor &output, const Tensor &dump, const PTO2TaskId dependencies[kExactFanin], bool infer_from_tensor_map
) {
    const uint32_t view_shape[1] = {kChunk};
    Tensor views[kExactFanin];
    L0TaskArgs args;
    for (uint32_t index = 0; index < kExactFanin; ++index) {
        const uint32_t view_offset[1] = {index * kChunk};
        views[index] = Tensor::view(output, view_shape, view_offset);
        if (infer_from_tensor_map) {
            args.add_input(views[index]);
        } else {
            // NO_DEP 让本用例只由显式 task id 建立 fanin，不让 TensorMap
            // 查询偶然补齐顺序。
            args.add_no_dep(views[index]);
        }
    }
    args.add_inout(dump);
    add_max_scalar_args(args);
    args.set_dependencies(dependencies, kExactFanin);
    rt_submit_aiv_task(FUNC_EXACT_FANIN16_AIV, args);
}

PTO_DEVICE_FUNC inline void
submit_explicit_fanin_case(const Tensor &input, const Tensor &output, const Tensor &dump, bool infer_from_tensor_map) {
    const uint32_t view_shape[1] = {kChunk};
    PTO2TaskId dependencies[kExactFanin];
    for (uint32_t writer = 0; writer < kExactFanin; ++writer) {
        const uint32_t view_offset[1] = {writer * kChunk};
        Tensor input_view = Tensor::view(input, view_shape, view_offset);
        Tensor output_view = Tensor::view(output, view_shape, view_offset);
        L0TaskArgs writer_args;
        writer_args.add_input(input_view);
        writer_args.add_inout(output_view);
        writer_args.add_scalar(static_cast<uint64_t>(kChunk));
        TaskOutputTensors writer_result = rt_submit_aic_task(FUNC_BOUNDARY_WRITER_AIC, writer_args);
        dependencies[writer] = writer_result.task_id();
    }
    submit_exact_fanin_reader(output, dump, dependencies, infer_from_tensor_map);
}

template <typename OutputRef>
PTO_DEVICE_FUNC inline void submit_max_output_consumers(const OutputRef outputs[kMaxOutputs], const Tensor &dump) {
    L0TaskArgs wide_args;
    for (uint32_t slot = 0; slot < kMaxOutputs - 1U; ++slot) {
        wide_args.add_input(outputs[slot]);
    }
    wide_args.add_inout(dump);
    add_max_scalar_args(wide_args);
    rt_submit_aiv_task(FUNC_MAX_ARGS_FANIN_AIV, wide_args);

    const uint32_t tail_shape[1] = {1};
    const uint32_t tail_offset[1] = {kMaxOutputs};
    Tensor tail_dump = Tensor::view(dump, tail_shape, tail_offset);
    L0TaskArgs tail_args;
    tail_args.add_input(outputs[kMaxOutputs - 1U]);
    tail_args.add_inout(tail_dump);
    tail_args.add_scalar(uint64_t{1});
    rt_submit_aic_task(FUNC_BOUNDARY_WRITER_AIC, tail_args);
}

PTO_DEVICE_FUNC inline void submit_max_fresh_outputs(const Tensor &dump) {
    const uint32_t output_shape[1] = {1};
    TensorCreateInfo output_info(output_shape, 1, DataType::FLOAT32);
    L0TaskArgs producer_args;
#if PTO_FDWIC_SHARED_MAP && PTO_FDWIC_SCHEDULER_MODE >= 1 && PTO_FDWIC_SCHEDULER_MODE <= 4
    SharedTaskOutputs produced = rt_submit_aic_task_deferred_compete_first(
        FUNC_MAX_OUTPUTS_AIC, kMaxOutputs, producer_args,
        [&](L0TaskArgs &args) PTO_DEVICE_FUNC {
            for (uint32_t slot = 0; slot < kMaxOutputs; ++slot)
                args.add_output(output_info);
        }
    );
    if (produced.size() != kMaxOutputs) return;
    FdwicOutputRef output_refs[kMaxOutputs];
    for (uint32_t slot = 0; slot < kMaxOutputs; ++slot)
        output_refs[slot] = produced.output_ref(slot);
    submit_max_output_consumers(output_refs, dump);
#else
    TaskOutputTensors produced =
        rt_submit_aic_task_compete_first(FUNC_MAX_OUTPUTS_AIC, producer_args, [&](L0TaskArgs &args) PTO_DEVICE_FUNC {
            for (uint32_t slot = 0; slot < kMaxOutputs; ++slot)
                args.add_output(output_info);
        });
    if (produced.size() != kMaxOutputs) return;
    __gm__ const Tensor *output_refs[kMaxOutputs];
    for (uint32_t slot = 0; slot < kMaxOutputs; ++slot)
        output_refs[slot] = &produced.get_ref(slot);
    // 非 shared 回退只为保持源文件可编译；本边界矩阵实际只运行
    // shared cross-core 四种模式。
    L0TaskArgs wide_args;
    for (uint32_t slot = 0; slot < kMaxOutputs - 1U; ++slot)
        wide_args.add_input(*output_refs[slot]);
    wide_args.add_inout(dump);
    add_max_scalar_args(wide_args);
    rt_submit_aiv_task(FUNC_MAX_ARGS_FANIN_AIV, wide_args);

    const uint32_t tail_shape[1] = {1};
    const uint32_t tail_offset[1] = {kMaxOutputs};
    Tensor tail_dump = Tensor::view(dump, tail_shape, tail_offset);
    L0TaskArgs tail_args;
    tail_args.add_input(*output_refs[kMaxOutputs - 1U]);
    tail_args.add_inout(tail_dump);
    tail_args.add_scalar(uint64_t{1});
    rt_submit_aic_task(FUNC_BOUNDARY_WRITER_AIC, tail_args);
#endif
}

PTO_DEVICE_FUNC inline void submit_max_existing_writers(const Tensor &output, const Tensor &dump) {
    const uint32_t output_shape[1] = {1};
    Tensor output_views[kMaxOutputs];
    L0TaskArgs producer_args;
    for (uint32_t slot = 0; slot < kMaxOutputs; ++slot) {
        const uint32_t output_offset[1] = {slot};
        output_views[slot] = Tensor::view(output, output_shape, output_offset);
        producer_args.add_output(output_views[slot]);
    }
    rt_submit_aic_task(FUNC_MAX_OUTPUTS_AIC, producer_args);
    submit_max_output_consumers(output_views, dump);
}

PTO_DEVICE_FUNC inline void submit_mixed_fanin_case(const Tensor &input, const Tensor &output, const Tensor &dump) {
    const uint32_t view_shape[1] = {kSkewChunk};
    Tensor output_views[kExactFanin];
    PTO2TaskId explicit_dependencies[kExactFanin];
    uint32_t explicit_count = 0;

    for (uint32_t producer = 0; producer < kExactFanin; ++producer) {
        const uint32_t view_offset[1] = {producer * kSkewChunk};
        Tensor input_view = Tensor::view(input, view_shape, view_offset);
        output_views[producer] = Tensor::view(output, view_shape, view_offset);
        L0TaskArgs writer_args;
        if ((producer & 1U) == 0) {
            writer_args.add_input(input_view);
            writer_args.add_inout(output_views[producer]);
            writer_args.add_scalar(static_cast<uint64_t>(kSkewChunk));
            // 让第一个自动依赖 producer 明显晚于其余 writer 完成；如果
            // consumer 漏掉这条边，数值 golden 会稳定暴露，而不是靠偶然时序。
            if (producer == 0) {
                writer_args.add_scalar(uint64_t{12000000});
                rt_submit_aic_task(FUNC_BOUNDARY_DELAYED_WRITER_AIC, writer_args);
            } else {
                rt_submit_aic_task(FUNC_BOUNDARY_WRITER_AIC, writer_args);
            }
        } else {
            writer_args.add_input(input_view);
            writer_args.add_output(output_views[producer]);
            writer_args.add_scalar(static_cast<uint64_t>(kSkewChunk));
            TaskOutputTensors result = rt_submit_aiv_task(FUNC_BOUNDARY_WRITER_AIV, writer_args);
            // 八个 AIV producer 各重复两次，显式列表本身需要先去重；再与
            // 八个自动 TensorMap producer 合并，最终恰好达到 fanin=16。
            explicit_dependencies[explicit_count++] = result.task_id();
            explicit_dependencies[explicit_count++] = result.task_id();
        }
    }

    L0TaskArgs consumer_args;
    for (uint32_t producer = 0; producer < kExactFanin; ++producer) {
        if ((producer & 1U) == 0) {
            consumer_args.add_input(output_views[producer]);
        } else {
            consumer_args.add_no_dep(output_views[producer]);
        }
    }
    consumer_args.add_inout(dump);
    add_max_scalar_args(consumer_args);
    consumer_args.set_dependencies(explicit_dependencies, explicit_count);
    rt_submit_aiv_task(FUNC_EXACT_FANIN16_AIV, consumer_args);
}

PTO_DEVICE_FUNC inline void
submit_history_lower_bound_case(const Tensor &input, const Tensor &output, const Tensor &dump, uint64_t n) {
    L0TaskArgs writer_args;
    writer_args.add_input(input);
    writer_args.add_output(output);
    writer_args.add_scalar(n);
    writer_args.add_scalar(uint64_t{12000000});
    rt_submit_aic_task(FUNC_BOUNDARY_DELAYED_WRITER_AIC, writer_args);

    // consumer 的 task id 为 64，task0 正好落在默认 H=64 的闭合下界。
    // 中间 task 不读写 TensorMap，避免把历史窗口边界与 region 容量混算。
    for (uint32_t task = 1; task < kHistoryLowerBound; ++task) {
        L0TaskArgs noop_args;
        if ((task & 1U) == 0) {
            rt_submit_aic_task(FUNC_BOUNDARY_NOOP_AIC, noop_args);
        } else {
            rt_submit_aiv_task(FUNC_BOUNDARY_NOOP_AIV, noop_args);
        }
    }
    submit_wide_reader(output, dump);
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
        constexpr uint32_t writer_count = 16;
        for (uint32_t writer = 0; writer < writer_count; ++writer) {
            const uint32_t begin = writer * 2U * kChunk;
            const uint32_t width = writer + 1U == writer_count ? kChunk : 2U * kChunk;
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
        return;
    }

    if (mode == 3 || mode == 4) {
        // mode3 只使用 16 个显式依赖；mode4 让同一组 producer 同时从
        // TensorMap 与显式依赖命中，验证跨来源去重后仍是 16。
        submit_explicit_fanin_case(input, output, dump, mode == 4);
        return;
    }

    if (mode == 5) {
        submit_max_fresh_outputs(dump);
        return;
    }

    if (mode == 6) {
        submit_max_existing_writers(output, dump);
        return;
    }

    if (mode == 7) {
        submit_mixed_fanin_case(input, output, dump);
        return;
    }

    if (mode == 8) submit_history_lower_bound_case(input, output, dump, n);
}

}  // extern "C"
