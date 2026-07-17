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
#include "ccec_utils.h"

PTO_SYNCALL_AIC_KERNEL_META(nested_lambda_cross_tu_ctx_m0_0_mix_aic);
PTO_SYNCALL_AIC_KERNEL_META(nested_lambda_cross_tu_ctx_m1_1_mix_aic);
PTO_SYNCALL_AIC_KERNEL_META(nested_lambda_cross_tu_ctx_m2_2_mix_aic);
PTO_SYNCALL_AIC_KERNEL_META(nested_lambda_cross_tu_ctx_m3_3_mix_aic);
PTO_SYNCALL_AIC_KERNEL_META(nested_lambda_cross_tu_args_4_mix_aic);
PTO_SYNCALL_AIC_KERNEL_META(nested_lambda_cross_tu_strong_5_mix_aic);
PTO_SYNCALL_AIC_KERNEL_META(nested_lambda_cross_tu_runtime_args_6_mix_aic);

namespace {

using nested_lambda_cross_tu_probe::CallerContext;
using nested_lambda_cross_tu_probe::DispatchPhase;
using nested_lambda_cross_tu_probe::Field;
using nested_lambda_cross_tu_probe::Variant;

constexpr int32_t kSiteId = 7;
static_assert(sizeof(L0TaskArgs) == nested_lambda_cross_tu_probe::kExpectedL0TaskArgsBytes);

PTO_DEVICE_FUNC void BindContext(int32_t site_id, int32_t phase, uint64_t caller_context, L0TaskArgs *args) {
    if (site_id != kSiteId) return;
    const auto *context = reinterpret_cast<const CallerContext *>(caller_context);
    if (phase == static_cast<int32_t>(DispatchPhase::Prepare)) {
        args->scalar(4) = context->salt;
        return;
    }
    args->add_input(*context->first, *context->second, *context->third);
}

PTO_DEVICE_FUNC void InitTensor(Tensor &tensor, uint32_t round, uint32_t tensor_index) {
    tensor.buffer.addr = nested_lambda_cross_tu_probe::TensorAddress(round, tensor_index);
    tensor.start_offset = nested_lambda_cross_tu_probe::TensorOffset(round, tensor_index);
    tensor.version = nested_lambda_cross_tu_probe::TensorVersion(round, tensor_index);
    tensor.shapes[0] = nested_lambda_cross_tu_probe::TensorShape(round, tensor_index);
}

PTO_DEVICE_FUNC void StoreField(__gm__ uint32_t *storage, Field field, uint32_t value) {
    st_dev_b32(&storage[nested_lambda_cross_tu_probe::FieldIndex(field)], value);
}

PTO_DEVICE_FUNC void StoreResults(
    __gm__ uint32_t *storage, Variant variant, uint32_t completed_rounds, uint32_t mismatches,
    uint32_t dispatcher_calls, uint32_t materializations, uint64_t checksum
) {
    StoreField(storage, Field::CompletedRounds, completed_rounds);
    StoreField(storage, Field::MismatchCount, mismatches);
    StoreField(storage, Field::DispatcherCalls, dispatcher_calls);
    StoreField(storage, Field::AddressMaterializations, materializations);
    StoreField(storage, Field::ChecksumLow, static_cast<uint32_t>(checksum));
    StoreField(storage, Field::ChecksumHigh, static_cast<uint32_t>(checksum >> 32));
    StoreField(storage, Field::L0TaskArgsBytes, static_cast<uint32_t>(sizeof(L0TaskArgs)));
    StoreField(storage, Field::VariantEcho, static_cast<uint32_t>(variant));
    dsb(DSB_ALL);
}

template <bool StrongDispatcher>
PTO_DEVICE_FUNC TaskOutputTensors SubmitContext(uint64_t caller_context, L0TaskArgs *args) {
    if constexpr (StrongDispatcher) {
        return nested_probe_submit_strong_context(kSiteId, caller_context, args);
    }
    return nested_probe_submit_weak_context(kSiteId, caller_context, args);
}

template <uint32_t Materializations, bool StrongDispatcher>
PTO_DEVICE_FUNC void RunContextVariant(__gm__ uint32_t *storage, Variant variant) {
    L0TaskArgs args;
    uint32_t completed_rounds = 0;
    uint32_t mismatches = 0;
    uint32_t dispatcher_calls = 0;
    uint64_t checksum = 0;

    for (uint32_t round = 0; round < nested_lambda_cross_tu_probe::kRounds; round++) {
        Tensor first;
        Tensor second;
        Tensor third;
        InitTensor(first, round, 0);
        InitTensor(second, round, 1);
        InitTensor(third, round, 2);
        const CallerContext context{&first, &second, &third, nested_lambda_cross_tu_probe::ContextSalt(round)};

        args.reset();
        if constexpr (Materializations >= 1) args.scalar(8) = reinterpret_cast<uint64_t>(&first);
        if constexpr (Materializations >= 2) args.scalar(9) = reinterpret_cast<uint64_t>(&second);
        if constexpr (Materializations >= 3) args.scalar(10) = reinterpret_cast<uint64_t>(&third);

        const TaskOutputTensors lazy_outputs =
            SubmitContext<StrongDispatcher>(reinterpret_cast<uint64_t>(&context), &args);
        const uint64_t lazy_actual = args.scalar(0);
        checksum += lazy_actual;
        dispatcher_calls += static_cast<uint32_t>(args.scalar(5));
        if (!lazy_outputs.empty() || args.scalar(6) != 0 ||
            lazy_actual != nested_lambda_cross_tu_probe::ExpectedLazyDigest(round)) {
            mismatches++;
        }

        for (uint32_t submit = 1; submit < nested_lambda_cross_tu_probe::kSubmitsPerRound; submit++) {
            args.reset();
            args.add_scalar(nested_lambda_cross_tu_probe::ControlInput(round, submit));
            const TaskOutputTensors control_outputs = nested_probe_submit_control(&args);
            const uint64_t control_actual = args.scalar(0);
            checksum += control_actual;
            if (!control_outputs.empty() ||
                control_actual != nested_lambda_cross_tu_probe::ExpectedControlResult(round, submit)) {
                mismatches++;
            }
        }
        completed_rounds++;
    }

    StoreResults(storage, variant, completed_rounds, mismatches, dispatcher_calls, Materializations, checksum);
}

template <bool RuntimeRead>
PTO_DEVICE_FUNC void RunArgsStorageVariant(__gm__ uint32_t *storage, Variant variant) {
    L0TaskArgs args;
    uint32_t completed_rounds = 0;
    uint32_t mismatches = 0;
    uint32_t dispatcher_calls = 0;
    uint64_t checksum = 0;

    for (uint32_t round = 0; round < nested_lambda_cross_tu_probe::kRounds; round++) {
        Tensor first;
        Tensor second;
        Tensor third;
        InitTensor(first, round, 0);
        InitTensor(second, round, 1);
        InitTensor(third, round, 2);

        args.reset();
        args.scalar(8) = reinterpret_cast<uint64_t>(&first);
        args.scalar(9) = reinterpret_cast<uint64_t>(&second);
        args.scalar(10) = reinterpret_cast<uint64_t>(&third);
        args.scalar(11) = nested_lambda_cross_tu_probe::ContextSalt(round);
        TaskOutputTensors lazy_outputs;
        if constexpr (RuntimeRead) {
            lazy_outputs = nested_probe_submit_args_runtime_read(&args);
        } else {
            lazy_outputs = nested_probe_submit_weak_args(kSiteId, &args);
        }
        const uint64_t lazy_actual = args.scalar(0);
        checksum += lazy_actual;
        dispatcher_calls += static_cast<uint32_t>(args.scalar(5));
        if (!lazy_outputs.empty() || args.scalar(6) != 0 ||
            lazy_actual != nested_lambda_cross_tu_probe::ExpectedLazyDigest(round)) {
            mismatches++;
        }

        for (uint32_t submit = 1; submit < nested_lambda_cross_tu_probe::kSubmitsPerRound; submit++) {
            args.reset();
            args.add_scalar(nested_lambda_cross_tu_probe::ControlInput(round, submit));
            const TaskOutputTensors control_outputs = nested_probe_submit_control(&args);
            const uint64_t control_actual = args.scalar(0);
            checksum += control_actual;
            if (!control_outputs.empty() ||
                control_actual != nested_lambda_cross_tu_probe::ExpectedControlResult(round, submit)) {
                mismatches++;
            }
        }
        completed_rounds++;
    }

    StoreResults(storage, variant, completed_rounds, mismatches, dispatcher_calls, 3, checksum);
}

}  // namespace

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void
nested_probe_weak_context_dispatch(int32_t site_id, int32_t phase, uint64_t caller_context, L0TaskArgs *args) {
    BindContext(site_id, phase, caller_context, args);
}

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void
nested_probe_weak_args_dispatch(int32_t site_id, int32_t phase, L0TaskArgs *args) {
    if (site_id != kSiteId) return;
    if (phase == static_cast<int32_t>(DispatchPhase::Prepare)) {
        args->scalar(4) = args->scalar(11);
        return;
    }
    const auto *first = reinterpret_cast<const Tensor *>(args->scalar(8));
    const auto *second = reinterpret_cast<const Tensor *>(args->scalar(9));
    const auto *third = reinterpret_cast<const Tensor *>(args->scalar(10));
    args->add_input(*first, *second, *third);
}

extern "C" PTO_DEVICE_FUNC void
nested_probe_strong_context_dispatch(int32_t site_id, int32_t phase, uint64_t caller_context, L0TaskArgs *args) {
    BindContext(site_id, phase, caller_context, args);
}

// Keep the caller body in a weak __aicore__ function, matching the real
// aicpu_orchestration_entry. Putting it directly in a __global__ wrapper would
// move its locals from the orchestration function stack to the kernel stack and
// miss the suspected reg93 stack-address materialization path.
extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void nested_probe_orchestration_ctx_m0(__gm__ uint32_t *storage) {
    RunContextVariant<0, false>(storage, Variant::WeakContextMaterialize0);
}

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void nested_probe_orchestration_ctx_m1(__gm__ uint32_t *storage) {
    RunContextVariant<1, false>(storage, Variant::WeakContextMaterialize1);
}

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void nested_probe_orchestration_ctx_m2(__gm__ uint32_t *storage) {
    RunContextVariant<2, false>(storage, Variant::WeakContextMaterialize2);
}

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void nested_probe_orchestration_ctx_m3(__gm__ uint32_t *storage) {
    RunContextVariant<3, false>(storage, Variant::WeakContextMaterialize3);
}

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void nested_probe_orchestration_args(__gm__ uint32_t *storage) {
    RunArgsStorageVariant<false>(storage, Variant::WeakArgsStorage);
}

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void nested_probe_orchestration_strong(__gm__ uint32_t *storage) {
    RunContextVariant<0, true>(storage, Variant::StrongContext);
}

extern "C" __attribute__((weak)) PTO_DEVICE_FUNC void
nested_probe_orchestration_runtime_args(__gm__ uint32_t *storage) {
    RunArgsStorageVariant<true>(storage, Variant::ArgsRuntimeRead);
}

extern "C" __global__ __aicore__ void nested_lambda_cross_tu_ctx_m0_0_mix_aic(__gm__ uint32_t *storage) {
    if (get_block_idx() == 0) nested_probe_orchestration_ctx_m0(storage);
}

extern "C" __global__ __aicore__ void nested_lambda_cross_tu_ctx_m1_1_mix_aic(__gm__ uint32_t *storage) {
    if (get_block_idx() == 0) nested_probe_orchestration_ctx_m1(storage);
}

extern "C" __global__ __aicore__ void nested_lambda_cross_tu_ctx_m2_2_mix_aic(__gm__ uint32_t *storage) {
    if (get_block_idx() == 0) nested_probe_orchestration_ctx_m2(storage);
}

extern "C" __global__ __aicore__ void nested_lambda_cross_tu_ctx_m3_3_mix_aic(__gm__ uint32_t *storage) {
    if (get_block_idx() == 0) nested_probe_orchestration_ctx_m3(storage);
}

extern "C" __global__ __aicore__ void nested_lambda_cross_tu_args_4_mix_aic(__gm__ uint32_t *storage) {
    if (get_block_idx() == 0) nested_probe_orchestration_args(storage);
}

extern "C" __global__ __aicore__ void nested_lambda_cross_tu_strong_5_mix_aic(__gm__ uint32_t *storage) {
    if (get_block_idx() == 0) nested_probe_orchestration_strong(storage);
}

extern "C" __global__ __aicore__ void nested_lambda_cross_tu_runtime_args_6_mix_aic(__gm__ uint32_t *storage) {
    if (get_block_idx() == 0) nested_probe_orchestration_runtime_args(storage);
}
