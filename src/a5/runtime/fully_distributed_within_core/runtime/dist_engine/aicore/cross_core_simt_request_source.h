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

#include <cstdint>

#include "dist_engine/common/cross_core_simt_request_protocol.h"
#include "pto_types.h"

namespace fdwic::cross_core {

PTO_DEVICE_FUNC inline bool
ValidateSimtL0TaskArgs(const L0TaskArgs &args, uint32_t task_id, uint32_t expected_output_count = UINT32_MAX) {
    if (task_id == UINT32_MAX || args.has_error || args.tensor_count() < 0 ||
        args.tensor_count() > static_cast<int32_t>(kExecMaxTensors) || args.scalar_count() < 0 ||
        args.scalar_count() > static_cast<int32_t>(kExecMaxScalars) ||
        args.explicit_dep_count() > kSimtRequestMaxExplicitDependencies) {
        return false;
    }
    uint32_t output_count = 0;
    for (int32_t tensor = 0; tensor < args.tensor_count(); ++tensor) {
        const TensorArgType tag = args.tag(tensor);
        if (!SimtRequestTensorTagValid(tag)) return false;
        if (tag == TensorArgType::OUTPUT) {
            if (!args.tensor(tensor).has_create_info()) return false;
            ++output_count;
            continue;
        }
#if PTO_FDWIC_SHARED_MAP
        if (args.tensor(tensor).tensor_from_shared_output()) {
            const FdwicOutputRef ref = args.tensor(tensor).shared_output_ref();
            if (!fdwic_plain_output_ref(ref) || ref.producer_task_id < 0 ||
                static_cast<uint32_t>(ref.producer_task_id) >= task_id || ref.output_slot < 0 ||
                static_cast<uint32_t>(ref.output_slot) >= MAX_TENSOR_ARGS) {
                return false;
            }
            continue;
        }
#endif
        if (!args.tensor(tensor).has_existing_tensor()) return false;
    }
    for (uint32_t dependency = 0; dependency < args.explicit_dep_count(); ++dependency) {
        const PTO2TaskId id = args.explicit_dep(dependency);
        if (!id.is_valid() || id.ring() != 0 || id.local() >= task_id) return false;
    }
    return expected_output_count == UINT32_MAX || output_count == expected_output_count;
}

PTO_DEVICE_FUNC inline uint32_t SimtL0TaskArgsReferenceMask(const L0TaskArgs &args) {
    uint32_t mask = 0;
#if PTO_FDWIC_SHARED_MAP
    for (int32_t tensor = 0; tensor < args.tensor_count(); ++tensor) {
        if (args.tensor(tensor).tensor_from_shared_output()) mask |= uint32_t{1} << static_cast<uint32_t>(tensor);
    }
#else
    (void)args;
#endif
    return mask;
}

struct SimtL0TaskArgsRequestSource {
    const L0TaskArgs &args;

    PTO_DEVICE_FUNC bool TensorIsReference(uint32_t tensor) const {
#if PTO_FDWIC_SHARED_MAP
        return args.tensor(static_cast<int32_t>(tensor)).tensor_from_shared_output();
#else
        (void)tensor;
        return false;
#endif
    }

    PTO_DEVICE_FUNC uint64_t TensorWord(uint32_t tensor, uint32_t word) const {
        const int32_t index = static_cast<int32_t>(tensor);
        if (args.tag(index) == TensorArgType::OUTPUT) {
            const auto *words = reinterpret_cast<const uint64_t *>(&args.tensor(index).create_info());
            return words[word];
        }
#if PTO_FDWIC_SHARED_MAP
        if (args.tensor(index).tensor_from_shared_output()) {
            const FdwicOutputRef ref = args.tensor(index).shared_output_ref();
            const auto *words = reinterpret_cast<const uint64_t *>(&ref);
            return word < sizeof(FdwicOutputRef) / sizeof(uint64_t) ? words[word] : 0;
        }
#endif
#if defined(__CCE_AICORE__)
        if (args.tensor(index).tensor_from_gm()) {
            __gm__ const auto *words = reinterpret_cast<__gm__ const uint64_t *>(&args.tensor(index).gm_ref());
            return words[word];
        }
#endif
        const auto *words = reinterpret_cast<const uint64_t *>(&args.tensor(index).ref());
        return words[word];
    }

    PTO_DEVICE_FUNC uint64_t Scalar(uint32_t scalar) const { return args.scalar(static_cast<int32_t>(scalar)); }

    PTO_DEVICE_FUNC uint64_t ExplicitDependency(uint32_t dependency) const { return args.explicit_dep(dependency).raw; }

    PTO_DEVICE_FUNC TensorArgType TensorTag(uint32_t tensor) const { return args.tag(static_cast<int32_t>(tensor)); }
};

}  // namespace fdwic::cross_core
