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
#include "../ccec/nested_lambda_cross_tu_api.h"
#include "../ccec/nested_lambda_cross_tu_layout.h"

#include <cstdint>
#include <cstdio>

namespace {

void InitTensor(Tensor &tensor, uint32_t round, uint32_t tensor_index) {
    tensor.buffer.addr = nested_lambda_cross_tu_probe::TensorAddress(round, tensor_index);
    tensor.start_offset = nested_lambda_cross_tu_probe::TensorOffset(round, tensor_index);
    tensor.version = nested_lambda_cross_tu_probe::TensorVersion(round, tensor_index);
    tensor.shapes[0] = nested_lambda_cross_tu_probe::TensorShape(round, tensor_index);
}

}  // namespace

int main() {
    L0TaskArgs args;
    uint32_t completed_rounds = 0;
    uint32_t mismatches = 0;
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
        const TaskOutputTensors lazy_outputs = nested_probe_submit_args_runtime_read(&args);
        const uint64_t lazy_actual = args.scalar(0);
        checksum += lazy_actual;
        if (!lazy_outputs.empty() || args.scalar(5) != 0 || args.scalar(6) != 0 ||
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

    const bool exact = completed_rounds == nested_lambda_cross_tu_probe::kRounds && mismatches == 0 &&
                       checksum == nested_lambda_cross_tu_probe::ExpectedTotalChecksum() &&
                       sizeof(L0TaskArgs) == nested_lambda_cross_tu_probe::kExpectedL0TaskArgsBytes;
    std::printf("=== CPU L0TaskArgs Runtime-Read Probe ===\n");
    std::printf(
        "[VALUES] rounds=%u mismatches=%u checksum=0x%016llx L0TaskArgs=%zuB\n", completed_rounds, mismatches,
        static_cast<unsigned long long>(checksum), sizeof(L0TaskArgs)
    );
    std::printf("[ASSERT] CPU L0TaskArgs args-runtime-read semantics %s\n", exact ? "PASS" : "FAIL");
    std::printf("[SUMMARY] semantic_failures=%u\n", exact ? 0U : 1U);
    return exact ? 0 : 1;
}
