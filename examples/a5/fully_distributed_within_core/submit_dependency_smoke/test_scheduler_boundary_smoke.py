#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Protocol-limit boundary cases shared by the four FDWIC schedulers."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestSchedulerProtocolBoundary(SceneTestCase):
    """Validate wide-argument and fanin limits with a compact callable."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/scheduler_boundary_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "BOUNDARY_WRITER_AIC",
                "source": "kernels/aic/fill_alloc.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 1,
                "name": "MAX_ARGS_FANIN_AIV",
                "source": "kernels/aiv/wide_fanin.cpp",
                "core_type": "aiv",
                "signature": [D.IN] * 31 + [D.INOUT],
            },
            {
                "func_id": 2,
                "name": "EXACT_FANIN16_AIV",
                "source": "kernels/aiv/fanin16.cpp",
                "core_type": "aiv",
                "signature": [D.IN] * 16 + [D.INOUT],
            },
            {
                "func_id": 3,
                "name": "MAX_OUTPUTS_AIC",
                "source": "kernels/aic/max_outputs.cpp",
                "core_type": "aic",
                "signature": [D.OUT] * 32,
            },
            {
                "func_id": 4,
                "name": "BOUNDARY_WRITER_AIV",
                "source": "kernels/aiv/make_right.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 5,
                "name": "BOUNDARY_NOOP_AIC",
                "source": "kernels/noop.cpp",
                "core_type": "aic",
                "signature": [],
            },
            {
                "func_id": 6,
                "name": "BOUNDARY_NOOP_AIV",
                "source": "kernels/noop.cpp",
                "core_type": "aiv",
                "signature": [],
            },
            {
                "func_id": 7,
                "name": "BOUNDARY_DELAYED_WRITER_AIC",
                "source": "kernels/aic/delayed_fill.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "A5OnboardBd32MaxTensorScalarArgsNoFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 64, "mode": 0},
        },
        {
            "name": "A5OnboardBd32MaxTensorArgsExactFanin16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 31 * 32, "mode": 1},
        },
        {
            "name": "A5OnboardBd32MaxTensorArgsDedupFanin1",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 31 * 32, "mode": 2},
        },
        {
            "name": "A5OnboardBd32ExactExplicitFanin16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 16 * 32, "mode": 3},
        },
        {
            "name": "A5OnboardBd32ExplicitAndTensorMapDedup16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 16 * 32, "mode": 4},
        },
        {
            "name": "A5OnboardBd32MaxFreshOutputs32",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 64, "mode": 5},
        },
        {
            "name": "A5OnboardBd32MaxExistingWriterRegions32",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 64, "mode": 6},
        },
        {
            "name": "A5OnboardBd32MixedAutoExplicitFanin16AcrossEngines",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 16 * 256, "mode": 7},
        },
        {
            "name": "A5OnboardBd32TensorMapHistoryLowerBound",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 32},
            "params": {"n": 256, "mode": 8},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        return TaskArgsBuilder(
            Tensor("input", torch.arange(n, dtype=torch.float32)),
            Tensor("output", torch.full((n,), -1.0, dtype=torch.float32)),
            Tensor("dump", torch.full((n,), -1.0, dtype=torch.float32)),
            Scalar("n", n),
            Scalar("mode", int(params["mode"])),
        )

    def compute_golden(self, args, params):
        mode = int(params["mode"])
        args.output[:] = -1.0
        args.dump[:] = -1.0
        if mode == 0:
            args.dump[:31] = args.input[0]
        elif mode in {1, 2}:
            args.output[:] = args.input + 7.0
            args.dump[:31] = args.input[::32] + 7.0
        elif mode in {3, 4}:
            args.output[:] = args.input + 7.0
            args.dump[:16] = args.input[::32] + 7.0
        elif mode == 5:
            args.dump[:31] = torch.arange(100, 131, dtype=torch.float32)
            args.dump[32] = 138.0
        elif mode == 6:
            args.output[:32] = torch.arange(100, 132, dtype=torch.float32)
            args.dump[:31] = args.output[:31]
            args.dump[32] = 138.0
        elif mode == 7:
            chunk = 256
            for producer in range(16):
                begin = producer * chunk
                end = begin + chunk
                if producer % 2 == 0:
                    args.output[begin:end] = args.input[begin:end] + 7.0
                    args.dump[producer] = args.input[begin] + 7.0
                else:
                    args.output[begin:end] = args.input[begin:end] * 3.0 + 5.0
                    args.dump[producer] = args.input[begin] * 3.0 + 5.0
        elif mode == 8:
            args.output[:] = args.input + 7.0
            args.dump[:31] = args.output[0]

        # 所有宽参消费者都会求和 16 个 scalar，同时校验请求和
        # execution payload 的 scalar 上限。
        scalar_slot = 31 if mode in {0, 1, 2, 5, 6, 8} else 16
        args.dump[scalar_slot] = sum(range(1, 17))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
