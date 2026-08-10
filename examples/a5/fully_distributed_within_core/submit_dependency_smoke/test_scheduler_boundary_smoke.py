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
        else:
            args.output[:] = args.input + 7.0
            args.dump[:31] = args.input[::32] + 7.0
        # 额外验证 16 个 scalar 也完整到达执行 kernel。
        args.dump[31] = sum(range(1, 17))


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
