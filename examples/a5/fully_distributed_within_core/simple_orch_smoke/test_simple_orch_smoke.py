#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal fdwic user-orchestration smoke test."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestSimpleOrchSmoke(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/simple_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MARK_AIC",
                "source": "kernels/aic/mark_core.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 1,
                "name": "MARK_AIV",
                "source": "kernels/aiv/mark_core.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "A5SimBd1Delta7",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 3, "delta": 7},
        },
        {
            "name": "A5OnboardBd1Delta7",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 3, "delta": 7},
        },
        {
            "name": "A5SimBd36Delta5",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 3, "delta": 5},
        },
        {
            "name": "A5OnboardBd36Delta5",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 3, "delta": 5},
        },
        {
            "name": "A5SimBd36Delta11",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 3, "delta": 11},
        },
        {
            "name": "A5OnboardBd36Delta11",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 3, "delta": 11},
        },
        {
            "name": "A5SimBd36ManyTasks",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 144, "delta": 13},
        },
        {
            "name": "A5OnboardBd36ManyTasks",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 144, "delta": 13},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        delta = int(params["delta"])
        x = torch.arange(n, dtype=torch.float32)
        y = torch.full((n * 16,), -1.0, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("input", x),
            Tensor("output", y),
            Scalar("n", n),
            Scalar("delta", delta),
        )

    def compute_golden(self, args, params):
        n = int(params["n"])
        delta = float(params["delta"])
        args.output[:] = -1.0
        for i in range(n):
            args.output[i * 16] = args.input[i] + delta


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
