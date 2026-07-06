#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""fdwic submit dependency smoke test."""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestSubmitDependencySmoke(SceneTestCase):
    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/submit_dependency_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.INOUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "FILL_ALLOC_AIC",
                "source": "kernels/aic/fill_alloc.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT],
            },
            {
                "func_id": 1,
                "name": "MAKE_LEFT_AIC",
                "source": "kernels/aic/make_left.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "MAKE_RIGHT_AIV",
                "source": "kernels/aiv/make_right.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 3,
                "name": "FANIN_AIV",
                "source": "kernels/aiv/fanin.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "A5SimBd1N16",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 16},
        },
        {
            "name": "A5OnboardBd1N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 16},
        },
        {
            "name": "A5OnboardBd2N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 2},
            "params": {"n": 16},
        },
        {
            "name": "A5OnboardBd4N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 4},
            "params": {"n": 16},
        },
        {
            "name": "A5OnboardBd8N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 8},
            "params": {"n": 16},
        },
        {
            "name": "A5OnboardBd16N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"n": 16},
        },
        {
            "name": "A5OnboardBd24N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"n": 16},
        },
        {
            "name": "A5SimBd36N128",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128},
        },
        {
            "name": "A5OnboardBd36N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 16},
        },
        {
            "name": "A5OnboardBd36N128",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        x = torch.arange(n, dtype=torch.float32)
        y = torch.full((n,), -1.0, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("input", x),
            Tensor("output", y),
            Scalar("n", n),
        )

    def compute_golden(self, args, params):
        del params
        args.output[:] = args.input * 6.0 + 23.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
