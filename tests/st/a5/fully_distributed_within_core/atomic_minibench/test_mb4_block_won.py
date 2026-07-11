#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MB-4 block.won delivery table — anchor deposit / follower drain / remaining zero.

Reuses the mix_coown workload (1C+2V MIX tasks exercising block.won) with
high block counts to stress the anchor->follower deposit/drain path and the
joint completion flag.

Criteria (docs MB-4):
  - Every multi-core task's remaining is decremented to 0 exactly once
  - No hang (watchdog), no early flag set
  - DEPSIG consistent (set PTO_DIST_DEPSIG=1)

Env knobs:
  PTO_DIST_DEPSIG=1           — enable dependency-graph signature (internal assert)
  PTO_DIST_FAKE_EXEC_NS=N     — uniform fake exec time to spread completion order
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
try:
    from tests.st.a5.fully_distributed_within_core.atomic_minibench._runtime_contract import DistRuntimeContractMixin
except ModuleNotFoundError:
    from _runtime_contract import DistRuntimeContractMixin


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestMB4BlockWon(DistRuntimeContractMixin, SceneTestCase):
    """MIX co-ownership stress: 1C+2V tasks exercising block.won at scale."""

    RTOL = 1e-3
    ATOL = 1e-3

    RUNTIME_ENV = {"PTO_DIST_DEPSIG": "1"}

    CALLABLE = {
        "orchestration": {
            "source": "../mix_coown/kernels/orchestration/mix_coown_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "MM",
                "source": "../mix_coown/kernels/aic/kernel_mm.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 1,
                "name": "ADD_V0",
                "source": "../mix_coown/kernels/aiv/kernel_add_v0.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 2,
                "name": "ADD_V1",
                "source": "../mix_coown/kernels/aiv/kernel_add_v1.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 3,
                "name": "SUM",
                "source": "../mix_coown/kernels/aiv/kernel_sum.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "Mix3",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"num_groups": 12, "tile_size": 64},
        },
        {
            # High block count: stress block.won deposit/drain across many blocks.
            "name": "Mix24",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"num_groups": 96, "tile_size": 64},
        },
        {
            # Full-core: every block participates, maximum block.won concurrency.
            "name": "Mix36",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"num_groups": 144, "tile_size": 64},
        },
    ]

    def generate_args(self, params):
        n = params["num_groups"]
        t = params["tile_size"]
        A = torch.randn(n, t, t, dtype=torch.float32) * 0.01
        B = torch.randn(n, t, t, dtype=torch.float32) * 0.01
        Cmm = torch.zeros(n, t, t, dtype=torch.float32)
        Vfinal = torch.zeros(n, t, t, dtype=torch.float32)
        config = torch.tensor([t, 1, n, 1], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()),
            Tensor("B", B.flatten()),
            Tensor("Cmm", Cmm.flatten()),
            Tensor("Vfinal", Vfinal.flatten()),
            Tensor("config", config),
        )

    def compute_golden(self, args, params):
        n = params["num_groups"]
        t = params["tile_size"]
        A = args.A.reshape(n, t, t)
        B = args.B.reshape(n, t, t)
        Cmm = args.Cmm.reshape(n, t, t)
        Vfinal = args.Vfinal.reshape(n, t, t)
        for g in range(n):
            Cmm[g] = torch.matmul(A[g], B[g])
            Vfinal[g] = 2.0 * (A[g] + B[g])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
