#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MB-7 core_progress + run-ahead backpressure — min consistency and balance.

Reuses the benchmark_bgemm workload with fake exec times to create unbalanced
core speeds, and sweeps PTO_DIST_RUNAHEAD to verify the backpressure window
constraint holds (no deadlock, no shared-ring overflow, DEPSIG unchanged).

Criteria (docs MB-7):
  - max(local_index) - min(local_index) <= runahead_max (sampled)
  - No deadlock (slowest core advances → releases)
  - DEPSIG does not change with throttling (throttle changes WHO wins, not the
    dependency graph)

Env knobs:
  PTO_DIST_RUNAHEAD=N        — run-ahead bound (0=disable, small=tight backpressure)
  PTO_DIST_FAKE_EXEC_NS=N    — uniform fake exec time (makes all cores equal speed)
  PTO_DIST_TENSORMAP_MODE=shared — shared mode (run-ahead backpressure is meaningful)
  PTO_DIST_DEPSIG=1          — dependency-graph signature (internal assert)
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestMB7Runahead(SceneTestCase):
    """Run-ahead backpressure + load balance via bgemm at scale."""

    RTOL = 1e-3
    ATOL = 1e-3

    RUNTIME_ENV = {"PTO_DIST_DEPSIG": "1"}

    CALLABLE = {
        "orchestration": {
            "source": "../benchmark_bgemm/kernels/orchestration/bgemm_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "GEMM",
                "source": "../benchmark_bgemm/kernels/aic/kernel_gemm_tile.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": "ADD",
                "source": "../benchmark_bgemm/kernels/aiv/kernel_tile_add.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            # Default runahead (engine auto = 2*num_workers): should complete normally.
            "name": "Default",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"matmul_add_task_num": 180, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
        {
            # Tight runahead: set PTO_DIST_RUNAHEAD=4 externally to force backpressure.
            # Golden must still pass (throttle changes scheduling, not data).
            "name": "TightRunahead",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"matmul_add_task_num": 180, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
        {
            # Full-core balance: 36 blocks, 360 tasks. Run with
            #   PTO_DIST_FAKE_EXEC_NS=1000 PTO_DIST_RUNAHEAD=8
            # to verify window constraint + no deadlock at scale.
            "name": "FullCore36",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"matmul_add_task_num": 360, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
        },
    ]

    def generate_args(self, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        B = torch.randn(num_groups, grid_k, incore_loop, tile_size, tile_size, dtype=torch.float32) * 0.01
        C = torch.zeros(incore_loop * num_groups, tile_size, tile_size, dtype=torch.float32)
        config = torch.tensor([tile_size, grid_k, num_groups, incore_loop], dtype=torch.int64)
        return TaskArgsBuilder(
            Tensor("A", A.flatten()), Tensor("B", B.flatten()), Tensor("C", C.flatten()), Tensor("config", config)
        )

    def compute_golden(self, args, params):
        tile_size = params["incore_data_size"]
        incore_loop = params["incore_loop"]
        grid_k = params["grid_k"]
        num_groups = params["matmul_add_task_num"] // grid_k
        A = args.A.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        B = args.B.reshape(num_groups, grid_k, incore_loop, tile_size, tile_size)
        C = args.C.reshape(incore_loop * num_groups, tile_size, tile_size)
        C[:] = 0.0
        for group in range(num_groups):
            for k_idx in range(grid_k):
                for i in range(incore_loop):
                    C[group * incore_loop + i] += torch.matmul(A[group, k_idx, i], B[group, k_idx, i])


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
