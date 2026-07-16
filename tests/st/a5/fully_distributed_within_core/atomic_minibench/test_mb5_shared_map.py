#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
 # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MB-5 shared TensorMap — serialized append + seq ABA + private==shared DEPSIG.

Runs the BGEMM workload in shared TensorMap mode (PTO_DIST_TENSORMAP_MODE=shared).
In shared mode, only the winning core appends to the global TensorMap ring;
in private mode, every core maintains its own replica. The dependency graph
(DEPSIG) must be bit-identical between the two modes — if shared mode produces
a different DEPSIG, the append sequencer or ABA guard is broken.

The UT (test_dist_atomic_mb5_shared_map.cpp) tests the data-structure logic on
host with std::thread. This ST tests the real runtime on hardware: if the
shared-mode ring overflows or the seq ABA guard fails, golden comparison fails
(wrong producer resolved → wrong data) or the run hangs.

Criteria (docs MB-5):
  - Golden passes in shared mode (correct producer resolution)
  - PTO_DIST_DEPSIG=1 signature matches private mode (run without
    PTO_DIST_TENSORMAP_MODE to get the private reference)
  - PTO_DIST_OVERHEAD=1 reports inserts ≈ D (single-ring), not C×D (per-core)

Env knobs:
  PTO_DIST_TENSORMAP_MODE=shared  — shared TensorMap mode (the subject under test)
  PTO_DIST_DEPSIG=1               — dependency-graph signature
  PTO_DIST_OVERHEAD=1             — insert/lookup/scan counters
  PTO_DIST_RUNAHEAD=N             — tight runahead to stress the shared ring (manual)
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
try:
    from tests.st.a5.fully_distributed_within_core.atomic_minibench._runtime_contract import DistRuntimeContractMixin
except ModuleNotFoundError:
    from _runtime_contract import DistRuntimeContractMixin


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestMB5SharedMap(DistRuntimeContractMixin, SceneTestCase):
    """Shared TensorMap: BGEMM in shared mode, DEPSIG must match private."""

    RTOL = 1e-3
    ATOL = 1e-3
    DIST_REQUIRE_TMOPS_ON_SIM = True

    RUNTIME_ENV = {
        "PTO_DIST_TENSORMAP_MODE": "shared",
        "PTO_DIST_DEPSIG": "1",
        "PTO_DIST_OVERHEAD": "1",
    }

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
            # Private is the reference run for the identical graph.  The
            # runtime contract records its DEPSIG/TMOPS instead of trusting the
            # numerical golden alone.
            "name": "PrivateReference",
            "oracle_group": "bgemm180",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"matmul_add_task_num": 180, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
            "runtime_env": {"PTO_DIST_TENSORMAP_MODE": "private"},
        },
        {
            # Standard BGEMM in shared mode: the mode marker must say shared,
            # and its DEPSIG must match PrivateReference.
            "name": "Shared",
            "oracle_group": "bgemm180",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {"matmul_add_task_num": 180, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
            "runtime_env": {"PTO_DIST_TENSORMAP_MODE": "shared"},
        },
        {
            # High block count: more cores appending to the shared ring,
            # maximum sequencer contention + seq ABA pressure.
            "name": "Shared24",
            "oracle_group": "bgemm360",
            "manual": True,
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"matmul_add_task_num": 360, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
            "runtime_env": {"PTO_DIST_TENSORMAP_MODE": "shared", "PTO_DIST_RUNAHEAD": "4"},
        },
        {
            # Tight runahead: stresses the shared ring's live window.
            # Inject PTO_DIST_RUNAHEAD=4 for this case to force backpressure.
            "name": "SharedTightRunahead",
            "oracle_group": "bgemm360",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"matmul_add_task_num": 360, "incore_data_size": 128, "incore_loop": 4, "grid_k": 2},
            "runtime_env": {
                "PTO_DIST_TENSORMAP_MODE": "shared",
                "PTO_DIST_RUNAHEAD": "4",
                "PTO_DIST_FAKE_EXEC_NS": "1000",
            },
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
