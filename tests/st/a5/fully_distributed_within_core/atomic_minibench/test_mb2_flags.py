#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
 # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MB-2 flags[] cacheline clobber — concurrent flag set on shared cache lines.

Submits N independent write-only tasks (no cross-task dependencies). Each task
writes its index into a distinct slot of a single output tensor. Completion
flags are densely packed (16 int32 flags per 64B cache line), so concurrent
flag-set by different cores maximises cacheline-clobber pressure.

On A5 onboard (no HW cache coherence), the buggy path (plain store + dcci
CACHELINE_OUT) clobbers neighbouring flags → some tasks never observed as
complete → golden fails (missing writes) or hangs. The fixed path
(atomicMax in dist_set_flag) writes the true HBM word directly.

Criteria (docs MB-2):
  - All consumers complete (no hang from lost flags)
  - Golden passes (all slots written — no data corruption)
  - PTO_DIST_DEPSIG=1 signature consistent

Onboard only: sim has no cache-line concept and masks the issue.
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
try:
    from tests.st.a5.fully_distributed_within_core.atomic_minibench._runtime_contract import DistRuntimeContractMixin
except ModuleNotFoundError:
    from _runtime_contract import DistRuntimeContractMixin


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestMB2FlagsClobber(DistRuntimeContractMixin, SceneTestCase):
    """flags[] cacheline clobber: N independent tasks, dense flag packing."""

    RTOL = 0
    ATOL = 0

    RUNTIME_ENV = {"PTO_DIST_DEPSIG": "1"}

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mb2_flags_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "WRITE_INDEX",
                "source": "kernels/aiv/kernel_write_index.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
        ],
    }

    CASES = [
        {
            # 512 independent tasks = 512 flags = 32 cache lines under
            # concurrent write pressure (16 flags / 64B line).
            # a5sim is the control: sim has no cache-line concept, always
            # passes (§7: "sim 掩盖此问题"). a5 is the real test.
            "name": "Flags512",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 8},
            "params": {"num_tasks": 512},
        },
        {
            # Full-core: 36 blocks, 1024 tasks — maximum flag concurrency.
            "name": "Flags1024FullCore",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"num_tasks": 1024},
        },
    ]

    def generate_args(self, params):
        n = params["num_tasks"]
        out = torch.zeros(n, dtype=torch.float32)
        config = torch.tensor([n], dtype=torch.int64)
        return TaskArgsBuilder(Tensor("out", out), Tensor("config", config))

    def compute_golden(self, args, params):
        n = params["num_tasks"]
        for i in range(n):
            args.out[i] = float(i + 1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
