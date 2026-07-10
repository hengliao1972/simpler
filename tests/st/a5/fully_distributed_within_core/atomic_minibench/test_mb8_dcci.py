#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
 # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MB-8 Coherent<T>/dcci seam — cross-core publish/observe correctness.

A minimal producer→consumer chain repeated N rounds. Each round the producer
writes a round-dependent pattern into a buffer; the consumer reads it (via
runtime dependency) and writes result = buf + 1. If the dcci publish/observe
seam is broken on A5 onboard (producer doesn't flush, or consumer doesn't
invalidate), the consumer reads stale data and golden fails.

This is the foundational seam test — all other onboard minibench failures
(MB-1/2/3/4/5/6/7) trace back to this seam if the root cause is cache
coherence rather than logic.

Criteria (docs MB-8):
  - Consumer reads the exact producer-written data every round (no stale cache)
  - Golden passes (result[i] = round*100 + i + 1)
  - No "flag arrives, data stale" scenario

Onboard only: sim compiles dcci to no-op (always passes as baseline).
"""

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

ELEMS = 16


@scene_test(level=2, runtime="fully_distributed_within_core")
class TestMB8DcciSeam(SceneTestCase):
    """dcci seam: producer→consumer cross-core publish/observe."""

    RTOL = 0
    ATOL = 0

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/mb8_dcci_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.INOUT, D.INOUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "PRODUCER",
                "source": "kernels/aiv/kernel_producer.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 1,
                "name": "CONSUMER",
                "source": "kernels/aiv/kernel_consumer.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            # 50 rounds of producer→consumer. Each round writes a different
            # pattern, so stale-cache reads produce wrong values immediately.
            # a5sim is the control: sim compiles dcci to no-op, always passes
            # (§7: "sim 恒绿作对照"). a5 is the real test.
            "name": "Rounds50",
            "platforms": ["a5sim", "a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 4},
            "params": {"num_rounds": 50},
        },
        {
            # High block count: more cores → more cache reuse → more stale hits.
            "name": "Rounds100FullCore",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"num_rounds": 100},
        },
    ]

    def generate_args(self, params):
        n = params["num_rounds"]
        buf = torch.zeros(ELEMS, dtype=torch.float32)
        result = torch.zeros(ELEMS, dtype=torch.float32)
        config = torch.tensor([n], dtype=torch.int64)
        return TaskArgsBuilder(Tensor("buf", buf), Tensor("result", result), Tensor("config", config))

    def compute_golden(self, args, params):
        n = params["num_rounds"]
        # After N rounds, buf holds the last round's pattern,
        # and result holds the last round's consumer output.
        for i in range(ELEMS):
            args.buf[i] = float((n - 1) * 100 + i)
            args.result[i] = float((n - 1) * 100 + i + 1)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
