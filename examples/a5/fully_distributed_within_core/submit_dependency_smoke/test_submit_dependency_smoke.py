#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
            "signature": [D.IN, D.INOUT, D.INOUT],
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
            {
                "func_id": 4,
                "name": "BUMP_INOUT_AIV",
                "source": "kernels/aiv/bump_inout.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 5,
                "name": "BUMP_OFFSET_AIV",
                "source": "kernels/aiv/bump_offset.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT],
            },
            {
                "func_id": 6,
                "name": "INSPECT_VIEW_AIV",
                "source": "kernels/aiv/inspect_view.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.INOUT, D.INOUT],
            },
            {
                "func_id": 7,
                "name": "INSPECT_ALLOC_AIC",
                "source": "kernels/aic/inspect_alloc.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.INOUT, D.INOUT],
            },
        ],
    }

    CASES = [
        {
            "name": "A5SimBd1N16",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5OnboardBd1N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5OnboardBd2N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 2},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5OnboardBd4N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 4},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5OnboardBd8N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 8},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5OnboardBd16N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 16},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5OnboardBd24N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5SimBd36N128",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 0},
        },
        {
            "name": "A5OnboardBd36N16",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 16, "mode": 0},
        },
        {
            "name": "A5OnboardBd36N128",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 0},
        },
        {
            "name": "A5SimBd36OverlapSubView",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 1},
        },
        {
            "name": "A5OnboardBd36OverlapSubView",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 1},
        },
        {
            "name": "A5SimBd36ExistingInoutChain",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 2},
        },
        {
            "name": "A5OnboardBd36ExistingInoutChain",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 2},
        },
        {
            "name": "A5SimBd36SubViewOnly",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 3},
        },
        {
            "name": "A5OnboardBd36SubViewOnly",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 3},
        },
        {
            "name": "A5OnboardBd1SubViewOnly",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 4096, "mode": 3},
        },
        {
            "name": "A5OnboardBd1ViewDescriptorOnly",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 4096, "mode": 20},
        },
        {
            "name": "A5OnboardBd1ArgViewDescriptorOnly",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 4096, "mode": 21},
        },
        {
            "name": "A5SimBd36ScalarOffsetOnly",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 4},
        },
        {
            "name": "A5OnboardBd36ScalarOffsetOnly",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 4},
        },
        {
            "name": "A5SimBd36InspectSubView",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 5},
        },
        {
            "name": "A5OnboardBd36InspectSubView",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 5},
        },
        {
            "name": "A5SimBd36InspectLocalCopy",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 6},
        },
        {
            "name": "A5OnboardBd36InspectLocalCopy",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 6},
        },
        {
            "name": "A5SimBd36AllocDescriptor",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 7},
        },
        {
            "name": "A5OnboardBd36AllocDescriptor",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 7},
        },
        {
            "name": "A5SimBd36AllocFillFanin",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 8},
        },
        {
            "name": "A5OnboardBd36AllocFillFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 8},
        },
        {
            "name": "A5SimBd1AllocFillFanin",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 8},
        },
        {
            "name": "A5OnboardBd1AllocFillFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 8},
        },
        {
            "name": "A5SimBd36LeftOutputFanin",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 9},
        },
        {
            "name": "A5OnboardBd36LeftOutputFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 9},
        },
        {
            "name": "A5SimBd36RightOutputFanin",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 10},
        },
        {
            "name": "A5OnboardBd36RightOutputFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 10},
        },
        {
            "name": "A5SimBd36TaskOutputViewFanin",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 15},
        },
        {
            "name": "A5OnboardBd36TaskOutputViewFanin",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 15},
        },
        {
            "name": "A5SimBd36DelayedRingFanout",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 257, "mode": 16},
        },
        {
            "name": "A5OnboardBd36DelayedRingFanout",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 257, "mode": 16},
        },
        {
            "name": "A5SimBd36AicConsumesAivOutput",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 17},
        },
        {
            "name": "A5OnboardBd36AicConsumesAivOutput",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 128, "mode": 17},
        },
        {
            "name": "A5SimBd1AicSlotCapacityWait",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 263, "mode": 18},
        },
        {
            "name": "A5OnboardBd1AicSlotCapacityWait",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 263, "mode": 18},
        },
        {
            "name": "A5SimBd1AicRingReuseStress",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 512, "mode": 28},
        },
        {
            "name": "A5OnboardBd1AicRingReuseStress",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 512, "mode": 28},
        },
        {
            "name": "A5SimBd36FaninRingReuseStress",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 512, "mode": 29},
        },
        {
            "name": "A5OnboardBd36FaninRingReuseStress",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 512, "mode": 29},
        },
        {
            "name": "A5SimBd36FaninLargeStress",
            "manual": True,
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 30},
        },
        {
            "name": "A5OnboardBd36FaninLargeStress",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 30},
        },
        {
            "name": "A5SimBd36L0TaskArgsTagPersistence",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 14},
        },
        {
            "name": "A5OnboardBd36L0TaskArgsTagPersistence",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 36},
            "params": {"n": 4096, "mode": 14},
        },
        {
            "name": "A5SimBd1AllocFillRunAhead1",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 1, "mode": 22},
        },
        {
            "name": "A5OnboardBd1AllocFillRunAhead1",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 1, "mode": 22},
        },
        {
            "name": "A5SimBd1AllocFillRunAheadDefaultHeap",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 1, "mode": 22},
        },
        {
            "name": "A5OnboardBd1AllocFillRunAheadDefaultHeap",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 1, "mode": 22},
        },
        {
            "name": "A5SimBd36AllocFillRunAhead1",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 36,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 1, "mode": 22},
        },
        {
            "name": "A5OnboardBd36AllocFillRunAhead1",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 36,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 1, "mode": 22},
        },
        {
            "name": "A5SimBd1AllocFillRunAhead16",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 16, "mode": 22},
        },
        {
            "name": "A5OnboardBd1AllocFillRunAhead16",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 16, "mode": 22},
        },
        {
            "name": "A5SimBd1AllocFillRunAhead32",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 32, "mode": 22},
        },
        {
            "name": "A5OnboardBd1AllocFillRunAhead32",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 32, "mode": 22},
        },
        {
            "name": "A5SimBd1AllocFillRunAhead67",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 67, "mode": 22},
        },
        {
            "name": "A5OnboardBd1AllocFillRunAhead67",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 67, "mode": 22},
        },
        {
            "name": "A5SimBd1AllocFillRunAhead128",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 128, "mode": 22},
        },
        {
            "name": "A5OnboardBd1AllocFillRunAhead128",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 128, "mode": 22},
        },
        {
            "name": "A5SimBd1AllocOnly1",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 1, "mode": 23},
        },
        {
            "name": "A5OnboardBd1AllocOnly1",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 1, "mode": 23},
        },
        {
            "name": "A5SimBd1TailAicOutput",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 24},
        },
        {
            "name": "A5OnboardBd1TailAicOutput",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 24},
        },
        {
            "name": "A5SimBd1DeadAicOutput",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 25},
        },
        {
            "name": "A5OnboardBd1DeadAicOutput",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 25},
        },
        {
            "name": "A5SimBd1AllocDescriptorInSlot",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 26},
        },
        {
            "name": "A5OnboardBd1AllocDescriptorInSlot",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 26},
        },
        {
            "name": "A5SimBd1AllocDescriptorInAicSlot",
            "platforms": ["a5sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 27},
        },
        {
            "name": "A5OnboardBd1AllocDescriptorInAicSlot",
            "manual": True,
            "platforms": ["a5"],
            "config": {"aicpu_thread_num": 4, "block_dim": 1},
            "params": {"n": 128, "mode": 27},
        },
        {
            "name": "A5SimBd1AllocHeapBackPressure",
            "platforms": ["a5sim"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 67, "mode": 19},
        },
        {
            "name": "A5OnboardBd1AllocHeapBackPressure",
            "manual": True,
            "platforms": ["a5"],
            "config": {
                "aicpu_thread_num": 4,
                "block_dim": 1,
                "runtime_env": {"ring_heap": 68 * 1024},
            },
            "params": {"n": 67, "mode": 19},
        },
    ]

    def generate_args(self, params):
        n = int(params["n"])
        x = torch.arange(n, dtype=torch.float32)
        y = torch.full((n,), -1.0, dtype=torch.float32)
        dump = torch.full((n,), -1.0, dtype=torch.float32)
        return TaskArgsBuilder(
            Tensor("input", x),
            Tensor("output", y),
            Tensor("dump", dump),
            Scalar("n", n),
            Scalar("mode", int(params.get("mode", 0))),
        )

    def compute_golden(self, args, params):  # noqa: PLR0912 - smoke cases intentionally share one golden dispatcher
        mode = int(params.get("mode", 0))
        if mode == 1:
            n = int(params["n"])
            args.output[:] = args.input + 7.0
            start = n // 4
            end = start + n // 2
            args.output[start:end] += 1.0
            return
        if mode == 2:
            args.output[:] = args.input + 8.0
            return
        if mode in {3, 4}:
            n = int(params["n"])
            start = n // 4
            end = start + n // 2
            args.output[:] = -1.0
            args.output[start:end] += 1.0
            return
        if mode == 5:
            args.output[:] = -1.0
            args.output[:4] = 1.0
            args.output[80:110] = 1.0
            args.dump[:3] = 1.0
            return
        if mode == 14:
            args.output[:] = -1.0
            args.output[:12] = 1.0
            return
        if mode == 6:
            args.output[:] = -1.0
            args.output[:4] = 1.0
            args.dump[:3] = 1.0
            return
        if mode == 7:
            args.output[:] = -1.0
            args.output[:10] = 1.0
            args.output[16:26] = 1.0
            return
        if mode == 8:
            args.output[:] = (args.input + 7.0) * 3.0 + 8.0
            return
        if mode == 9:
            args.output[:] = (args.input * 2.0 + 3.0) * 3.0 + 8.0
            return
        if mode == 10:
            args.output[:] = (args.input * 3.0 + 5.0) * 3.0 + 8.0
            return
        if mode == 15:
            n = int(params["n"])
            start = n // 4
            end = start + n // 2
            args.output[:] = -1.0
            args.output[: n // 2] = (args.input[start:end] * 2.0 + 3.0) * 3.0 + 8.0
            return
        if mode == 16:
            sub_n = int(params["n"]) // 8
            args.output[:] = -1.0
            expected = (args.input[:sub_n] + 7.0) * 3.0 + 8.0
            args.output[: 6 * sub_n] = expected.repeat(6)
            return
        if mode == 17:
            args.output[:] = args.input * 3.0 + 12.0
            return
        if mode == 18:
            sub_n = 32
            args.output[:] = -1.0
            expected = args.input[:sub_n] * 3.0 + 12.0
            args.output[: 6 * sub_n] = expected.repeat(6)
            return
        if mode == 28:
            sub_n = 32
            args.output[:] = -1.0
            expected = args.input[:sub_n] * 3.0 + 12.0
            args.output[: 12 * sub_n] = expected.repeat(12)
            return
        if mode == 29:
            sub_n = 32
            args.output[:] = -1.0
            expected = args.input[:sub_n] * 6.0 + 23.0
            args.output[: 12 * sub_n] = expected.repeat(12)
            return
        if mode == 30:
            sub_n = 32
            args.output[:] = -1.0
            expected = args.input[:sub_n] * 6.0 + 23.0
            args.output[: 96 * sub_n] = expected.repeat(96)
            return
        if mode == 19:
            args.output[:] = (args.input * 2.0 + 3.0) * 3.0 + 8.0
            return
        if mode == 20:
            args.output[:] = -1.0
            args.output[:8] = 1.0
            return
        if mode == 21:
            args.output[:] = -1.0
            args.output[[0, 1, 2, 3, 4, 7, 8, 9, 10]] = 1.0
            args.output[[5, 6]] = 0.0
            return
        if mode == 22:
            args.output[:] = -1.0
            return
        if mode == 23:
            args.output[:] = -1.0
            return
        if mode == 24:
            args.output[:] = args.input + 7.0
            return
        if mode == 25:
            args.output[:] = -1.0
            return
        if mode == 26:
            args.output[:] = -1.0
            args.output[0] = 0.0
            args.output[1:4] = 1.0
            args.dump[:3] = 1.0
            return
        if mode == 27:
            args.output[:] = -1.0
            args.output[:10] = 1.0
            args.dump[:3] = 1.0
            return
        args.output[:] = args.input * 6.0 + 23.0


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
