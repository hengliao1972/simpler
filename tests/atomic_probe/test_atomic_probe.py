# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Executable entry points for the A5 cache-line probe suite."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def test_cpu_cacheline_control(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    assert compiler is not None, "g++ is required for the CPU control"
    binary = tmp_path / "cpu_atomicity"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++17",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(HERE / "cpu" / "cpu_atomicity.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_cpu_nested_lambda_compiler_probe() -> None:
    subprocess.run(
        [str(HERE / "run_nested_lambda.sh"), "cpu"],
        cwd=HERE,
        check=True,
    )


def _onboard_environment(st_device_ids: list[int]) -> dict[str, str]:
    assert os.environ.get("ASCEND_HOME_PATH"), "ASCEND_HOME_PATH is required for A5 probes"
    environment = os.environ.copy()
    environment["ATOMIC_PROBE_DEVICE"] = str(int(st_device_ids[0]))
    return environment


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.timeout(900)
def test_a5_ascendc_cacheline_probes(st_device_ids: list[int]) -> None:
    subprocess.run(
        [str(HERE / "ascendc" / "_run_asc_probe.sh")],
        cwd=HERE / "ascendc",
        env=_onboard_environment(st_device_ids),
        check=True,
    )


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.timeout(900)
def test_a5_ccec_cacheline_probes(st_device_ids: list[int]) -> None:
    subprocess.run(
        [str(HERE / "ccec" / "run_all.sh")],
        cwd=HERE / "ccec",
        env=_onboard_environment(st_device_ids),
        check=True,
    )


@pytest.mark.requires_hardware
@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(1)
@pytest.mark.timeout(900)
def test_a5_ccec_nested_lambda_call_boundary_controls(st_device_ids: list[int]) -> None:
    environment = _onboard_environment(st_device_ids)
    subprocess.run(
        [str(HERE / "ccec" / "run_all.sh"), "nested_lambda_cross_tu", "build"],
        cwd=HERE / "ccec",
        env=environment,
        check=True,
    )
    for mode in ("args-runtime-read", "weak-context-materialize-0"):
        environment["ATOMIC_PROBE_MODE"] = mode
        subprocess.run(
            [str(HERE / "ccec" / "run_all.sh"), "nested_lambda_cross_tu", "run"],
            cwd=HERE / "ccec",
            env=environment,
            check=True,
        )
