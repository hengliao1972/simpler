# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the install-time runtime pre-build entry point."""

import sys

import pytest


def test_shared_environment_does_not_change_install_default(monkeypatch):
    from simpler_setup import build_runtimes  # noqa: PLC0415

    runtimes = {
        "a2a3": ["fully_distributed_within_core", "host_build_graph", "tensormap_and_ringbuffer"],
        "a5": ["fully_distributed_within_core", "host_build_graph"],
    }
    monkeypatch.setattr(build_runtimes, "discover_runtimes", lambda arch: runtimes[arch])
    monkeypatch.setenv("PTO_FDWIC_TENSORMAP_MODE", "shared")

    tasks = build_runtimes._collect_runtime_build_tasks(["a2a3sim", "a5sim"], None)

    assert tasks == [
        ("a2a3sim", "fully_distributed_within_core", "private"),
        ("a2a3sim", "host_build_graph", "private"),
        ("a2a3sim", "tensormap_and_ringbuffer", "private"),
        ("a5sim", "fully_distributed_within_core", "private"),
        ("a5sim", "host_build_graph", "private"),
    ]


def test_explicit_modes_collect_both_fdwic_artifact_families_once(monkeypatch):
    from simpler_setup import build_runtimes  # noqa: PLC0415

    monkeypatch.setattr(
        build_runtimes,
        "discover_runtimes",
        lambda _arch: ["fully_distributed_within_core", "host_build_graph"],
    )
    monkeypatch.setenv("PTO_FDWIC_TENSORMAP_MODE", "shared")

    tasks = build_runtimes._collect_runtime_build_tasks(
        ["a5sim"],
        ["private", "shared", "private"],
    )

    assert tasks == [
        ("a5sim", "fully_distributed_within_core", "private"),
        ("a5sim", "fully_distributed_within_core", "shared"),
        ("a5sim", "host_build_graph", "private"),
    ]


@pytest.mark.parametrize("modes", [[], ["typo"], "shared"])
def test_explicit_modes_reject_invalid_api_values(modes):
    from simpler_setup import build_runtimes  # noqa: PLC0415

    with pytest.raises(ValueError, match="fdwic_tensormap_modes|Invalid FDWIC TensorMap mode"):
        build_runtimes._normalize_fdwic_tensormap_modes(modes)


def test_build_all_passes_mode_only_to_matching_runtime(tmp_path, monkeypatch):
    from simpler_setup import build_runtimes  # noqa: PLC0415

    calls = []

    class _FakeRuntimeBuilder:
        _LIB_DIR = None
        _CACHE_DIR = None

        def __init__(self, platform, fdwic_tensormap_mode):
            self.platform = platform
            self.mode = fdwic_tensormap_mode

        def ensure_simpler_log(self, build):
            calls.append(("simpler_log", self.platform, self.mode, build))

        def ensure_sim_context(self, build):
            calls.append(("sim_context", self.platform, self.mode, build))

        def get_binaries(self, runtime_name, build):
            calls.append((runtime_name, self.platform, self.mode, build))

    monkeypatch.setattr(build_runtimes, "RuntimeBuilder", _FakeRuntimeBuilder)
    monkeypatch.setattr(
        build_runtimes,
        "discover_runtimes",
        lambda _arch: ["fully_distributed_within_core", "host_build_graph"],
    )
    monkeypatch.setenv("PTO_FDWIC_TENSORMAP_MODE", "shared")

    build_runtimes.build_all(
        lib_dir=tmp_path / "lib",
        cache_dir=tmp_path / "cache",
        platforms=["a5sim"],
        fdwic_tensormap_modes=["private", "shared"],
    )

    assert ("fully_distributed_within_core", "a5sim", "private", True) in calls
    assert ("fully_distributed_within_core", "a5sim", "shared", True) in calls
    assert calls.count(("host_build_graph", "a5sim", "private", True)) == 1
    assert ("simpler_log", "a5sim", "private", True) in calls
    assert ("sim_context", "a5sim", "private", True) in calls


def test_cli_forwards_repeatable_fdwic_modes(tmp_path, monkeypatch):
    from simpler_setup import build_runtimes  # noqa: PLC0415

    observed = {}
    monkeypatch.setattr(build_runtimes, "build_all", lambda **kwargs: observed.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_runtimes.py",
            "--lib-dir",
            str(tmp_path / "lib"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--platforms",
            "a5sim",
            "--fdwic-tensormap",
            "private",
            "--fdwic-tensormap",
            "shared",
        ],
    )

    build_runtimes.main()

    assert observed["platforms"] == ["a5sim"]
    assert observed["fdwic_tensormap_modes"] == ["private", "shared"]
