# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Regression: SceneTestCase compile cache must release its ChipCallables.

The session-lifetime ``_compile_cache`` in ``simpler_setup.scene_test`` used
to hold every compiled ``ChipCallable`` until Python interpreter shutdown.
At shutdown the nanobind module destructor can run before module globals
are cleared, which surfaces as ``nanobind: leaked N instances of type
_task_interface.ChipCallable`` on stderr. ``clear_compile_cache`` (invoked
from ``pytest_sessionfinish``) drops the cache and forces GC so those
instances die while the extension is still live.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from _task_interface import ArgDirection, ChipCallable  # pyright: ignore[reportMissingImports]

from conftest import _configure_fdwic_profile

# ``simpler_setup/__init__.py`` re-exports the ``scene_test`` *decorator*,
# which shadows the submodule attribute when accessed via ``simpler_setup``.
# Importing the names directly from the submodule avoids that ambiguity.
from simpler_setup.scene_test import (
    _aicore_override_cache,
    _assert_fdwic_perf_clock_elf,
    _assert_fdwic_submit_pmu_elf,
    _assert_fdwic_swimlane_elf,
    _compile_cache,
    _convert_case_swimlane,
    _fdwic_compile_definitions,
    _fdwic_profile,
    _profiled_cache_key,
    _run_swimlane_converter,
    clear_compile_cache,
    run_class_cases,
)

_scene_test_module = importlib.import_module("simpler_setup.scene_test")


def _build_chip_callable(tag: str) -> ChipCallable:
    return ChipCallable.build(
        signature=[ArgDirection.IN],
        func_name=tag,
        binary=b"\x00" * 16,
        children=[],
    )


def test_clear_compile_cache_drops_cached_chip_callables():
    """clear_compile_cache empties the dict so nanobind instances can die.

    The leak this guards against is ``_compile_cache`` retaining every
    compiled ``ChipCallable`` for the full pytest session. The regression
    surface is therefore "dict still has entries after the cleanup call"
    — if someone breaks ``clear_compile_cache`` (forgets the ``.clear()``,
    swaps the cache key schema, introduces a secondary holder that the
    cleanup doesn't know about), this assertion fails.
    """
    _compile_cache.clear()
    _aicore_override_cache.clear()
    for i in range(3):
        _compile_cache[("t", "plat", f"rt{i}")] = _build_chip_callable(f"n{i}")
    _aicore_override_cache[("t", "plat", "rt0", "none")] = Path("/tmp/fake-aicore.o")
    assert len(_compile_cache) == 3
    assert len(_aicore_override_cache) == 1

    clear_compile_cache()

    assert _compile_cache == {}
    assert _aicore_override_cache == {}


def test_fdwic_profile_partitions_compile_cache(monkeypatch):
    """Normal and both private profiles must use distinct AICore overrides."""
    base = ("Case", "a5", "fully_distributed_within_core")

    monkeypatch.delenv("PTO_FDWIC_PROFILE", raising=False)
    assert _fdwic_profile() == "none"
    assert _profiled_cache_key(base) == (*base, "none")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "perf-clock")
    assert _fdwic_profile() == "perf-clock"
    assert _profiled_cache_key(base) == (*base, "perf-clock")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-none")
    assert _fdwic_profile() == "submit-pmu-none"
    assert _profiled_cache_key(base) == (*base, "submit-pmu-none")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-arg-build")
    assert _fdwic_profile() == "submit-pmu-arg-build"
    assert _profiled_cache_key(base) == (*base, "submit-pmu-arg-build")


def test_fdwic_private_profiles_have_isolated_compile_definitions():
    assert _fdwic_compile_definitions("none") is None
    assert _fdwic_compile_definitions("perf-clock") == [
        "PTO_FDWIC_PERF_CLOCK=1",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-none") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-arg-build") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=1",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]


def test_fdwic_profile_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("PTO_FDWIC_PROFILE", "typo")

    with pytest.raises(ValueError, match="Unsupported PTO_FDWIC_PROFILE"):
        _fdwic_profile()


def test_perf_clock_elf_gate_accepts_required_symbol_without_observers(monkeypatch, tmp_path):
    """The final image must keep its marker and remove observer slow paths."""
    symbol_table = "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 _Z30dist_perf_clock_expect_submitsj\n"
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    _assert_fdwic_perf_clock_elf(tmp_path / "aicore_kernel.o")


@pytest.mark.parametrize(
    ("symbol_table", "message"),
    [
        ("", "missing defined perf-clock marker"),
        (
            "12: 0000000000000000 0 FUNC GLOBAL DEFAULT UND _Z30dist_perf_clock_expect_submitsj\n",
            "missing defined perf-clock marker",
        ),
        (
            "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 "
            "_Z30dist_perf_clock_expect_submitsj\n"
            "42: 0000000000000048 8 OBJECT LOCAL DEFAULT 17 g_fdwic_atomic_record_index\n",
            r"profiling symbol\(s\) still present",
        ),
        (
            "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 "
            "_Z30dist_perf_clock_expect_submitsj\n"
            "43: 0000000000000080 8 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n",
            r"profiling symbol\(s\) still present",
        ),
        (
            "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 "
            "_Z30dist_perf_clock_expect_submitsj\n"
            "44: 0000000000000088 8 OBJECT LOCAL DEFAULT 17 g_fdwic_submit_pmu_reg_base\n",
            r"profiling symbol\(s\) still present",
        ),
    ],
)
def test_perf_clock_elf_gate_rejects_incomplete_image(monkeypatch, tmp_path, symbol_table, message):
    """A missing marker or residual observer state must fail closed."""
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    with pytest.raises(RuntimeError, match=message):
        _assert_fdwic_perf_clock_elf(tmp_path / "aicore_kernel.o")


def test_perf_clock_elf_gate_reports_readelf_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="not an ELF"),
    )

    with pytest.raises(RuntimeError, match="readelf failed.*not an ELF"):
        _assert_fdwic_perf_clock_elf(tmp_path / "aicore_kernel.o")


def test_submit_pmu_elf_gate_accepts_only_whole_window_observer(monkeypatch, tmp_path):
    symbol_table = (
        "10: 0000000000001000 64 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
        "11: 0000000000001040 96 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    _assert_fdwic_submit_pmu_elf(tmp_path / "aicore_kernel.o", "submit-pmu-none")


def test_submit_pmu_phase_elf_gate_requires_running_shadow_reader(monkeypatch, tmp_path):
    symbol_table = (
        "10: 0000000000001000 64 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
        "11: 0000000000001040 96 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
        "12: 00000000000010a0 64 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_phase_read_shadow_counters\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    _assert_fdwic_submit_pmu_elf(tmp_path / "aicore_kernel.o", "submit-pmu-arg-build")


@pytest.mark.parametrize(
    ("symbol_table", "message"),
    [
        ("", "missing defined submit-pmu marker"),
        (
            "1: 0 0 FUNC GLOBAL DEFAULT UND dist_submit_pmu_expect_submits\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n",
            "missing defined submit-pmu marker",
        ),
        (
            "1: 0 1 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
            "3: 0 1 FUNC LOCAL DEFAULT 1 fdwic_swimlane_detail_record_atomic\n",
            r"unrelated profiling symbol\(s\) still present",
        ),
        (
            "1: 0 1 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 get_aicore_pmu_ring\n",
            r"unrelated profiling symbol\(s\) still present",
        ),
        (
            "1: 0 1 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 get_aicore_pmu_reg_base\n",
            r"unrelated profiling symbol\(s\) still present",
        ),
        (
            "1: 0 1 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 dist_perf_clock_expect_submits\n",
            r"unrelated profiling symbol\(s\) still present",
        ),
    ],
)
def test_submit_pmu_elf_gate_rejects_incomplete_or_mixed_image(monkeypatch, tmp_path, symbol_table, message):
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    with pytest.raises(RuntimeError, match=message):
        _assert_fdwic_submit_pmu_elf(tmp_path / "aicore_kernel.o", "submit-pmu-none")


def test_submit_pmu_phase_elf_gate_rejects_missing_running_shadow_reader(monkeypatch, tmp_path):
    symbol_table = (
        "10: 0000000000001000 64 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
        "11: 0000000000001040 96 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    with pytest.raises(RuntimeError, match="missing defined submit-pmu marker"):
        _assert_fdwic_submit_pmu_elf(tmp_path / "aicore_kernel.o", "submit-pmu-arg-build")


def test_swimlane_elf_gate_accepts_merged_phase_atomic_observer(monkeypatch, tmp_path):
    symbol_table = (
        "2796: 0000000000012d8c 4080 FUNC LOCAL DEFAULT 1 "
        "_ZN12_GLOBAL__N_131fdwic_atomic_poll_boundary_slowEm\n"
        "3010: 0000000000013d7c 344 FUNC LOCAL DEFAULT 1 "
        "_ZN12_GLOBAL__N_135fdwic_swimlane_detail_record_atomicEv\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    _assert_fdwic_swimlane_elf(tmp_path / "aicore_kernel.o")


@pytest.mark.parametrize(
    ("symbol_table", "message"),
    [
        ("", "missing defined swimlane observer"),
        (
            "1: 0 1 FUNC LOCAL DEFAULT 1 fdwic_atomic_poll_boundary_slow\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_swimlane_detail_record_atomic\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 dist_perf_clock_expect_submits\n",
            "private-profile symbol.*leaked",
        ),
        (
            "1: 0 1 FUNC LOCAL DEFAULT 1 fdwic_atomic_poll_boundary_slow\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_swimlane_detail_record_atomic\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n",
            "private-profile symbol.*leaked",
        ),
        (
            "1: 0 1 FUNC LOCAL DEFAULT 1 fdwic_atomic_poll_boundary_slow\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_swimlane_detail_record_atomic\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 get_fdwic_submit_pmu_reg_base\n",
            "private-profile symbol.*leaked",
        ),
    ],
)
def test_swimlane_elf_gate_rejects_wrong_image(monkeypatch, tmp_path, symbol_table, message):
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    with pytest.raises(RuntimeError, match=message):
        _assert_fdwic_swimlane_elf(tmp_path / "aicore_kernel.o")


class _FakePytestConfig:
    def __init__(self, **options):
        self.options = {
            "--fdwic-profile": "submit-pmu-none",
            "--platform": "a5",
            "--runtime": "fully_distributed_within_core",
            "--level": 2,
            "--rounds": 1,
            **options,
        }

    def getoption(self, option, default=None):
        return self.options.get(option, default)


def test_submit_pmu_profile_publishes_environment(monkeypatch):
    monkeypatch.delenv("PTO_FDWIC_PROFILE", raising=False)

    _configure_fdwic_profile(_FakePytestConfig())

    assert _fdwic_profile() == "submit-pmu-none"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"--platform": "a5sim"}, "requires --platform a5"),
        ({"--runtime": "tensormap_and_ringbuffer"}, "only supports runtime fully_distributed_within_core"),
        ({"--level": 3}, "only supports SceneTest level 2"),
        ({"--rounds": 2}, "requires --rounds 1"),
    ],
)
def test_submit_pmu_profile_rejects_wrong_execution_scope(options, message):
    with pytest.raises(pytest.UsageError, match=message):
        _configure_fdwic_profile(_FakePytestConfig(**options))


@pytest.mark.parametrize(
    "option",
    [
        "--enable-l2-swimlane",
        "--dump-args",
        "--enable-pmu",
        "--enable-dep-gen",
        "--enable-scope-stats",
        "--enable-device-log-timing",
        "--enable-swimlane-overhead",
        "--use-example-exec-time",
    ],
)
def test_submit_pmu_profile_rejects_other_diagnostics(option):
    with pytest.raises(pytest.UsageError, match=option):
        _configure_fdwic_profile(_FakePytestConfig(**{option: 1}))


def test_strict_fdwic_v4_converter_requires_closure_artifacts(monkeypatch, tmp_path):
    raw = tmp_path / "l2_swimlane_records.json"
    raw.write_text("{}")
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="converted", stderr=""),
    )

    with pytest.raises(RuntimeError, match="did not publish required artifact"):
        _run_swimlane_converter(input_path=raw, strict_fdwic_v4=True)

    (tmp_path / "merged_swimlane.json").write_text("{}")
    (tmp_path / "swimlane_exclusive_analysis.json").write_text("{}")
    _run_swimlane_converter(input_path=raw, strict_fdwic_v4=True)


def test_strict_fdwic_v4_converter_propagates_validation_failure(monkeypatch, tmp_path):
    raw = tmp_path / "l2_swimlane_records.json"
    raw.write_text("{}")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="integer closure failed")

    monkeypatch.setattr(_scene_test_module.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="closure validation failed.*integer closure failed"):
        _run_swimlane_converter(input_path=raw, strict_fdwic_v4=True)


def test_strict_fdwic_v4_case_requires_raw_artifact(tmp_path):
    with pytest.raises(RuntimeError, match="required FDWIC schema-v4 raw artifact was not produced"):
        _convert_case_swimlane("Case", tmp_path, strict_fdwic_v4=True)


def test_device_failure_keeps_original_error_and_disables_strict_conversion(monkeypatch, tmp_path):
    class FailingCase:
        _st_level = 2
        _st_runtime = "fully_distributed_within_core"

        @staticmethod
        def _run_and_validate(*args, **kwargs):
            raise ValueError("device execution failed")

    worker = SimpleNamespace(_config={"platform": "a5", "device_id": 0})
    strict_values = []
    monkeypatch.setattr(_scene_test_module, "_build_output_prefix", lambda _label: tmp_path)
    monkeypatch.setattr(
        _scene_test_module,
        "_convert_case_swimlane",
        lambda *args, strict_fdwic_v4=False, **kwargs: strict_values.append(strict_fdwic_v4),
    )

    with pytest.raises(ValueError, match="device execution failed"):
        run_class_cases(
            worker,
            FailingCase(),
            [{"name": "Case1", "config": {}, "params": {}}],
            callable_obj=None,
            sub_handles={},
            rounds=1,
            skip_golden=False,
            enable_l2_swimlane=4,
            enable_dump_args=False,
            enable_pmu=0,
            enable_dep_gen=False,
            enable_scope_stats=False,
        )

    assert strict_values == [False]
