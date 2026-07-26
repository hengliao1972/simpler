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
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from _task_interface import ArgDirection, ChipCallable  # pyright: ignore[reportMissingImports]

from conftest import _configure_fdwic_profile, _configure_fdwic_tensormap

# ``simpler_setup/__init__.py`` re-exports the ``scene_test`` *decorator*,
# which shadows the submodule attribute when accessed via ``simpler_setup``.
# Importing the names directly from the submodule avoids that ambiguity.
from simpler_setup.scene_test import (
    _aicore_override_cache,
    _assert_fdwic_perf_clock_elf,
    _assert_fdwic_submit_pmu_elf,
    _assert_fdwic_submit_pmu_host_elf,
    _assert_fdwic_swimlane_elf,
    _compile_cache,
    _convert_case_swimlane,
    _fdwic_build_identity_cache,
    _fdwic_compile_definitions,
    _fdwic_profile,
    _fdwic_tensormap_compile_definitions,
    _fdwic_tensormap_mode,
    _profiled_cache_key,
    _render_case_fdwic_submit_pmu,
    _run_swimlane_converter,
    _validate_case_fdwic_perf_clock,
    _validate_fdwic_tensormap_test_classes,
    clear_compile_cache,
    maybe_build_aicore_override,
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
    _fdwic_build_identity_cache.clear()
    for i in range(3):
        _compile_cache[("t", "plat", f"rt{i}")] = _build_chip_callable(f"n{i}")
    _aicore_override_cache[("t", "plat", "rt0", "private", "none")] = Path("/tmp/fake-aicore.o")
    _fdwic_build_identity_cache[("t", "plat", "rt0", "private", "submit-pmu-none")] = object()
    assert len(_compile_cache) == 3
    assert len(_aicore_override_cache) == 1
    assert len(_fdwic_build_identity_cache) == 1

    clear_compile_cache()

    assert _compile_cache == {}
    assert _aicore_override_cache == {}
    assert _fdwic_build_identity_cache == {}


def test_fdwic_profile_partitions_compile_cache(monkeypatch):
    """TensorMap mode and every evidence profile must partition AICore images."""
    base = ("Case", "a5", "fully_distributed_within_core")

    monkeypatch.delenv("PTO_FDWIC_TENSORMAP_MODE", raising=False)
    monkeypatch.delenv("PTO_FDWIC_PROFILE", raising=False)
    assert _fdwic_profile() == "none"
    assert _profiled_cache_key(base) == (*base, "private", "none")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "perf-clock")
    assert _fdwic_profile() == "perf-clock"
    assert _profiled_cache_key(base) == (*base, "private", "perf-clock")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "perf-clock-kernel")
    assert _fdwic_profile() == "perf-clock-kernel"
    assert _profiled_cache_key(base) == (*base, "private", "perf-clock-kernel")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-none")
    assert _fdwic_profile() == "submit-pmu-none"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-none")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-arg-build")
    assert _fdwic_profile() == "submit-pmu-arg-build"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-arg-build")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-empty-bracket")
    assert _fdwic_profile() == "submit-pmu-empty-bracket"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-empty-bracket")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-materialize")
    assert _fdwic_profile() == "submit-pmu-materialize"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-materialize")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-claim")
    assert _fdwic_profile() == "submit-pmu-claim"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-claim")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-register")
    assert _fdwic_profile() == "submit-pmu-register"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-register")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-submit-transition")
    assert _fdwic_profile() == "submit-pmu-submit-transition"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-submit-transition")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-efdrain-control")
    assert _fdwic_profile() == "submit-pmu-efdrain-control"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-efdrain-control")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-prepare-map")
    assert _fdwic_profile() == "submit-pmu-prepare-map"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-prepare-map")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-fanin")
    assert _fdwic_profile() == "submit-pmu-fanin"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-fanin")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-winner-build-control")
    assert _fdwic_profile() == "submit-pmu-winner-build-control"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-winner-build-control")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-alloc-complete-control")
    assert _fdwic_profile() == "submit-pmu-alloc-complete-control"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-alloc-complete-control")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-loser-replay")
    assert _fdwic_profile() == "submit-pmu-loser-replay"
    assert _profiled_cache_key(base) == (*base, "private", "submit-pmu-loser-replay")

    monkeypatch.setenv("PTO_FDWIC_TENSORMAP_MODE", "shared")
    assert _profiled_cache_key(base) == (*base, "shared", "submit-pmu-loser-replay")


def test_fdwic_tensormap_mode_and_compile_definition_contract(monkeypatch):
    monkeypatch.delenv("PTO_FDWIC_TENSORMAP_MODE", raising=False)
    assert _fdwic_tensormap_mode() == "private"
    assert _fdwic_tensormap_compile_definitions("a5", "fully_distributed_within_core") == ["PTO_FDWIC_SHARED_MAP=0"]

    monkeypatch.setenv("PTO_FDWIC_TENSORMAP_MODE", "shared")
    assert _fdwic_tensormap_mode() == "shared"
    assert _fdwic_tensormap_compile_definitions("a5sim", "fully_distributed_within_core") == ["PTO_FDWIC_SHARED_MAP=1"]
    with pytest.raises(ValueError, match="only supported"):
        _fdwic_tensormap_compile_definitions("a5", "host_build_graph")

    monkeypatch.setenv("PTO_FDWIC_TENSORMAP_MODE", "typo")
    with pytest.raises(ValueError, match="Unsupported PTO_FDWIC_TENSORMAP_MODE"):
        _fdwic_tensormap_mode()


def test_fdwic_evidence_profiles_have_isolated_compile_definitions():
    assert _fdwic_compile_definitions("none") is None
    assert _fdwic_compile_definitions("perf-clock") == [
        "PTO_FDWIC_PERF_CLOCK=1",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("perf-clock-kernel") == [
        "PTO_FDWIC_PERF_CLOCK=1",
        "PTO_FDWIC_PERF_CLOCK_KERNEL=1",
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
    assert _fdwic_compile_definitions("submit-pmu-empty-bracket") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=2",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-materialize") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=3",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-claim") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=4",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-register") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=5",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-submit-transition") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=6",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-efdrain-control") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=7",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-prepare-map") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=8",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-fanin") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=9",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-winner-build-control") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=10",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-alloc-complete-control") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=11",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    assert _fdwic_compile_definitions("submit-pmu-loser-replay") == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=12",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]


@pytest.mark.parametrize(
    ("profile", "expected_compile_definitions"),
    [
        ("submit-pmu-none", ["PTO_FDWIC_SUBMIT_PMU=1", "PTO_FDWIC_TRACE_ENABLED=0"]),
        (
            "submit-pmu-efdrain-control",
            [
                "PTO_FDWIC_SUBMIT_PMU=1",
                "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=7",
                "PTO_FDWIC_TRACE_ENABLED=0",
            ],
        ),
        (
            "submit-pmu-prepare-map",
            [
                "PTO_FDWIC_SUBMIT_PMU=1",
                "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=8",
                "PTO_FDWIC_TRACE_ENABLED=0",
            ],
        ),
        (
            "submit-pmu-fanin",
            [
                "PTO_FDWIC_SUBMIT_PMU=1",
                "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=9",
                "PTO_FDWIC_TRACE_ENABLED=0",
            ],
        ),
        (
            "submit-pmu-winner-build-control",
            [
                "PTO_FDWIC_SUBMIT_PMU=1",
                "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=10",
                "PTO_FDWIC_TRACE_ENABLED=0",
            ],
        ),
        (
            "submit-pmu-alloc-complete-control",
            [
                "PTO_FDWIC_SUBMIT_PMU=1",
                "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=11",
                "PTO_FDWIC_TRACE_ENABLED=0",
            ],
        ),
        (
            "submit-pmu-loser-replay",
            [
                "PTO_FDWIC_SUBMIT_PMU=1",
                "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=12",
                "PTO_FDWIC_TRACE_ENABLED=0",
            ],
        ),
    ],
)
def test_submit_pmu_override_registers_build_identity_after_elf_gate(
    monkeypatch, tmp_path, profile, expected_compile_definitions
):
    """The profiled cache key must own one identity frozen from the built files."""
    runtime = "fully_distributed_within_core"
    profiled_key = ("QualifiedCase", "a5", runtime, "private", profile)
    orch = tmp_path / "orch.cpp"
    orch.write_text("// orchestration\n")
    host = tmp_path / "libhost_runtime.so"
    host.write_bytes(b"host")
    aicpu = tmp_path / "libaicpu_kernel.so"
    aicpu.write_bytes(b"aicpu")
    order = []
    captured = {}
    identity = object()

    class FakeRuntimeBuilder:
        _CACHE_DIR = tmp_path / "build" / "cache"
        _LIB_DIR = tmp_path / "build" / "lib"

        def __init__(self, platform, fdwic_tensormap_mode=None):
            assert platform == "a5"
            assert fdwic_tensormap_mode == "private"

        def build_aicore_with_extra_sources(
            self,
            name,
            extra_sources,
            cache_key,
            pto_isa_root=None,
            compile_definitions=None,
        ):
            assert name == runtime
            assert extra_sources == [orch.resolve()]
            order.append("build")
            binary = (
                self._LIB_DIR / "a5" / "onboard" / runtime / "private" / "aicore-extra" / cache_key / "aicore_kernel.o"
            )
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"aicore")
            return binary

        @staticmethod
        def effective_compile_definitions(name, compile_definitions=None):
            assert name == runtime
            return ["PTO_FDWIC_SHARED_MAP=0", *(compile_definitions or [])]

        def get_binaries(self, name):
            assert name == runtime
            return SimpleNamespace(host_path=host, aicpu_path=aicpu)

    def fake_elf_gate(binary, selected_profile):
        assert binary.is_file()
        assert selected_profile == profile
        order.append("elf-gate")

    def fake_host_elf_gate(binary, selected_profile):
        assert binary == host
        assert selected_profile == profile
        order.append("host-elf-gate")

    def fake_capture_build_identity(**kwargs):
        order.append("capture")
        captured.update(kwargs)
        return identity

    runtime_builder_module = importlib.import_module("simpler_setup.runtime_builder")
    report_module = importlib.import_module("simpler_setup.tools.fdwic_submit_pmu_report")
    monkeypatch.delenv("PTO_FDWIC_TENSORMAP_MODE", raising=False)
    monkeypatch.setenv("PTO_FDWIC_PROFILE", profile)
    monkeypatch.setattr(runtime_builder_module, "RuntimeBuilder", FakeRuntimeBuilder)
    monkeypatch.setattr(_scene_test_module, "_assert_fdwic_submit_pmu_elf", fake_elf_gate)
    monkeypatch.setattr(_scene_test_module, "_assert_fdwic_submit_pmu_host_elf", fake_host_elf_gate)
    monkeypatch.setattr(report_module, "capture_build_identity", fake_capture_build_identity)
    _fdwic_build_identity_cache.clear()

    binary = maybe_build_aicore_override(profiled_key, "a5", runtime, str(orch), [], pto_isa_root="/pto")

    extra_key = binary.parent.name
    expected_build_dir = (
        FakeRuntimeBuilder._CACHE_DIR / "a5" / "onboard" / runtime / "private" / "aicore-extra" / extra_key / "aicore"
    )
    assert order == ["build", "elf-gate", "host-elf-gate", "capture"]
    assert captured == {
        "profile": profile,
        "profiled_cache_key": profiled_key,
        "aicore_extra_cache_key": extra_key,
        "compile_definitions": ["PTO_FDWIC_SHARED_MAP=0", *expected_compile_definitions],
        "aicore_kernel": binary,
        "aicore_build_dir": expected_build_dir,
        "host_runtime": host,
        "aicpu_runtime": aicpu,
    }
    assert _fdwic_build_identity_cache == {profiled_key: identity}
    _fdwic_build_identity_cache.clear()


def test_submit_pmu_render_publishes_bound_provenance_and_html(monkeypatch, tmp_path):
    report_module = importlib.import_module("simpler_setup.tools.fdwic_submit_pmu_report")
    raw = tmp_path / report_module.DEFAULT_INPUT_NAME
    raw.write_text("{}")
    identity = object()
    called = []

    def fake_write_report_with_provenance(input_path, build_identity, output_path):
        called.append((input_path, build_identity, output_path))
        output_path.write_text("<html></html>")
        provenance = tmp_path / report_module.DEFAULT_PROVENANCE_NAME
        provenance.write_text("{}")
        return output_path, provenance

    monkeypatch.setattr(report_module, "write_report_with_provenance", fake_write_report_with_provenance)

    report = _render_case_fdwic_submit_pmu("Case", tmp_path, identity)

    assert report == tmp_path / report_module.DEFAULT_OUTPUT_NAME
    assert called == [(raw, identity, report)]


def test_submit_pmu_run_rejects_missing_build_identity_before_case(monkeypatch):
    class MissingIdentityCase:
        _st_level = 2
        _st_runtime = "fully_distributed_within_core"
        called = False

        @classmethod
        def _run_and_validate(cls, *args, **kwargs):
            cls.called = True

    monkeypatch.delenv("PTO_FDWIC_TENSORMAP_MODE", raising=False)
    monkeypatch.setenv("PTO_FDWIC_PROFILE", "submit-pmu-none")
    _fdwic_build_identity_cache.clear()
    worker = SimpleNamespace(_config={"platform": "a5", "device_id": 0})

    with pytest.raises(RuntimeError, match="build identity is missing"):
        run_class_cases(
            worker,
            MissingIdentityCase(),
            [{"name": "Case1", "config": {}, "params": {}}],
            callable_obj=None,
            sub_handles={},
            rounds=1,
            skip_golden=False,
            enable_l2_swimlane=0,
            enable_dump_args=False,
            enable_pmu=0,
            enable_dep_gen=False,
            enable_scope_stats=False,
        )

    assert not MissingIdentityCase.called


def test_submit_pmu_run_resolves_profiled_identity_for_render(monkeypatch, tmp_path):
    class IdentityCase:
        _st_level = 2
        _st_runtime = "fully_distributed_within_core"

        @staticmethod
        def _run_and_validate(*args, **kwargs):
            return None

    profile = "submit-pmu-none"
    key = (IdentityCase.__qualname__, "a5", IdentityCase._st_runtime, "private", profile)
    identity = object()
    rendered = []
    monkeypatch.delenv("PTO_FDWIC_TENSORMAP_MODE", raising=False)
    monkeypatch.setenv("PTO_FDWIC_PROFILE", profile)
    monkeypatch.setattr(_scene_test_module, "_build_output_prefix", lambda _label: tmp_path)
    monkeypatch.setattr(
        _scene_test_module,
        "_render_case_fdwic_submit_pmu",
        lambda case_label, output_prefix, build_identity: rendered.append((case_label, output_prefix, build_identity)),
    )
    _fdwic_build_identity_cache.clear()
    _fdwic_build_identity_cache[key] = identity
    worker = SimpleNamespace(_config={"platform": "a5", "device_id": 0})

    run_class_cases(
        worker,
        IdentityCase(),
        [{"name": "Case1", "config": {}, "params": {}}],
        callable_obj=None,
        sub_handles={},
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
    )

    assert rendered == [("IdentityCase_Case1", tmp_path, identity)]
    _fdwic_build_identity_cache.clear()


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


def test_perf_clock_kernel_elf_gate_requires_its_dedicated_marker(monkeypatch, tmp_path):
    symbol_table = (
        "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 _Z30dist_perf_clock_expect_submitsj\n"
        "37411: 0000000000001b98 8 FUNC WEAK DEFAULT 1 dist_perf_clock_kernel_profile_marker\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    _assert_fdwic_perf_clock_elf(tmp_path / "aicore_kernel.o", "perf-clock-kernel")


def test_perf_clock_kernel_elf_gate_rejects_plain_perf_clock_image(monkeypatch, tmp_path):
    symbol_table = "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 _Z30dist_perf_clock_expect_submitsj\n"
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    with pytest.raises(RuntimeError, match="missing defined perf-clock marker"):
        _assert_fdwic_perf_clock_elf(tmp_path / "aicore_kernel.o", "perf-clock-kernel")


def _write_perf_clock_artifact(tmp_path: Path, profile: str) -> Path:
    is_kernel = profile == "perf-clock-kernel"
    output_name = "fdwic_perf_clock_kernel_summary.json" if is_kernel else "fdwic_perf_clock_summary.json"
    cores = []
    for core_id in range(96):
        core = {
            "core_id": core_id,
            "core_type": "aic" if core_id < 32 else "aiv",
            "submit_count": 5,
            "first_submit_start": 100 + core_id,
            "last_submit_end": 200 + core_id,
            "elapsed_ticks": 100,
        }
        if is_kernel:
            has_kernel = core_id == 0
            core.update(
                kernel_elapsed_ticks=10 if has_kernel else 0,
                kernel_calls=1 if has_kernel else 0,
                non_kernel_residual_ticks=90 if has_kernel else 100,
            )
        cores.append(core)
    payload = {
        "schema": "fdwic-perf-clock-kernel-v1" if is_kernel else "fdwic-perf-clock-v1",
        "mode": profile,
        "num_cores": 96,
        "aic_cores": 32,
        "aiv_cores": 64,
        "expected_submits_per_core": 5,
        "global_first_submit_start": 100,
        "global_last_submit_end": 295,
        "global_submit_span_ticks": 195,
        "cores": cores,
    }
    if is_kernel:
        payload.update(
            kernel_calls=1,
            min_kernel_calls_in_window=1,
            max_kernel_calls_in_window=4,
            kernel_elapsed_ticks_sum=10,
            non_kernel_residual_ticks_sum=9590,
            groups={
                "aic": {
                    "cores": 32,
                    "elapsed_min_ticks": 100,
                    "elapsed_max_ticks": 100,
                    "elapsed_sum_ticks": 3200,
                    "kernel_min_ticks": 0,
                    "kernel_max_ticks": 10,
                    "kernel_sum_ticks": 10,
                    "kernel_calls_min": 0,
                    "kernel_calls_max": 1,
                    "kernel_calls_sum": 1,
                    "residual_min_ticks": 90,
                    "residual_max_ticks": 100,
                    "residual_sum_ticks": 3190,
                },
                "aiv": {
                    "cores": 64,
                    "elapsed_min_ticks": 100,
                    "elapsed_max_ticks": 100,
                    "elapsed_sum_ticks": 6400,
                    "kernel_min_ticks": 0,
                    "kernel_max_ticks": 0,
                    "kernel_sum_ticks": 0,
                    "kernel_calls_min": 0,
                    "kernel_calls_max": 0,
                    "kernel_calls_sum": 0,
                    "residual_min_ticks": 100,
                    "residual_max_ticks": 100,
                    "residual_sum_ticks": 6400,
                },
            },
        )
    else:
        payload["groups"] = {
            "aic": {"min_ticks": 100, "max_ticks": 100},
            "aiv": {"min_ticks": 100, "max_ticks": 100},
        }
    artifact = tmp_path / output_name
    artifact.write_text(json.dumps(payload))
    return artifact


@pytest.mark.parametrize("profile", ["perf-clock", "perf-clock-kernel"])
def test_perf_clock_case_artifact_contract_accepts_exact_closure(tmp_path, profile):
    artifact = _write_perf_clock_artifact(tmp_path, profile)

    assert _validate_case_fdwic_perf_clock("Case", tmp_path, profile) == artifact


def test_perf_clock_kernel_case_artifact_contract_rejects_wrong_integer_aggregate(tmp_path):
    artifact = _write_perf_clock_artifact(tmp_path, "perf-clock-kernel")
    payload = json.loads(artifact.read_text())
    payload["kernel_elapsed_ticks_sum"] += 1
    artifact.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="invalid perf-clock-kernel integer aggregates"):
        _validate_case_fdwic_perf_clock("Case", tmp_path, "perf-clock-kernel")


def test_perf_clock_case_artifact_contract_rejects_wrong_schema(tmp_path):
    artifact = _write_perf_clock_artifact(tmp_path, "perf-clock")
    payload = json.loads(artifact.read_text())
    payload["schema"] = "wrong"
    artifact.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="invalid perf-clock artifact contract"):
        _validate_case_fdwic_perf_clock("Case", tmp_path, "perf-clock")


def test_perf_clock_kernel_case_artifact_contract_rejects_zero_length_core_window(tmp_path):
    artifact = _write_perf_clock_artifact(tmp_path, "perf-clock-kernel")
    payload = json.loads(artifact.read_text())
    core = payload["cores"][1]
    core["last_submit_end"] = core["first_submit_start"]
    core["elapsed_ticks"] = 0
    core["non_kernel_residual_ticks"] = 0
    artifact.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="elapsed tick closure failed"):
        _validate_case_fdwic_perf_clock("Case", tmp_path, "perf-clock-kernel")


def test_perf_clock_kernel_case_artifact_contract_recomputes_pa_call_range(tmp_path):
    artifact = _write_perf_clock_artifact(tmp_path, "perf-clock-kernel")
    payload = json.loads(artifact.read_text())
    payload["expected_submits_per_core"] = 1280
    for core in payload["cores"]:
        core["submit_count"] = 1280
    artifact.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="global Kernel call range closure failed"):
        _validate_case_fdwic_perf_clock("Case", tmp_path, "perf-clock-kernel")


def test_perf_clock_kernel_case_artifact_contract_rejects_too_few_actual_calls(tmp_path):
    artifact = _write_perf_clock_artifact(tmp_path, "perf-clock-kernel")
    payload = json.loads(artifact.read_text())
    payload["expected_submits_per_core"] = 1280
    payload["min_kernel_calls_in_window"] = 256
    payload["max_kernel_calls_in_window"] = 1024
    for core in payload["cores"]:
        core["submit_count"] = 1280
    artifact.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="global Kernel call range closure failed"):
        _validate_case_fdwic_perf_clock("Case", tmp_path, "perf-clock-kernel")


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
        (
            "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 "
            "_Z30dist_perf_clock_expect_submitsj\n"
            "37411: 0000000000001b98 8 FUNC WEAK DEFAULT 1 dist_perf_clock_kernel_profile_marker\n",
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


@pytest.mark.parametrize(
    "profile",
    [
        "submit-pmu-arg-build",
        "submit-pmu-empty-bracket",
        "submit-pmu-materialize",
        "submit-pmu-claim",
        "submit-pmu-register",
        "submit-pmu-submit-transition",
        "submit-pmu-efdrain-control",
        "submit-pmu-prepare-map",
        "submit-pmu-fanin",
        "submit-pmu-winner-build-control",
        "submit-pmu-alloc-complete-control",
        "submit-pmu-loser-replay",
    ],
)
def test_submit_pmu_phase_elf_gate_requires_all_running_phase_readers(monkeypatch, tmp_path, profile):
    symbol_table = (
        "10: 0000000000001000 64 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
        "11: 0000000000001040 96 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
        "12: 00000000000010a0 64 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_phase_read_shadow_counters\n"
        "13: 00000000000010e0 64 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_phase_read_scalar_shadow\n"
        "14: 0000000000001120 64 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_phase_read_total_shadow\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    _assert_fdwic_submit_pmu_elf(tmp_path / "aicore_kernel.o", profile)


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


@pytest.mark.parametrize(
    "profile",
    [
        "submit-pmu-arg-build",
        "submit-pmu-empty-bracket",
        "submit-pmu-materialize",
        "submit-pmu-claim",
        "submit-pmu-register",
        "submit-pmu-submit-transition",
        "submit-pmu-efdrain-control",
        "submit-pmu-prepare-map",
        "submit-pmu-fanin",
        "submit-pmu-winner-build-control",
        "submit-pmu-alloc-complete-control",
        "submit-pmu-loser-replay",
    ],
)
def test_submit_pmu_phase_elf_gate_rejects_incomplete_running_phase_readers(monkeypatch, tmp_path, profile):
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

    with pytest.raises(RuntimeError, match="missing defined submit-pmu marker"):
        _assert_fdwic_submit_pmu_elf(tmp_path / "aicore_kernel.o", profile)


@pytest.mark.parametrize(
    "phase_reader",
    (
        "fdwic_submit_pmu_phase_read_shadow_counters",
        "fdwic_submit_pmu_phase_read_scalar_shadow",
        "fdwic_submit_pmu_phase_read_total_shadow",
    ),
)
def test_submit_pmu_none_elf_gate_rejects_every_running_phase_reader(monkeypatch, tmp_path, phase_reader):
    symbol_table = (
        "10: 0000000000001000 64 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n"
        "11: 0000000000001040 96 FUNC LOCAL DEFAULT 1 fdwic_submit_pmu_read_counters\n"
        f"12: 00000000000010a0 64 FUNC LOCAL DEFAULT 1 {phase_reader}\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    with pytest.raises(RuntimeError, match=r"unrelated profiling symbol\(s\) still present"):
        _assert_fdwic_submit_pmu_elf(tmp_path / "aicore_kernel.o", "submit-pmu-none")


@pytest.mark.parametrize(
    "profile",
    (
        "submit-pmu-efdrain-control",
        "submit-pmu-prepare-map",
        "submit-pmu-fanin",
        "submit-pmu-winner-build-control",
        "submit-pmu-alloc-complete-control",
        "submit-pmu-loser-replay",
    ),
)
def test_submit_pmu_host_elf_gate_accepts_exact_profile_and_hooks(monkeypatch, tmp_path, profile):
    host_runtime = tmp_path / "libhost_runtime.so"
    host_runtime.write_bytes(b"prefix\0" + profile.encode() + b"\0suffix")
    symbol_table = (
        "10: 0000000000001000 64 FUNC GLOBAL DEFAULT 1 fdwic_submit_pmu_host_init\n"
        "11: 0000000000001040 64 FUNC GLOBAL DEFAULT 1 fdwic_submit_pmu_host_export\n"
        "12: 0000000000001080 64 FUNC GLOBAL DEFAULT 1 fdwic_submit_pmu_host_finalize\n"
    )
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    _assert_fdwic_submit_pmu_host_elf(host_runtime, profile)


@pytest.mark.parametrize(
    ("image", "symbol_table", "message"),
    (
        (
            b"submit-pmu-submit-transition\0",
            "10: 0 1 FUNC GLOBAL DEFAULT 1 fdwic_submit_pmu_host_init\n"
            "11: 0 1 FUNC GLOBAL DEFAULT 1 fdwic_submit_pmu_host_export\n"
            "12: 0 1 FUNC GLOBAL DEFAULT 1 fdwic_submit_pmu_host_finalize\n",
            "missing exact profile marker",
        ),
        (
            b"submit-pmu-efdrain-control\0",
            "10: 0 1 FUNC GLOBAL DEFAULT 1 fdwic_submit_pmu_host_init\n",
            "missing defined host hook",
        ),
    ),
)
def test_submit_pmu_host_elf_gate_rejects_stale_or_incomplete_runtime(
    monkeypatch, tmp_path, image, symbol_table, message
):
    host_runtime = tmp_path / "libhost_runtime.so"
    host_runtime.write_bytes(image)
    monkeypatch.setattr(
        _scene_test_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=symbol_table, stderr=""),
    )

    with pytest.raises(RuntimeError, match=message):
        _assert_fdwic_submit_pmu_host_elf(host_runtime, "submit-pmu-efdrain-control")


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
            "isolated-profile symbol.*leaked",
        ),
        (
            "1: 0 1 FUNC LOCAL DEFAULT 1 fdwic_atomic_poll_boundary_slow\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_swimlane_detail_record_atomic\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 dist_submit_pmu_expect_submits\n",
            "isolated-profile symbol.*leaked",
        ),
        (
            "1: 0 1 FUNC LOCAL DEFAULT 1 fdwic_atomic_poll_boundary_slow\n"
            "2: 0 1 FUNC LOCAL DEFAULT 1 fdwic_swimlane_detail_record_atomic\n"
            "3: 0 1 FUNC WEAK DEFAULT 1 get_fdwic_submit_pmu_reg_base\n",
            "isolated-profile symbol.*leaked",
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
            "--fdwic-tensormap": "private",
            "--fdwic-profile": "submit-pmu-none",
            "--platform": "a5",
            "--runtime": "fully_distributed_within_core",
            "--level": 2,
            "--rounds": 1,
            **options,
        }

    def getoption(self, option, default=None):
        return self.options.get(option, default)


@pytest.mark.parametrize(
    "profile",
    (
        "submit-pmu-none",
        "submit-pmu-efdrain-control",
        "submit-pmu-prepare-map",
        "submit-pmu-fanin",
        "submit-pmu-winner-build-control",
        "submit-pmu-alloc-complete-control",
        "submit-pmu-loser-replay",
    ),
)
def test_submit_pmu_profile_publishes_environment(monkeypatch, profile):
    monkeypatch.delenv("PTO_FDWIC_PROFILE", raising=False)

    _configure_fdwic_profile(_FakePytestConfig(**{"--fdwic-profile": profile}))

    assert _fdwic_profile() == profile


def test_perf_clock_kernel_profile_publishes_environment(monkeypatch):
    monkeypatch.delenv("PTO_FDWIC_PROFILE", raising=False)

    _configure_fdwic_profile(_FakePytestConfig(**{"--fdwic-profile": "perf-clock-kernel"}))

    assert _fdwic_profile() == "perf-clock-kernel"


def test_shared_tensormap_mode_publishes_environment(monkeypatch):
    monkeypatch.delenv("PTO_FDWIC_TENSORMAP_MODE", raising=False)

    _configure_fdwic_tensormap(_FakePytestConfig(**{"--fdwic-tensormap": "shared"}))

    assert _fdwic_tensormap_mode() == "shared"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"--platform": "a2a3sim"}, "requires --platform a5 or a5sim"),
        ({"--runtime": "host_build_graph"}, "only supports runtime fully_distributed_within_core"),
        ({"--level": 3}, "only supports SceneTest level 2"),
    ],
)
def test_shared_tensormap_mode_rejects_wrong_execution_scope(options, message):
    with pytest.raises(pytest.UsageError, match=message):
        _configure_fdwic_tensormap(_FakePytestConfig(**{"--fdwic-tensormap": "shared", **options}))


def test_standalone_shared_tensormap_rejects_mixed_runtime_or_l3_classes():
    fdwic_l2 = type(
        "FdwicL2",
        (),
        {"_st_level": 2, "_st_runtime": "fully_distributed_within_core"},
    )
    fdwic_l3 = type(
        "FdwicL3",
        (),
        {"_st_level": 3, "_st_runtime": "fully_distributed_within_core"},
    )
    other_l2 = type("OtherL2", (), {"_st_level": 2, "_st_runtime": "host_build_graph"})

    _validate_fdwic_tensormap_test_classes("private", [fdwic_l3, other_l2])
    _validate_fdwic_tensormap_test_classes("shared", [fdwic_l2])
    with pytest.raises(ValueError, match=r"FdwicL3, OtherL2"):
        _validate_fdwic_tensormap_test_classes("shared", [fdwic_l2, other_l2, fdwic_l3])


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
