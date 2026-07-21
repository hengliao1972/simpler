# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SceneTestCase framework — unified scene test infrastructure.

``@scene_test`` decorator + ``SceneTestCase`` base class.
pytest: ``pytest --platform a2a3sim``
standalone: ``python test_xxx.py -p a2a3sim``

A scene test class declares three things:
  CALLABLE: what to compile/prepare
    L2: orchestration (C++ source) + incores (C++ kernels)
    L3: orchestration (Python DAG fn) + callables (ChipCallable + SubCallable)
  CASES: how to run (per-case platform, config, params)
  generate_args / compute_golden: data + golden comparison
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from .log_config import DEFAULT_LOG_LEVEL, LOG_LEVEL_CHOICES, configure_logging
from .pto_isa import ensure_pto_isa_root

if TYPE_CHECKING:
    from .tools.fdwic_submit_pmu_report import SubmitPmuBuildIdentity

logger = logging.getLogger(__name__)

_compile_cache: dict[tuple[Any, ...], object] = {}
_aicore_override_cache: dict[tuple[Any, ...], Path] = {}
_fdwic_build_identity_cache: dict[tuple[Any, ...], SubmitPmuBuildIdentity] = {}

_FDWIC_PROFILE_ENV = "PTO_FDWIC_PROFILE"
_FDWIC_PROFILE_NONE = "none"
_FDWIC_PROFILE_PERF_CLOCK = "perf-clock"
_FDWIC_PROFILE_PERF_CLOCK_KERNEL = "perf-clock-kernel"
_FDWIC_PROFILE_SUBMIT_PMU_NONE = "submit-pmu-none"
_FDWIC_PROFILE_SUBMIT_PMU_ARG_BUILD = "submit-pmu-arg-build"
_FDWIC_PROFILE_SUBMIT_PMU_EMPTY_BRACKET = "submit-pmu-empty-bracket"
_FDWIC_PROFILE_SUBMIT_PMU_MATERIALIZE = "submit-pmu-materialize"
_FDWIC_PROFILE_SUBMIT_PMU_CLAIM = "submit-pmu-claim"
_FDWIC_PROFILE_SUBMIT_PMU_REGISTER = "submit-pmu-register"
_FDWIC_PROFILE_SUBMIT_PMU_SUBMIT_TRANSITION = "submit-pmu-submit-transition"
_FDWIC_PROFILE_SUBMIT_PMU_EFDRAIN_CONTROL = "submit-pmu-efdrain-control"
_FDWIC_PERF_CLOCK_ARTIFACTS = {
    _FDWIC_PROFILE_PERF_CLOCK: ("fdwic_perf_clock_summary.json", "fdwic-perf-clock-v1"),
    _FDWIC_PROFILE_PERF_CLOCK_KERNEL: (
        "fdwic_perf_clock_kernel_summary.json",
        "fdwic-perf-clock-kernel-v1",
    ),
}
_FDWIC_SUBMIT_PMU_PHASE_PROFILES = frozenset(
    {
        _FDWIC_PROFILE_SUBMIT_PMU_ARG_BUILD,
        _FDWIC_PROFILE_SUBMIT_PMU_EMPTY_BRACKET,
        _FDWIC_PROFILE_SUBMIT_PMU_MATERIALIZE,
        _FDWIC_PROFILE_SUBMIT_PMU_CLAIM,
        _FDWIC_PROFILE_SUBMIT_PMU_REGISTER,
        _FDWIC_PROFILE_SUBMIT_PMU_SUBMIT_TRANSITION,
        _FDWIC_PROFILE_SUBMIT_PMU_EFDRAIN_CONTROL,
    }
)
_FDWIC_SUBMIT_PMU_PROFILES = frozenset({_FDWIC_PROFILE_SUBMIT_PMU_NONE, *_FDWIC_SUBMIT_PMU_PHASE_PROFILES})
_FDWIC_PERF_CLOCK_PROFILES = frozenset(_FDWIC_PERF_CLOCK_ARTIFACTS)
_FDWIC_PRIVATE_PROFILES = frozenset({*_FDWIC_PERF_CLOCK_PROFILES, *_FDWIC_SUBMIT_PMU_PROFILES})
_FDWIC_PROFILES = frozenset({_FDWIC_PROFILE_NONE, *_FDWIC_PRIVATE_PROFILES})


def _fdwic_profile() -> str:
    profile = os.environ.get(_FDWIC_PROFILE_ENV, _FDWIC_PROFILE_NONE) or _FDWIC_PROFILE_NONE
    if profile not in _FDWIC_PROFILES:
        raise ValueError(f"Unsupported {_FDWIC_PROFILE_ENV}={profile!r}")
    return profile


def _fdwic_compile_definitions(profile: str) -> list[str] | None:
    """Return the ABI-preserving compile gates for one private FDWIC image."""
    if profile == _FDWIC_PROFILE_PERF_CLOCK:
        return ["PTO_FDWIC_PERF_CLOCK=1", "PTO_FDWIC_TRACE_ENABLED=0"]
    if profile == _FDWIC_PROFILE_PERF_CLOCK_KERNEL:
        return [
            "PTO_FDWIC_PERF_CLOCK=1",
            "PTO_FDWIC_PERF_CLOCK_KERNEL=1",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_NONE:
        return ["PTO_FDWIC_SUBMIT_PMU=1", "PTO_FDWIC_TRACE_ENABLED=0"]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_ARG_BUILD:
        return [
            "PTO_FDWIC_SUBMIT_PMU=1",
            "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=1",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_EMPTY_BRACKET:
        return [
            "PTO_FDWIC_SUBMIT_PMU=1",
            "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=2",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_MATERIALIZE:
        return [
            "PTO_FDWIC_SUBMIT_PMU=1",
            "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=3",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_CLAIM:
        return [
            "PTO_FDWIC_SUBMIT_PMU=1",
            "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=4",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_REGISTER:
        return [
            "PTO_FDWIC_SUBMIT_PMU=1",
            "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=5",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_SUBMIT_TRANSITION:
        return [
            "PTO_FDWIC_SUBMIT_PMU=1",
            "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=6",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_EFDRAIN_CONTROL:
        return [
            "PTO_FDWIC_SUBMIT_PMU=1",
            "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=7",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]
    return None


def _profiled_cache_key(cache_key) -> tuple[Any, ...]:
    base = cache_key if isinstance(cache_key, tuple) else (cache_key,)
    return (*base, _fdwic_profile())


def clear_compile_cache() -> None:
    """Drop every cached ``ChipCallable`` and force a GC pass.

    The cache keeps nanobind-owned ``ChipCallable`` instances alive for the
    whole pytest session. Module-level dicts are cleared by Python in an
    order that can outlive the nanobind module destructor, which then
    prints ``leaked N instances of type _task_interface.ChipCallable`` to
    stderr at interpreter shutdown. Call this from ``pytest_sessionfinish``
    (and other session-end paths) so the instances die while the nanobind
    module is still wired up.
    """
    _compile_cache.clear()
    _aicore_override_cache.clear()
    _fdwic_build_identity_cache.clear()
    gc.collect()


def _aicore_extra_cache_key(cache_key, sources: list[Path]) -> str:
    h = hashlib.sha256()
    h.update(repr(cache_key).encode("utf-8"))
    for source in sources:
        h.update(str(source.resolve()).encode("utf-8"))
        if source.is_file():
            h.update(source.read_bytes())
    return h.hexdigest()[:16]


def _write_aicore_incore_wrapper(cache_key: str, incores: list[dict]) -> Path | None:
    if not incores:
        return None

    wrapper_dir = _project_root() / "build" / "cache" / "aicore-extra-wrappers"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = wrapper_dir / f"linked_incores_{cache_key}.cpp"

    lines = [
        "#include <cstdint>",
        '#include "pto_types.h"',
        "",
    ]
    for core_type, guard, caller in (
        ("aic", "#ifndef __DAV_VEC__", "pto_call_linked_kernel_aic"),
        ("aiv", "#ifdef __DAV_VEC__", "pto_call_linked_kernel_aiv"),
    ):
        lines.extend([guard, ""])
        for incore in incores:
            if str(incore["core_type"]).lower() != core_type:
                continue
            func_id = int(incore["func_id"])
            source = Path(incore["source"]).resolve()
            symbol = f"pto_linked_kernel_func_{func_id}"
            lines.extend(
                [
                    f"#define kernel_entry {symbol}",
                    f'#include "{source}"',
                    "#undef kernel_entry",
                    "",
                ]
            )

        lines.append(f'extern "C" PTO_DEVICE_FUNC int32_t {caller}(int32_t func_id, __gm__ int64_t *args) {{')
        for incore in incores:
            if str(incore["core_type"]).lower() != core_type:
                continue
            func_id = int(incore["func_id"])
            symbol = f"pto_linked_kernel_func_{func_id}"
            lines.append(f"    if (func_id == {func_id}) {{")
            lines.append(f"        {symbol}(args);")
            lines.append("        return 1;")
            lines.append("    }")
        lines.extend(["    return 0;", "}", "#endif", ""])

    wrapper_path.write_text("\n".join(lines))
    return wrapper_path


def get_aicore_path_override(cache_key) -> Path | None:
    return _aicore_override_cache.get(_profiled_cache_key(cache_key))


def _fdwic_elf_symbol_rows(binary: Path) -> list[tuple[str, str, str]]:
    """Return ``(kind, section, name)`` rows from one final FDWIC ELF."""
    try:
        result = subprocess.run(
            ["readelf", "-Ws", "-W", str(binary)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("readelf is required to validate the FDWIC ELF") from exc
    if result.returncode != 0:
        raise RuntimeError(f"readelf failed for FDWIC ELF {binary}: {result.stderr.strip()}")

    symbol_rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].endswith(":"):
            continue
        symbol_rows.append((fields[3], fields[6], fields[7]))
    return symbol_rows


def _assert_fdwic_perf_clock_elf(binary: Path, profile: str = _FDWIC_PROFILE_PERF_CLOCK) -> None:
    """Prove the final CCEC image excludes FDWIC trace/atomic and platform PMU code."""
    if profile not in _FDWIC_PERF_CLOCK_PROFILES:
        raise ValueError(f"Unsupported perf-clock ELF profile {profile!r}")
    symbol_rows = _fdwic_elf_symbol_rows(binary)

    required = ["dist_perf_clock_expect_submits"]
    if profile == _FDWIC_PROFILE_PERF_CLOCK_KERNEL:
        required.append("dist_perf_clock_kernel_profile_marker")
    forbidden = [
        "fdwic_atomic_poll_boundary_slow",
        "fdwic_swimlane_detail_record_atomic",
        "g_fdwic_swimlane_",
        "g_fdwic_atomic_",
        "g_fdwic_poll_",
        "set_aicore_profiling_flag",
        "get_aicore_profiling_flag",
        "set_l2_swimlane_aicore_head_slot",
        "get_l2_swimlane_aicore_head",
        "set_aicore_pmu_ring",
        "get_aicore_pmu_ring",
        "set_aicore_pmu_reg_base",
        "get_aicore_pmu_reg_base",
        "dist_submit_pmu_expect_submits",
        "fdwic_submit_pmu_read_counters",
        "set_fdwic_submit_pmu_reg_base",
        "get_fdwic_submit_pmu_reg_base",
        "g_fdwic_submit_pmu_",
    ]
    if profile == _FDWIC_PROFILE_PERF_CLOCK:
        forbidden.extend(("dist_perf_clock_kernel_profile_marker", "g_fdwic_perf_clock_kernel_"))
    missing = [
        symbol
        for symbol in required
        if not any(kind == "FUNC" and ndx != "UND" and symbol in name for kind, ndx, name in symbol_rows)
    ]
    present = [symbol for symbol in forbidden if any(symbol in name for _kind, _ndx, name in symbol_rows)]
    if missing or present:
        details = []
        if missing:
            details.append(f"missing defined perf-clock marker(s): {', '.join(missing)}")
        if present:
            details.append(f"profiling symbol(s) still present: {', '.join(present)}")
        raise RuntimeError(f"Invalid perf-clock AICore image {binary}: {'; '.join(details)}")


def _assert_fdwic_submit_pmu_elf(binary: Path, profile: str) -> None:
    """Prove one submit-PMU image contains exactly its selected observer."""
    symbol_rows = _fdwic_elf_symbol_rows(binary)

    phase_reader = "fdwic_submit_pmu_phase_read_shadow_counters"
    required = ["dist_submit_pmu_expect_submits", "fdwic_submit_pmu_read_counters"]
    if profile in _FDWIC_SUBMIT_PMU_PHASE_PROFILES:
        required.append(phase_reader)
    forbidden = (
        "dist_perf_clock_expect_submits",
        "g_fdwic_perf_clock_",
        "fdwic_atomic_poll_boundary_slow",
        "fdwic_swimlane_detail_record_atomic",
        "g_fdwic_swimlane_",
        "g_fdwic_atomic_",
        "g_fdwic_poll_",
        "set_aicore_profiling_flag",
        "get_aicore_profiling_flag",
        "set_l2_swimlane_aicore_head_slot",
        "get_l2_swimlane_aicore_head",
        "set_aicore_pmu_ring",
        "get_aicore_pmu_ring",
        "set_aicore_pmu_reg_base",
        "get_aicore_pmu_reg_base",
        "pmu_aicore_record_task",
    )
    if profile == _FDWIC_PROFILE_SUBMIT_PMU_NONE:
        forbidden = (*forbidden, phase_reader)
    missing = [
        symbol
        for symbol in required
        if not any(kind == "FUNC" and ndx != "UND" and symbol in name for kind, ndx, name in symbol_rows)
    ]
    present = [symbol for symbol in forbidden if any(symbol in name for _kind, _ndx, name in symbol_rows)]
    if missing or present:
        details = []
        if missing:
            details.append(f"missing defined submit-pmu marker(s): {', '.join(missing)}")
        if present:
            details.append(f"unrelated profiling symbol(s) still present: {', '.join(present)}")
        raise RuntimeError(f"Invalid {profile} AICore image {binary}: {'; '.join(details)}")


def _assert_fdwic_submit_pmu_host_elf(binary: Path, profile: str) -> None:
    """Reject a stale host runtime before it can launch a profiled device case."""
    if profile not in _FDWIC_SUBMIT_PMU_PROFILES:
        raise ValueError(f"Unsupported submit-PMU host profile {profile!r}")
    symbol_rows = _fdwic_elf_symbol_rows(binary)
    required = (
        "fdwic_submit_pmu_host_init",
        "fdwic_submit_pmu_host_export",
        "fdwic_submit_pmu_host_finalize",
    )
    missing = [
        symbol
        for symbol in required
        if not any(kind == "FUNC" and ndx != "UND" and symbol in name for kind, ndx, name in symbol_rows)
    ]
    try:
        image = Path(binary).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Cannot read submit-PMU host runtime {binary}: {exc}") from exc
    profile_marker = profile.encode("utf-8") + b"\0"
    if missing or profile_marker not in image:
        details = []
        if missing:
            details.append(f"missing defined host hook(s): {', '.join(missing)}")
        if profile_marker not in image:
            details.append(f"missing exact profile marker {profile!r}; rebuild the a5 FDWIC host runtime")
        raise RuntimeError(f"Invalid {profile} host runtime {binary}: {'; '.join(details)}")


def _assert_fdwic_swimlane_elf(binary: Path) -> None:
    """Prove the normal real-A5 FDWIC image carries the merged phase/atomic observer."""
    symbol_rows = _fdwic_elf_symbol_rows(binary)
    required = ("fdwic_atomic_poll_boundary_slow", "fdwic_swimlane_detail_record_atomic")
    forbidden = (
        "dist_perf_clock_expect_submits",
        "dist_submit_pmu_expect_submits",
        "fdwic_submit_pmu_read_counters",
        "set_fdwic_submit_pmu_reg_base",
        "get_fdwic_submit_pmu_reg_base",
        "g_fdwic_submit_pmu_",
    )
    missing = [
        symbol
        for symbol in required
        if not any(kind == "FUNC" and ndx != "UND" and symbol in name for kind, ndx, name in symbol_rows)
    ]
    present = [symbol for symbol in forbidden if any(symbol in name for _kind, _ndx, name in symbol_rows)]
    if missing or present:
        details = []
        if missing:
            details.append(f"missing defined swimlane observer(s): {', '.join(missing)}")
        if present:
            details.append(f"private-profile symbol(s) leaked into normal image: {', '.join(present)}")
        raise RuntimeError(f"Invalid FDWIC swimlane AICore image {binary}: {'; '.join(details)}")


def maybe_build_aicore_override(
    cache_key,
    platform: str,
    runtime: str,
    orch_source: str,
    incores: list[dict],
    pto_isa_root: str | None = None,
) -> Path | None:
    profile = _fdwic_profile()
    if profile in _FDWIC_PRIVATE_PROFILES and (platform != "a5" or runtime != "fully_distributed_within_core"):
        raise ValueError(f"{profile} is only supported by the real a5 fully_distributed_within_core runtime")
    if platform not in {"a5", "a5sim"} or runtime != "fully_distributed_within_core":
        return None

    from .runtime_builder import RuntimeBuilder  # noqa: PLC0415

    source_paths = [Path(orch_source).resolve()]
    if platform == "a5":
        incore_sources = [Path(incore["source"]).resolve() for incore in incores]
        key = _aicore_extra_cache_key(cache_key, [Path(orch_source).resolve(), *incore_sources])
        wrapper_path = _write_aicore_incore_wrapper(key, incores)
        if wrapper_path is not None:
            source_paths.append(wrapper_path)
    key = _aicore_extra_cache_key(cache_key, source_paths)
    # Keep PTO2_PROFILING at its normal value because it also owns the public
    # Arg layout. Each private profile independently removes the dist
    # swimlane/atomic path without changing orchestration/incore ABI.
    compile_definitions = _fdwic_compile_definitions(profile)
    builder = RuntimeBuilder(platform)
    binary = builder.build_aicore_with_extra_sources(
        runtime,
        source_paths,
        key,
        pto_isa_root=pto_isa_root,
        compile_definitions=compile_definitions,
    )
    if profile in _FDWIC_PERF_CLOCK_PROFILES:
        _assert_fdwic_perf_clock_elf(binary, profile)
    elif profile in _FDWIC_SUBMIT_PMU_PROFILES:
        _assert_fdwic_submit_pmu_elf(binary, profile)
        from .tools.fdwic_submit_pmu_report import capture_build_identity  # noqa: PLC0415

        host_runtime = builder.get_binaries(runtime).host_path
        _assert_fdwic_submit_pmu_host_elf(host_runtime, profile)
        output_key_dir = Path(binary).resolve().parent
        aicore_build_dir = builder._CACHE_DIR / output_key_dir.relative_to(builder._LIB_DIR) / "aicore"
        _fdwic_build_identity_cache[cache_key] = capture_build_identity(
            profile=profile,
            profiled_cache_key=cache_key,
            aicore_extra_cache_key=key,
            compile_definitions=compile_definitions or (),
            aicore_kernel=binary,
            aicore_build_dir=aicore_build_dir,
            host_runtime=host_runtime,
        )
    elif platform == "a5" and runtime == "fully_distributed_within_core":
        _assert_fdwic_swimlane_elf(binary)
    return binary


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


class Tensor(NamedTuple):
    """Tensor argument spec."""

    name: str
    value: Any  # torch.Tensor


class Scalar(NamedTuple):
    """Scalar argument spec (ctypes scalar)."""

    name: str
    value: Any  # ctypes.c_float, ctypes.c_int64, etc.


# ---------------------------------------------------------------------------
# TaskArgsBuilder — ordered container with named access
# ---------------------------------------------------------------------------


class TaskArgsBuilder:
    """Test-side task arguments container.

    Maintains insertion order (tensors before scalars) and provides
    attribute access by name for use in compute_golden.

    Usage::

        args = TaskArgsBuilder(
            Tensor("a", torch.full((N,), 2.0)),
            Tensor("b", torch.full((N,), 3.0)),
            Tensor("f", torch.zeros(N)),
            Scalar("scale", ctypes.c_float(1.5)),
        )
        args.a  # → tensor
        args.f[:] = args.a + args.b  # in compute_golden
    """

    def __init__(self, *specs):
        self._specs: list = []
        self._data: dict[str, Any] = {}
        self._has_scalar = False
        for spec in specs:
            if isinstance(spec, Tensor):
                self._add_tensor(spec)
            elif isinstance(spec, Scalar):
                self._add_scalar(spec)

    def add_tensor(self, name: str, value: Any) -> None:
        """Add a tensor. Must be called before any add_scalar."""
        self._add_tensor(Tensor(name, value))

    def add_scalar(self, name: str, value: Any) -> None:
        """Add a scalar. After this, add_tensor is not allowed."""
        self._add_scalar(Scalar(name, value))

    def _add_tensor(self, spec: Tensor) -> None:
        if self._has_scalar:
            raise ValueError("Cannot add tensor after scalar (tensor-before-scalar ordering required)")
        self._specs.append(spec)
        self._data[spec.name] = spec.value

    def _add_scalar(self, spec: Scalar) -> None:
        self._has_scalar = True
        self._specs.append(spec)
        self._data[spec.name] = spec.value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"TaskArgsBuilder has no argument '{name}'") from None

    def clone(self) -> TaskArgsBuilder:
        """Deep clone: all tensors are cloned, scalars copied."""
        import torch  # noqa: PLC0415

        new = TaskArgsBuilder.__new__(TaskArgsBuilder)
        new._specs = []
        new._data = {}
        new._has_scalar = False
        for spec in self._specs:
            if isinstance(spec, Tensor):
                cloned = spec.value.clone() if isinstance(spec.value, torch.Tensor) else spec.value
                new_spec = Tensor(spec.name, cloned)
                new._specs.append(new_spec)
                new._data[spec.name] = cloned
            elif isinstance(spec, Scalar):
                import copy  # noqa: PLC0415

                new._has_scalar = True
                cloned_val = copy.copy(spec.value)
                new_spec = Scalar(spec.name, cloned_val)
                new._specs.append(new_spec)
                new._data[spec.name] = cloned_val
        return new

    @property
    def specs(self) -> list:
        """Ordered list of Tensor/Scalar specs."""
        return self._specs

    def tensor_names(self) -> list[str]:
        """Names of all tensor arguments, in order."""
        return [s.name for s in self._specs if isinstance(s, Tensor)]


# ---------------------------------------------------------------------------
# CallableNamespace — dot-access container for L3 callables
# ---------------------------------------------------------------------------


class CallableNamespace:
    """Dot-access container for compiled/prepared callables.

    Used by L3 orch functions to access callables by name::

        callables.vector_kernel       # → ChipCallable object
        callables.vector_kernel_sig   # → signature list
        callables.verify              # → CallableHandle

    Also provides ``keep()`` for lifetime management: L3 orch functions
    that build transient Python objects (e.g. ChipStorageTaskArgs) whose
    raw pointers are submitted to the C++ scheduler must register them
    via ``keep()`` so they outlive the scheduler drain::

        def run_dag(w, callables, task_args, config):
            chip_args, _ = _build_chip_task_args(task_args, callables.vector_kernel_sig)
            callables.keep(chip_args)  # survive until drain finishes
            ...
    """

    def __init__(self, entries: dict):
        self._entries = dict(entries)
        self._keepalive: list = []

    def __getattr__(self, name: str):
        try:
            return self._entries[name]
        except KeyError:
            raise AttributeError(f"CallableNamespace has no entry '{name}'") from None

    def keep(self, *objs):
        """Register objects to keep alive until this namespace is destroyed."""
        if not objs:
            return None
        self._keepalive.extend(objs)
        return objs[0] if len(objs) == 1 else objs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chip_task_args(test_args: TaskArgsBuilder, orch_signature: list):
    """Build `ChipStorageTaskArgs` (POD) from `TaskArgsBuilder`.

    Used by the L2 path (`ChipWorker.run(callable, chip_args, config)`): the
    chip worker expects the runtime.so ABI-shaped POD directly (no tags).

    Returns:
        chip_args: ChipStorageTaskArgs (POD)
        output_names: list of tensor names that are OUTPUT or INOUT
    """
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        ChipStorageTaskArgs,
        scalar_to_uint64,
    )

    from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

    chip_args = ChipStorageTaskArgs()
    output_names: list[str] = []

    tensor_idx = 0
    for spec in test_args.specs:
        if isinstance(spec, Tensor):
            if tensor_idx >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                    f"orchestration signature (length {len(orch_signature)}). "
                    f"Update CALLABLE['orchestration']['signature'] to match generate_args()."
                )
            direction = orch_signature[tensor_idx]
            chip_args.add_tensor(make_tensor_arg(spec.value))
            if direction in (ArgDirection.OUT, ArgDirection.INOUT):
                output_names.append(spec.name)
            tensor_idx += 1
        elif isinstance(spec, Scalar):
            chip_args.add_scalar(scalar_to_uint64(spec.value))

    return chip_args, output_names


def _build_l3_task_args(test_args: TaskArgsBuilder, orch_signature: list):
    """Build a tagged `TaskArgs` (vector-backed, with `TensorArgType` tags) from
    `TaskArgsBuilder`.

    Used by the L3 path (`orch.submit_next_level(callable, args, config)`):
    the orchestrator reads the tags to drive dependency inference.

    Returns:
        chip_args: TaskArgs (tagged)
        output_names: list of tensor names that are OUTPUT or INOUT
    """
    from simpler.task_interface import (  # noqa: PLC0415
        ArgDirection,
        TaskArgs,
        TensorArgType,
        scalar_to_uint64,
    )

    from simpler_setup.torch_interop import make_tensor_arg  # noqa: PLC0415

    _DIR_TO_TAG = {
        ArgDirection.IN: TensorArgType.INPUT,
        ArgDirection.OUT: TensorArgType.OUTPUT_EXISTING,
        ArgDirection.INOUT: TensorArgType.INOUT,
    }

    chip_args = TaskArgs()
    output_names: list[str] = []

    tensor_idx = 0
    for spec in test_args.specs:
        if isinstance(spec, Tensor):
            if tensor_idx >= len(orch_signature):
                raise ValueError(
                    f"Tensor '{spec.name}' at index {tensor_idx} has no matching entry in "
                    f"orchestration signature (length {len(orch_signature)}). "
                    f"Update CALLABLE['orchestration']['signature'] to match generate_args()."
                )
            direction = orch_signature[tensor_idx]
            tag = _DIR_TO_TAG.get(direction, TensorArgType.INPUT)
            chip_args.add_tensor(make_tensor_arg(spec.value), tag)
            if direction in (ArgDirection.OUT, ArgDirection.INOUT):
                output_names.append(spec.name)
            tensor_idx += 1
        elif isinstance(spec, Scalar):
            chip_args.add_scalar(scalar_to_uint64(spec.value))

    return chip_args, output_names


@contextmanager
def _temporary_env(env_updates):
    """Temporarily set environment variables."""
    if not env_updates:
        yield
        return
    old = {k: os.environ.get(k) for k in env_updates}
    for k, v in env_updates.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _log_round_timings(timings):
    """Print per-round + summary host/device wall (µs) for multi-round runs.

    Replaces the device-log scraping that ``tools/benchmark_rounds.sh`` used to
    do — Worker.run now returns the timing directly, so the per-round table
    can be emitted by the framework without grep+awk over device logs.

    Output goes to ``print`` (not ``logger.info``) because benchmark numbers
    are a user-facing artifact and the project's default log level suppresses
    INFO in standalone test_*.py runs.

    ``timings`` is a list of (host_wall_us, device_wall_us) tuples. The device
    column reports 0 when the runtime was built without PTO2_PROFILING (the
    default build has it on) or when an L3+ DAG is in use (per-task device
    cycles aren't aggregated).
    """
    if not timings:
        return
    n = len(timings)
    host_vals = sorted(t[0] for t in timings)
    dev_vals = sorted(t[1] for t in timings)
    host_avg = sum(host_vals) / n
    # Device avg averages over non-zero rounds only. When --rounds > 1 the
    # framework restricts enable_l2_swimlane (and thus orch_summary capture)
    # to round 0, so the other rounds report 0; including those zeros in the
    # average would silently halve / quarter the reported device wall and
    # mislead the benchmark consumer. dev_count carries the sample size into
    # the summary line for transparency.
    dev_nonzero = [v for v in dev_vals if v > 0.0]
    dev_count = len(dev_nonzero)
    dev_avg = (sum(dev_nonzero) / dev_count) if dev_count else 0.0
    show_device = dev_count > 0

    header = f"  {'Round':<6}  {'Host (us)':>12}"
    if show_device:
        header += f"  {'Device (us)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, (h, d) in enumerate(timings):
        line = f"  {i:<6d}  {h:>12.1f}"
        if show_device:
            line += f"  {d:>12.1f}"
        print(line)
    summary = f"  Avg Host: {host_avg:.1f} us"
    if show_device:
        summary += f"  |  Avg Device: {dev_avg:.1f} us [{dev_count}/{n} rounds captured]"
    summary += f"  ({n} rounds)"
    print(summary)

    trim = 10
    if n > 2 * trim:
        tc = n - 2 * trim
        host_trim = sum(host_vals[trim:-trim]) / tc
        msg = f"  Trimmed Avg Host: {host_trim:.1f} us"
        if show_device and dev_count > 2 * trim:
            dev_nonzero_sorted = sorted(dev_nonzero)
            dev_trim_count = dev_count - 2 * trim
            dev_trim = sum(dev_nonzero_sorted[trim:-trim]) / dev_trim_count
            msg += f"  |  Trimmed Avg Device: {dev_trim:.1f} us"
        msg += f"  (dropped {trim} low + {trim} high, {tc} rounds used)"
        print(msg)


def _get_device_log_dir(device_id) -> Path:
    """Return the CANN device log directory for *device_id*.

    Delegates to ``device_log_resolver.get_log_root`` so the log-relocation env
    vars (ASCEND_PROCESS_LOG_PATH / ASCEND_WORK_PATH) are honored in one place.
    """
    from .tools.device_log_resolver import get_log_root  # noqa: PLC0415

    return get_log_root() / f"device-{device_id}"


def _snapshot_time() -> int:
    """Record current time as a baseline for detecting device logs written after this point."""
    import time  # noqa: PLC0415

    return time.time_ns()


def _snapshot_log_offsets(log_dir: Path) -> dict[str, int]:
    """Byte size of each existing ``*.log`` in *log_dir*, keyed by path string.

    Lets the post-run parser read only the bytes this run appends, so blocks an
    earlier case wrote into the same (reused-process) log file are not counted.
    """
    offsets: dict[str, int] = {}
    if log_dir.exists():
        for p in log_dir.glob("*.log"):
            try:
                offsets[str(p)] = p.stat().st_size
            except FileNotFoundError:
                continue
    return offsets


def _print_device_log_timing(device_id, before_time, baseline_offsets, expected_rounds, timeout=20.0):
    """Read the freshest device log and print per-round Total / Orch / Sched.

    Driven by ``--enable-device-log-timing``. The orch/sched ``LOG_INFO_V9``
    markers are emitted every round by ``PTO2_PROFILING`` (default-on, and
    independent of ``--enable-l2-swimlane``), so this works for any ``--rounds``
    count — unlike the swimlane-backed device column in ``_log_round_timings``,
    which only captures round 0 when ``--rounds > 1``.

    CANN flushes the device log asynchronously, so this polls until the fresh
    logs both exist (mtime past ``before_time``) and carry at least
    ``expected_rounds`` timing blocks, rather than reading once and racing the
    flush. CANN also rotates the log at 20 MB, so a long run's blocks may span
    several files; all files written after ``before_time`` are read in mtime
    order, not just the newest. ``baseline_offsets`` (from ``_snapshot_log_offsets``)
    caps each pre-existing file's read to the bytes appended after the run began.
    """
    import time  # noqa: PLC0415

    from .tools.device_log_timing import format_device_log_timing, parse_device_log_timing  # noqa: PLC0415

    if os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT") == "1":
        logger.warning(
            "device-log timing: ASCEND_SLOG_PRINT_TO_STDOUT=1 routes CANN logs to stdout, not to a "
            "device log file — the orch/sched timing markers are in the run output above, not parseable here."
        )
        return

    log_dir = _get_device_log_dir(device_id)
    deadline = time.monotonic() + timeout
    fresh: list[Path] = []
    rounds = []
    while time.monotonic() < deadline:
        if log_dir.exists():
            # stat() once per file, skipping any that vanish mid-scan from log
            # rotation (the case this reader handles), then sort on the cached mtime.
            paths_with_mtime = []
            for p in log_dir.glob("*.log"):
                try:
                    mtime = p.stat().st_mtime_ns
                except FileNotFoundError:
                    continue
                if mtime > before_time:
                    paths_with_mtime.append((mtime, p))
            paths_with_mtime.sort()
            fresh = [p for _, p in paths_with_mtime]
            if fresh:
                rounds = parse_device_log_timing(fresh, offsets=baseline_offsets)
                if len(rounds) >= expected_rounds:
                    break
        time.sleep(0.5)

    if not fresh:
        logger.warning("device-log timing: no device log written after run under %s", log_dir)
        return
    if len(rounds) < expected_rounds:
        logger.warning(
            "device-log timing: only %d/%d round blocks flushed within %.0fs (CANN async log lag)",
            len(rounds),
            expected_rounds,
            timeout,
        )
    source = fresh[-1] if len(fresh) == 1 else f"{len(fresh)} files under {log_dir}"
    print(format_device_log_timing(rounds, source=source))


def _resolve_callable_paths(cls, cls_dir):
    """Resolve relative source paths in CALLABLE against cls_dir."""
    callable_spec = cls.CALLABLE
    if "callables" in callable_spec:
        # L3: resolve inside each ChipCallable entry
        resolved = []
        for entry in callable_spec["callables"]:
            if "orchestration" in entry:
                entry = dict(entry)
                _resolve_chip_entry_paths(entry, cls_dir)
            resolved.append(entry)
        callable_spec["callables"] = resolved
    else:
        # L2: resolve orchestration + incores directly
        _resolve_chip_entry_paths(callable_spec, cls_dir)


def _resolve_chip_entry_paths(entry, cls_dir):
    """Resolve relative source paths in a chip entry (orchestration + incores)."""
    if "orchestration" in entry:
        orch = entry["orchestration"]
        if isinstance(orch, dict) and "source" in orch and not os.path.isabs(orch["source"]):
            entry["orchestration"] = dict(orch)
            entry["orchestration"]["source"] = str(cls_dir / orch["source"])
    if "incores" in entry:
        resolved = []
        for k in entry["incores"]:
            k = dict(k)
            if "source" in k and not os.path.isabs(k["source"]):
                k["source"] = str(cls_dir / k["source"])
            resolved.append(k)
        entry["incores"] = resolved


def _extract_name_map(callable_spec: dict) -> dict:
    """Extract name mapping from a CALLABLE spec.

    Each level exports only its own ``callable_id_to_name`` — the mapping
    from next-level-down IDs to human-readable names.  No cross-level
    nesting: the perf data declares its level, the mapping declares its
    level, and the tool matches them.

    * **L2** — ``callable_id`` = incore ``func_id``::

        {"level": 2, "orchestrator_name": "PagedAttn",
         "callable_id_to_name": {"0": "QK", "1": "SF"}}

    * **L3** — ``callable_id`` = index in ``callables`` list::

        {"level": 3, "orchestrator_name": "run_dag",
         "callable_id_to_name": {"0": "vec_kernel", "1": "verify"}}
    """
    if "callables" not in callable_spec:
        # L2: orchestration + incores
        callable_id_to_name: dict[str, str] = {}
        orch = callable_spec.get("orchestration", {})
        orchestrator_name = orch.get("name") if isinstance(orch, dict) else None
        for k in callable_spec.get("incores", []):
            if "name" in k and "func_id" in k:
                callable_id_to_name[str(k["func_id"])] = k["name"]
        result: dict = {"level": 2, "orchestrator_name": orchestrator_name}
        if callable_id_to_name:
            result["callable_id_to_name"] = callable_id_to_name
        return result

    # L3: Python orch function + callables list
    orch = callable_spec.get("orchestration")
    orchestrator_name = getattr(orch, "__name__", None) if callable(orch) else None

    callable_id_to_name = {}
    for idx, entry in enumerate(callable_spec["callables"]):
        callable_id_to_name[str(idx)] = entry.get("name", f"callable_{idx}")

    result = {"level": 3, "orchestrator_name": orchestrator_name}
    if callable_id_to_name:
        result["callable_id_to_name"] = callable_id_to_name
    return result


def _dump_name_map(mapping: dict, output_path: Path) -> Path | None:
    """Write name mapping to JSON if it contains any names. Returns path or None."""
    import json as _json  # noqa: PLC0415

    if not mapping.get("callable_id_to_name") and not mapping.get("orchestrator_name"):
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        _json.dump(mapping, f, indent=2)
    return output_path


def _inject_dist_swimlane_names(callable_spec: dict) -> None:
    """Resolve incore func_ids -> names in a distributed-runtime swimlane.

    The fully_distributed_within_core engine writes a Chrome-trace swimlane to
    ``$PTO_DIST_SWIMLANE`` whose events carry only ``func_id`` (the device has
    no incore name — ``CoreCallable`` stores none). When that capture happened
    this run, rewrite each event's label to the kernel name from the CALLABLE
    spec (``func_id -> name``) and stamp ``args.name`` so both Perfetto and
    ``simpler_setup.tools.dist_swimlane_render`` show e.g. ``GEMM``/``ADD``
    instead of ``f0``/``f1``. No-op when the env var is unset, the file is
    absent, or the spec carries no names.
    """
    import json as _json  # noqa: PLC0415

    path = os.environ.get("PTO_DIST_SWIMLANE")
    if not path or not os.path.isfile(path):
        return
    names = _extract_name_map(callable_spec).get("callable_id_to_name", {})
    if not names:
        return
    try:
        with open(path) as f:
            data = _json.load(f)
    except (OSError, ValueError):
        return
    changed = False
    for e in data.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        args = e.get("args", {})
        nm = names.get(str(args.get("func_id")))
        if nm is None:
            continue
        args["name"] = nm
        tid = args.get("task_id")
        # A distributed-runtime span carries a `phase` (kernel / build / alloc /
        # replay / drain_won). The func_id -> name map only knows the kernel
        # label, so prefix non-kernel phases to keep e.g. a task's `build` span
        # distinct from its `kernel` span on the same lane (both share func_id).
        phase = args.get("phase")
        label = nm if (phase in (None, "kernel")) else f"{phase}:{nm}"
        e["name"] = f"{label}#{tid}" if tid is not None else label
        changed = True
    if changed:
        with open(path, "w") as f:
            _json.dump(data, f, indent=2)


def _parse_case_selector(value: str) -> tuple[str | None, str | None]:
    """Parse one ``--case`` value into ``(class_name, case_name)``.

    ``Foo`` -> ``(None, "Foo")`` (any class)
    ``ClassA::Foo`` -> ``("ClassA", "Foo")``
    ``ClassA::`` -> ``("ClassA", None)`` (all cases in ClassA)
    ``::Foo`` -> ``(None, "Foo")``
    """
    if "::" in value:
        cls_part, case_part = value.split("::", 1)
        return (cls_part or None, case_part or None)
    return (None, value)


def _match_selectors(cls_name: str, case_name: str, selectors: list[tuple]) -> bool:
    """True if ``(cls_name, case_name)`` matches any selector (empty list means no selector filter)."""
    if not selectors:
        return True
    for sel_cls, sel_case in selectors:
        if (sel_cls is None or sel_cls == cls_name) and (sel_case is None or sel_case == case_name):
            return True
    return False


def _select_cases(test_classes, platform: str, selectors: list[tuple], manual_mode: str):
    """Resolve (class, case) pairs to run. Validates selectors strictly.

    Filters: platform match -> selector match -> manual_mode (exclude/include/only).
    Raises ``ValueError`` on unknown selector class/case or empty selection.
    """
    class_index = {c.__name__: c for c in test_classes}
    for sel_cls, _ in selectors:
        if sel_cls is not None and sel_cls not in class_index:
            available = ", ".join(sorted(class_index)) or "(none)"
            raise ValueError(f"--case: unknown class '{sel_cls}'. Available: {available}")
    for sel_cls, sel_case in selectors:
        if sel_case is None:
            continue
        scoped = [class_index[sel_cls]] if sel_cls else test_classes
        if not any(case["name"] == sel_case for c in scoped for case in c.CASES):
            scope = sel_cls or "any class"
            raise ValueError(f"--case: case '{sel_case}' not found in {scope}")

    selected: list[tuple] = []
    for cls in test_classes:
        for case in cls.CASES:
            if platform not in case["platforms"]:
                continue
            if not _match_selectors(cls.__name__, case["name"], selectors):
                continue
            is_manual = bool(case.get("manual"))
            if manual_mode == "exclude" and is_manual:
                continue
            if manual_mode == "only" and not is_manual:
                continue
            selected.append((cls, case))

    if not selected:
        if selectors:
            sel_str = ", ".join(f"{c or '*'}::{n or '*'}" for c, n in selectors)
            hint = " (matches are manual; pass --manual include or only)" if manual_mode == "exclude" else ""
            raise ValueError(f"--case: no cases matched [{sel_str}] for platform={platform}{hint}")
        raise ValueError(f"No cases matched platform={platform} (manual={manual_mode})")
    return selected


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _outputs_dir() -> Path:
    """Root directory under which per-case output prefixes are created."""
    return _project_root() / "outputs"


def _build_output_prefix(case_label: str) -> Path:
    """Per-case directory for diagnostic artifacts.

    Each case gets its own ``outputs/<case_label>_<timestamp>/`` directory; the
    runtime writes ``l2_swimlane_records.json``, ``args_dump/``, and ``pmu.csv``
    under that root with fixed filenames. Two cases of the same name run in
    the same second is not a contemplated scenario (parallel xdist runs differ
    by class+method).

    The directory is created here: the dep_gen host replay (and any other
    writer) ``fopen``s ``<prefix>/<file>`` directly without an mkdir of its
    own, so the prefix must exist before the runtime call.
    """
    from datetime import datetime  # noqa: PLC0415

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = _sanitize_for_filename(case_label)
    prefix = _outputs_dir() / f"{safe_label}_{timestamp}"
    prefix.mkdir(parents=True, exist_ok=True)
    return prefix


def _run_swimlane_converter(
    input_path: Path | None = None,
    func_names_path: Path | None = None,
    enable_overhead: bool = False,
    *,
    strict_fdwic_v4: bool = False,
) -> None:
    """Invoke the bundled swimlane converter as a subprocess.

    When ``input_path`` is given, the converter derives its output filename from
    the input's timestamp (see ``swimlane_converter._resolve_output_path``).
    Without it, the converter auto-selects the latest ``l2_swimlane_records_*.json``.

    ``enable_overhead`` forwards the converter's ``--overhead`` flag — adds the
    8 Overhead Analysis counter tracks (per-engine idle/ready/overhead + system
    all/has overhead) under the AICPU Scheduler process. Needs deps.json; the
    converter silently no-ops if deps is absent.
    """
    import logging  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    cmd = [sys.executable, "-m", "simpler_setup.tools.swimlane_converter"]
    if input_path is not None:
        cmd.append(str(input_path))
    if func_names_path is not None:
        cmd += ["--func-names", str(func_names_path)]
    if enable_overhead:
        cmd.append("--overhead")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        if strict_fdwic_v4:
            if input_path is None:
                raise RuntimeError("strict FDWIC schema-v4 conversion requires an explicit raw input path")
            required_outputs = (
                input_path.parent / "merged_swimlane.json",
                input_path.parent / "swimlane_exclusive_analysis.json",
            )
            missing_outputs = [str(path) for path in required_outputs if not path.is_file() or path.stat().st_size == 0]
            if missing_outputs:
                raise RuntimeError(
                    "FDWIC schema-v4 conversion did not publish required artifact(s): " + ", ".join(missing_outputs)
                )
        logger.info("Swimlane JSON generation completed")
    except subprocess.CalledProcessError as e:
        if strict_fdwic_v4:
            details = e.stderr.strip() or e.stdout.strip() or str(e)
            raise RuntimeError(f"FDWIC schema-v4 conversion/closure validation failed: {details}") from e
        logger.warning(f"Failed to generate swimlane JSON: {e}")
        if e.stdout:
            logger.debug(f"stdout: {e.stdout}")
        if e.stderr:
            logger.debug(f"stderr: {e.stderr}")


def _sanitize_for_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def _convert_case_swimlane(
    case_label: str,
    output_prefix: Path,
    callable_spec: dict | None = None,
    enable_overhead: bool = False,
    *,
    strict_fdwic_v4: bool = False,
) -> None:
    """Post-case: invoke the swimlane converter on the perf file the runtime
    just wrote into ``<output_prefix>/l2_swimlane_records.json``. No diff/rename
    dance — the path is known a priori from CallConfig.output_prefix.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    perf_file = output_prefix / "l2_swimlane_records.json"
    if not perf_file.exists():
        if strict_fdwic_v4:
            raise RuntimeError(f"[{case_label}] required FDWIC schema-v4 raw artifact was not produced: {perf_file}")
        logger.warning(f"[{case_label}] {perf_file} not produced; skipping conversion")
        return

    # Dump callable name mapping if the CALLABLE spec provides names
    func_names_path = None
    if callable_spec:
        mapping = _extract_name_map(callable_spec)
        safe_label = _sanitize_for_filename(case_label)
        func_names_path = _dump_name_map(mapping, output_prefix / f"name_map_{safe_label}.json")

    _run_swimlane_converter(
        input_path=perf_file,
        func_names_path=func_names_path,
        enable_overhead=enable_overhead,
        strict_fdwic_v4=strict_fdwic_v4,
    )


def _run_deps_viewer(
    input_path: Path,
    func_names_path: Path | None = None,
) -> None:
    """Invoke the bundled deps_viewer tool as a subprocess (text mode).

    Produces ``deps_viewer.txt`` next to ``input_path`` by default.
    """
    import logging  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    cmd = [sys.executable, "-m", "simpler_setup.tools.deps_viewer", str(input_path), "--format", "text"]
    if func_names_path is not None:
        cmd += ["--func-names", str(func_names_path)]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        logger.info("deps_viewer text generation completed")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to generate deps_viewer.txt: {e}")
        if e.stdout:
            logger.debug(f"stdout: {e.stdout}")
        if e.stderr:
            logger.debug(f"stderr: {e.stderr}")


def _graph_case_dep_gen(
    case_label: str,
    output_prefix: Path,
    callable_spec: dict | None = None,
) -> None:
    """Post-case: invoke deps_viewer on the deps.json the dep-gen host
    replay just wrote into ``<output_prefix>/deps.json``.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    deps_file = output_prefix / "deps.json"
    if not deps_file.exists():
        logger.warning(f"[{case_label}] {deps_file} not produced; skipping deps_viewer")
        return

    func_names_path = None
    if callable_spec:
        safe_label = _sanitize_for_filename(case_label)
        name_map_path = output_prefix / f"name_map_{safe_label}.json"
        if name_map_path.exists():
            func_names_path = name_map_path
        else:
            mapping = _extract_name_map(callable_spec)
            func_names_path = _dump_name_map(mapping, name_map_path)

    _run_deps_viewer(input_path=deps_file, func_names_path=func_names_path)


def _plot_case_scope_stats(case_label: str, output_prefix: Path) -> None:
    """Post-case: turn ``<output_prefix>/scope_stats/scope_stats.jsonl`` into
    the self-contained scope_stats HTML report. Path is known a priori from
    CallConfig.output_prefix.
    """
    import logging  # noqa: PLC0415

    logger = logging.getLogger(__name__)
    jsonl_file = output_prefix / "scope_stats" / "scope_stats.jsonl"
    if not jsonl_file.exists():
        logger.warning(f"[{case_label}] {jsonl_file} not produced; skipping scope_stats plot")
        return

    import sys  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    tools_dir = _Path(__file__).resolve().parent / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import scope_stats_plot  # noqa: PLC0415

        scope_stats_plot.process(jsonl_file)
    finally:
        sys.path.remove(str(tools_dir))


def _render_case_fdwic_submit_pmu(
    case_label: str, output_prefix: Path, build_identity: SubmitPmuBuildIdentity
) -> Path:
    """Strictly validate the real-PA Submit-PMU raw and publish its HTML."""
    from .tools.fdwic_submit_pmu_report import (  # noqa: PLC0415
        DEFAULT_INPUT_NAME,
        DEFAULT_OUTPUT_NAME,
        write_report_with_provenance,
    )

    raw = output_prefix / DEFAULT_INPUT_NAME
    if not raw.is_file() or raw.stat().st_size == 0:
        raise RuntimeError(f"[{case_label}] submit-PMU did not publish a non-empty {raw}")
    report = output_prefix / DEFAULT_OUTPUT_NAME
    published_report, provenance = write_report_with_provenance(raw, build_identity, report)
    if published_report != report:
        raise RuntimeError(f"[{case_label}] submit-PMU published an unexpected report path {published_report}")
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError(f"[{case_label}] submit-PMU did not publish a non-empty {report}")
    if not provenance.is_file() or provenance.stat().st_size == 0:
        raise RuntimeError(f"[{case_label}] submit-PMU did not publish a non-empty {provenance}")
    return report


def _validate_case_fdwic_perf_clock(  # noqa: PLR0912 -- fail-closed artifact contract is intentionally explicit
    case_label: str, output_prefix: Path, profile: str
) -> Path:
    """Validate the exact artifact contract of one successful perf-clock case."""
    try:
        output_name, schema = _FDWIC_PERF_CLOCK_ARTIFACTS[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported perf-clock artifact profile {profile!r}") from exc

    artifact = output_prefix / output_name
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise RuntimeError(f"[{case_label}] {profile} did not publish a non-empty {artifact}")
    try:
        payload = json.loads(artifact.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"[{case_label}] {artifact} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"[{case_label}] {artifact} root must be a JSON object")

    expected_scalars = {
        "schema": schema,
        "mode": profile,
        "num_cores": 96,
        "aic_cores": 32,
        "aiv_cores": 64,
    }
    mismatches = [
        f"{name}={payload.get(name)!r}, expected {expected!r}"
        for name, expected in expected_scalars.items()
        if payload.get(name) != expected
    ]
    if mismatches:
        raise RuntimeError(f"[{case_label}] invalid {profile} artifact contract: {'; '.join(mismatches)}")

    expected_submits = payload.get("expected_submits_per_core")
    cores = payload.get("cores")
    if type(expected_submits) is not int or expected_submits <= 0:
        raise RuntimeError(f"[{case_label}] {profile} expected_submits_per_core must be a positive integer")
    if not isinstance(cores, list) or len(cores) != 96 or not all(isinstance(core, dict) for core in cores):
        raise RuntimeError(f"[{case_label}] {profile} cores must contain exactly 96 objects")

    core_ids = [core.get("core_id") for core in cores]
    core_types = [core.get("core_type") for core in cores]
    if set(core_ids) != set(range(96)) or len(core_ids) != len(set(core_ids)):
        raise RuntimeError(f"[{case_label}] {profile} core_id topology is not exactly 0..95")
    if core_types.count("aic") != 32 or core_types.count("aiv") != 64:
        raise RuntimeError(f"[{case_label}] {profile} core_type topology is not 32 AIC + 64 AIV")

    for core in cores:
        if core.get("submit_count") != expected_submits:
            raise RuntimeError(
                f"[{case_label}] {profile} core {core.get('core_id')} submit_count does not match "
                "expected_submits_per_core"
            )
        start = core.get("first_submit_start")
        end = core.get("last_submit_end")
        elapsed = core.get("elapsed_ticks")
        ordered_window = (
            type(start) is int
            and type(end) is int
            and type(elapsed) is int
            and start > 0
            and end - start == elapsed
            and (end > start if profile == _FDWIC_PROFILE_PERF_CLOCK_KERNEL else end >= start)
        )
        if not ordered_window:
            raise RuntimeError(f"[{case_label}] {profile} core {core.get('core_id')} elapsed tick closure failed")

    global_start = min(core["first_submit_start"] for core in cores)
    global_end = max(core["last_submit_end"] for core in cores)
    if (
        payload.get("global_first_submit_start") != global_start
        or payload.get("global_last_submit_end") != global_end
        or payload.get("global_submit_span_ticks") != global_end - global_start
    ):
        raise RuntimeError(f"[{case_label}] {profile} global Submit tick closure failed")

    kernel_ticks: list[int] = []
    kernel_calls: list[int] = []
    residual_ticks: list[int] = []
    if profile == _FDWIC_PROFILE_PERF_CLOCK_KERNEL:
        kernel_ticks = [core.get("kernel_elapsed_ticks") for core in cores]
        kernel_calls = [core.get("kernel_calls") for core in cores]
        residual_ticks = [core.get("non_kernel_residual_ticks") for core in cores]
        if not all(type(value) is int and value >= 0 for value in (*kernel_ticks, *kernel_calls, *residual_ticks)):
            raise RuntimeError(f"[{case_label}] {profile} per-core Kernel aggregates must be non-negative integers")

    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise RuntimeError(f"[{case_label}] {profile} groups must be a JSON object")
    for core_type in ("aic", "aiv"):
        group = groups.get(core_type)
        typed_cores = [core for core in cores if core["core_type"] == core_type]
        elapsed_values = [core["elapsed_ticks"] for core in typed_cores]
        if not isinstance(group, dict):
            raise RuntimeError(f"[{case_label}] {profile} groups.{core_type} must be a JSON object")
        expected_group = (
            {
                "min_ticks": min(elapsed_values),
                "max_ticks": max(elapsed_values),
            }
            if profile == _FDWIC_PROFILE_PERF_CLOCK
            else {
                "cores": len(typed_cores),
                "elapsed_min_ticks": min(elapsed_values),
                "elapsed_max_ticks": max(elapsed_values),
                "elapsed_sum_ticks": sum(elapsed_values),
                "kernel_min_ticks": min(core["kernel_elapsed_ticks"] for core in typed_cores),
                "kernel_max_ticks": max(core["kernel_elapsed_ticks"] for core in typed_cores),
                "kernel_sum_ticks": sum(core["kernel_elapsed_ticks"] for core in typed_cores),
                "kernel_calls_min": min(core["kernel_calls"] for core in typed_cores),
                "kernel_calls_max": max(core["kernel_calls"] for core in typed_cores),
                "kernel_calls_sum": sum(core["kernel_calls"] for core in typed_cores),
                "residual_min_ticks": min(core["non_kernel_residual_ticks"] for core in typed_cores),
                "residual_max_ticks": max(core["non_kernel_residual_ticks"] for core in typed_cores),
                "residual_sum_ticks": sum(core["non_kernel_residual_ticks"] for core in typed_cores),
            }
        )
        group_mismatches = [
            f"{name}={group.get(name)!r}, recomputed {expected}"
            for name, expected in expected_group.items()
            if type(group.get(name)) is not int or group.get(name) != expected
        ]
        if group_mismatches:
            raise RuntimeError(
                f"[{case_label}] invalid {profile} groups.{core_type} integer aggregates: {'; '.join(group_mismatches)}"
            )

    if profile == _FDWIC_PROFILE_PERF_CLOCK_KERNEL:
        if any(
            kernel + residual != core["elapsed_ticks"]
            for core, kernel, residual in zip(cores, kernel_ticks, residual_ticks, strict=True)
        ):
            raise RuntimeError(f"[{case_label}] {profile} per-core Kernel/residual tick closure failed")
        if any((calls == 0) != (ticks == 0) for calls, ticks in zip(kernel_calls, kernel_ticks, strict=True)):
            raise RuntimeError(f"[{case_label}] {profile} per-core Kernel call/tick presence closure failed")
        exact_aggregates = {
            "kernel_calls": sum(kernel_calls),
            "kernel_elapsed_ticks_sum": sum(kernel_ticks),
            "non_kernel_residual_ticks_sum": sum(residual_ticks),
        }
        aggregate_mismatches = [
            f"{name}={payload.get(name)!r}, recomputed {expected}"
            for name, expected in exact_aggregates.items()
            if payload.get(name) != expected
        ]
        if aggregate_mismatches:
            raise RuntimeError(
                f"[{case_label}] invalid {profile} integer aggregates: {'; '.join(aggregate_mismatches)}"
            )
        min_calls = payload.get("min_kernel_calls_in_window")
        max_calls = payload.get("max_kernel_calls_in_window")
        batches, remainder = divmod(expected_submits, 5)
        aic_calls = sum(core["kernel_calls"] for core in cores if core["core_type"] == "aic")
        aiv_calls = sum(core["kernel_calls"] for core in cores if core["core_type"] == "aiv")
        if (
            remainder != 0
            or type(min_calls) is not int
            or type(max_calls) is not int
            or min_calls != batches
            or max_calls != 4 * batches
            or not batches <= aic_calls <= 2 * batches
            or not 0 <= aiv_calls <= 2 * batches
            or not batches <= exact_aggregates["kernel_calls"] <= 4 * batches
        ):
            raise RuntimeError(f"[{case_label}] {profile} global Kernel call range closure failed")
    return artifact


def _format_case_context(cls_name: str, case: dict, worker) -> str:
    platform = worker._config.get("platform", "<unknown>")
    config = case.get("config", {})
    params = case.get("params", {})
    return f"{cls_name}_{case['name']} platform={platform} config={config!r} params={params!r}"


def run_class_cases(  # noqa: PLR0913 -- shared layer-5 entry; kwargs mirror CLI surface
    worker,
    cls_inst,
    cases,
    *,
    callable_obj,
    sub_handles,
    rounds,
    skip_golden,
    enable_l2_swimlane,
    enable_dump_args,
    enable_pmu,
    enable_dep_gen,
    enable_scope_stats,
    enable_device_log_timing=False,
    enable_swimlane_overhead=False,
):
    """Execute a pre-filtered list of cases for one class (layers 5-6).

    Caller is responsible for platform/selector/manual filtering. Profiling
    snapshots wrap each case. Validation failures propagate; caller decides
    fail-fast vs collect semantics.
    """
    cls_name = type(cls_inst).__name__
    callable_spec = getattr(type(cls_inst), "CALLABLE", None)
    fdwic_profile = _fdwic_profile()
    private_profile_on = fdwic_profile in _FDWIC_PRIVATE_PROFILES
    submit_pmu_build_identity = None
    if fdwic_profile in _FDWIC_SUBMIT_PMU_PROFILES:
        platform = worker._config.get("platform")
        runtime = getattr(type(cls_inst), "_st_runtime", None)
        build_identity_key = _profiled_cache_key((type(cls_inst).__qualname__, platform, runtime))
        submit_pmu_build_identity = _fdwic_build_identity_cache.get(build_identity_key)
        if submit_pmu_build_identity is None:
            raise RuntimeError(
                "Submit-PMU build identity is missing for "
                f"profiled cache key {build_identity_key!r}; refusing to run without build provenance"
            )
    diagnostics_on = (
        enable_l2_swimlane
        or enable_dump_args
        or enable_pmu
        or enable_dep_gen
        or enable_scope_stats
        or private_profile_on
    )
    # device-log timing wraps each case here (not inside _run_and_validate*),
    # the same way swimlane conversion does — _run_and_validate_l2 is overridden
    # by some SceneTestCase subclasses, so threading a kwarg through it would
    # break those overrides. Onboard L2 only; the orch/sched markers don't exist
    # on sim and an L3 run spans multiple chip logs.
    dlt_on = (
        enable_device_log_timing
        and getattr(cls_inst, "_st_level", None) == 2
        and not str(worker._config.get("platform", "")).endswith("sim")
    )
    dlt_device_id = worker._config.get("device_id", 0)
    if enable_device_log_timing and getattr(cls_inst, "_st_level", None) == 3:
        logger.warning("device-log timing is not supported for L3 (multi-chip); ignoring")
    for case in cases:
        case_label = f"{cls_name}_{case['name']}"
        case_context = _format_case_context(cls_name, case, worker)
        print(f"[scene_test] RUN CASE {case_context}", flush=True)
        # Per-case directory the runtime writes into. Required (non-empty) when
        # any diagnostic flag is on; CallConfig::validate() throws otherwise.
        # scope_stats now writes <prefix>/scope_stats/scope_stats.jsonl (sibling of
        # l2_swimlane_records.json / deps.json), so it pulls output_prefix the
        # same way the other DFX flags do.
        prefix = _build_output_prefix(case_label) if diagnostics_on else Path("")
        dlt_baseline = _snapshot_time() if dlt_on else None
        dlt_offsets = _snapshot_log_offsets(_get_device_log_dir(dlt_device_id)) if dlt_on else None
        case_succeeded = False
        try:
            cls_inst._run_and_validate(
                worker,
                callable_obj,
                case,
                sub_handles=sub_handles,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_args=enable_dump_args,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                enable_scope_stats=enable_scope_stats,
                output_prefix=str(prefix) if diagnostics_on else "",
            )
            case_succeeded = True
        except BaseException as exc:
            exc.add_note(f"SceneTest case context: {case_context}")
            raise
        finally:
            if enable_l2_swimlane:
                strict_fdwic_v4 = (
                    case_succeeded
                    and enable_l2_swimlane == 4
                    and getattr(cls_inst, "_st_runtime", None) == "fully_distributed_within_core"
                    and worker._config.get("platform") == "a5"
                )
                _convert_case_swimlane(
                    case_label,
                    prefix,
                    callable_spec=callable_spec,
                    enable_overhead=enable_swimlane_overhead,
                    strict_fdwic_v4=strict_fdwic_v4,
                )
            if enable_dep_gen:
                _graph_case_dep_gen(case_label, prefix, callable_spec=callable_spec)
            if enable_scope_stats:
                _plot_case_scope_stats(case_label, prefix)
            if case_succeeded and fdwic_profile in _FDWIC_SUBMIT_PMU_PROFILES:
                _render_case_fdwic_submit_pmu(case_label, prefix, submit_pmu_build_identity)
            if case_succeeded and fdwic_profile in _FDWIC_PERF_CLOCK_PROFILES:
                _validate_case_fdwic_perf_clock(case_label, prefix, fdwic_profile)
            if dlt_baseline is not None:
                _print_device_log_timing(dlt_device_id, dlt_baseline, dlt_offsets, rounds)


def _compare_outputs(test_args, golden_args, output_names, rtol, atol):
    """Compare output tensors against golden values."""
    import torch  # noqa: PLC0415

    mismatches = []
    for name in output_names:
        actual = getattr(test_args, name)
        expected = getattr(golden_args, name)
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            diff = (actual - expected).abs().max().item()
            mismatches.append(
                f"Golden mismatch on '{name}': max_diff={diff}, rtol={rtol}, atol={atol}, "
                f"actual_head={actual.flatten()[:128].tolist()}, expected_head={expected.flatten()[:128].tolist()}"
            )
    if mismatches:
        raise AssertionError("\n".join(mismatches))


def _compile_chip_callable_from_spec(spec, platform, runtime, cache_key):
    """Compile a chip entry spec (orchestration + incores) -> ChipCallable. Session-cached."""
    cache_key = _profiled_cache_key(cache_key)
    if cache_key in _compile_cache:
        return _compile_cache[cache_key]

    from simpler.task_interface import ChipCallable, CoreCallable  # noqa: PLC0415

    from .elf_parser import extract_text_section  # noqa: PLC0415
    from .kernel_compiler import KernelCompiler  # noqa: PLC0415
    from .pto_isa import ensure_pto_isa_root  # noqa: PLC0415

    orch = spec["orchestration"]
    incores = spec["incores"]

    pto_isa_root = ensure_pto_isa_root()
    kc = KernelCompiler(platform=platform)
    is_sim = platform.endswith("sim")

    orch_binary = kc.compile_orchestration(runtime, orch["source"])
    inc_dirs = kc.get_orchestration_include_dirs(runtime)

    kernel_binaries = []
    for k in incores:
        incore = kc.compile_incore(
            k["source"], core_type=k["core_type"], pto_isa_root=pto_isa_root, extra_include_dirs=inc_dirs
        )
        if not is_sim and runtime != "fully_distributed_within_core":
            incore = extract_text_section(incore)
        kernel_binaries.append((k["func_id"], CoreCallable.build(signature=k.get("signature", []), binary=incore)))

    chip_callable = ChipCallable.build(
        signature=orch.get("signature", []),
        func_name=orch["function_name"],
        binary=orch_binary,
        children=kernel_binaries,
        config_name=orch.get("config_name", ""),
    )
    aicore_override = maybe_build_aicore_override(
        cache_key, platform, runtime, orch["source"], incores, pto_isa_root=pto_isa_root
    )
    if aicore_override is not None:
        _aicore_override_cache[cache_key] = aicore_override
    _compile_cache[cache_key] = chip_callable
    return chip_callable


# ---------------------------------------------------------------------------
# @scene_test decorator
# ---------------------------------------------------------------------------


def scene_test(level: int, runtime: str):
    """Decorator marking a SceneTestCase with level and runtime.

    Platforms are declared per-case in CASES, not here.
    """

    def decorator(cls):
        cls._st_level = level
        cls._st_runtime = runtime
        cls_dir = Path(inspect.getfile(cls)).parent
        if hasattr(cls, "CALLABLE"):
            _resolve_callable_paths(cls, cls_dir)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# SceneTestCase base class
# ---------------------------------------------------------------------------


class SceneTestCase:
    """Base class for scene tests at any hierarchy level.

    Subclasses declare CALLABLE, CASES, generate_args(), compute_golden().
    """

    CALLABLE: dict = {}
    CASES: list[dict] = []
    RTOL: float = 1e-5
    ATOL: float = 1e-5
    RUNTIME_ENV: dict = {}

    def generate_args(self, params) -> TaskArgsBuilder:
        """Return TaskArgsBuilder with ordered Tensor/Scalar specs."""
        raise NotImplementedError

    def compute_golden(self, args: TaskArgsBuilder, params) -> None:
        """Compute expected outputs in-place on a cloned TaskArgsBuilder."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Callable compilation
    # ------------------------------------------------------------------

    @classmethod
    def compile_chip_callable(cls, platform):
        """Compile CALLABLE -> ChipCallable (L2). Session-cached."""
        cache_key = (cls.__qualname__, platform, cls._st_runtime)
        return _compile_chip_callable_from_spec(cls.CALLABLE, platform, cls._st_runtime, cache_key)

    @classmethod
    def _compile_l3_callables(cls, platform):
        """Compile all ChipCallable entries in CALLABLE['callables'] (L3)."""
        compiled = {}
        for entry in cls.CALLABLE["callables"]:
            if "orchestration" in entry:
                name = entry["name"]
                cache_key = (cls.__qualname__, name, platform, cls._st_runtime)
                chip = _compile_chip_callable_from_spec(entry, platform, cls._st_runtime, cache_key)
                compiled[name] = chip
                compiled[f"{name}_sig"] = entry["orchestration"].get("signature", [])
        return compiled

    # ------------------------------------------------------------------
    # Worker creation
    # ------------------------------------------------------------------

    @classmethod
    def _create_worker(cls, platform, device_id=0):
        """Create the L2 Worker for the standalone path.

        Mirrors the ``st_worker`` pytest fixture, which yields a ``Worker``
        (not a raw ``ChipWorker``) — ``_run_and_validate_l2`` is shared by both
        paths and calls ``worker.register(...)`` / ``worker.run(handle, ...)``,
        which only the ``Worker`` wrapper exposes.
        """
        from simpler.worker import Worker  # noqa: PLC0415

        cache_key = (cls.__qualname__, platform, cls._st_runtime)
        cls.compile_chip_callable(platform)
        aicore_override = get_aicore_path_override(cache_key)
        kwargs = {}
        if aicore_override is not None:
            kwargs["aicore_path_override"] = aicore_override
        w = Worker(level=2, device_id=device_id, platform=platform, runtime=cls._st_runtime, **kwargs)
        w.init()
        return w

    # ------------------------------------------------------------------
    # Default build methods
    # ------------------------------------------------------------------

    def build_callable(self, platform):
        """Build callable for the current level.

        L2: returns ChipCallable.
        L3: returns dict of {name: ChipCallable, name_sig: signature}.
        """
        if self._st_level == 2:
            return self.compile_chip_callable(platform)
        elif self._st_level == 3:
            return self._compile_l3_callables(platform)
        raise ValueError(f"Unsupported level: {self._st_level}")

    def _build_config(
        self,
        config_dict,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        *,
        output_prefix="",
    ):
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        config = CallConfig()
        # Default to 0 (CallConfig "auto" sentinel) when a case omits
        # block_dim — DeviceRunner resolves it to the stream's max capacity
        # at run() time. Cases that need a specific value still set it
        # explicitly in their config dict.
        config.block_dim = config_dict.get("block_dim", 0)
        config.aicpu_thread_num = config_dict.get("aicpu_thread_num", 3)
        # Per-task ring sizing (tensormap_and_ringbuffer only; 0 = unset),
        # nested under the "runtime_env" key. Takes precedence over the
        # PTO2_RING_* env vars / RUNTIME_ENV.
        runtime_env = config_dict.get("runtime_env", {})
        config.runtime_env.ring_task_window = runtime_env.get("ring_task_window", 0)
        config.runtime_env.ring_heap = runtime_env.get("ring_heap", 0)
        config.runtime_env.ring_dep_pool = runtime_env.get("ring_dep_pool", 0)
        if "ring_task_windows" in runtime_env:
            config.runtime_env.ring_task_windows = runtime_env["ring_task_windows"]
        if "ring_heaps" in runtime_env:
            config.runtime_env.ring_heaps = runtime_env["ring_heaps"]
        if "ring_dep_pools" in runtime_env:
            config.runtime_env.ring_dep_pools = runtime_env["ring_dep_pools"]
        config.enable_l2_swimlane = enable_l2_swimlane
        config.enable_dump_tensor = enable_dump_args
        config.enable_pmu = enable_pmu  # 0=disabled, >0=enabled with event type
        config.enable_dep_gen = enable_dep_gen
        config.enable_scope_stats = enable_scope_stats
        # Sim-only trace-driven replay: only fully_distributed_within_core
        # implements it; every other runtime rejects the flag here so no other
        # runtime needs to adapt (the C++ side leaves it a weak no-op). When on,
        # each incore's CALLABLE example_exec_time_ns (nanoseconds) is plumbed
        # to the runtime, which busy-waits it instead of running the real kernel.
        if getattr(self, "_st_use_example_exec_time", 0):
            if self._st_runtime != "fully_distributed_within_core":
                raise NotImplementedError(
                    "--use-example-exec-time is only supported by the "
                    "fully_distributed_within_core runtime, not "
                    f"{self._st_runtime!r}"
                )
            config.use_example_exec_time = 1
            config.example_exec_time_ns = self._collect_example_exec_time_ns()
        # `output_prefix` is required by CallConfig::validate() whenever any
        # diagnostic flag is enabled. Caller threads it down from the per-case
        # directory built by _build_output_prefix().
        if output_prefix:
            config.output_prefix = str(output_prefix)
        return config

    def _collect_example_exec_time_ns(self):
        """Per-func reference durations (nanoseconds) keyed by func_id, read from
        the CALLABLE ``incores`` ``example_exec_time_ns`` fields (an integer
        nanosecond count). A func without the field stays 0 (runs for real under
        use_example_exec_time)."""
        incores = self.CALLABLE.get("incores", [])
        max_fid = max((k.get("func_id", -1) for k in incores), default=-1)
        table = [0] * (max_fid + 1)
        for k in incores:
            fid = k.get("func_id")
            ns = k.get("example_exec_time_ns")
            if ns is not None and isinstance(fid, int) and fid >= 0:
                table[fid] = int(ns)
        return table

    def _resolve_env(self):
        env = self.RUNTIME_ENV
        if not env:
            return {}
        cls_dir = Path(inspect.getfile(type(self))).parent
        out = {}
        for k, v in env.items():
            s = str(v)
            if (k.endswith("_DIR") or k.endswith("_PATH")) and not Path(s).is_absolute():
                s = str((cls_dir / s).resolve())
            out[k] = s
        return out

    # ------------------------------------------------------------------
    # Run + validate
    # ------------------------------------------------------------------

    def _run_and_validate(  # noqa: PLR0913 -- threads CLI diagnostic flags + case context
        self,
        worker,
        callable_obj,
        case,
        sub_handles=None,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        if self._st_level == 2:
            self._run_and_validate_l2(
                worker,
                callable_obj,
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_args=enable_dump_args,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                enable_scope_stats=enable_scope_stats,
                output_prefix=output_prefix,
            )
        elif self._st_level == 3:
            self._run_and_validate_l3(
                worker,
                callable_obj,
                sub_handles or {},
                case,
                rounds=rounds,
                skip_golden=skip_golden,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_args=enable_dump_args,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                enable_scope_stats=enable_scope_stats,
                output_prefix=output_prefix,
            )

    def _run_and_validate_l2(  # noqa: PLR0913 -- threads CLI diagnostic flags + case context
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        params = case.get("params", {})
        config_dict = case.get("config", {})
        orch_sig = self.CALLABLE.get("orchestration", {}).get("signature", [])

        # The L2 entry point is `Worker.run(handle, args, cfg)`. Reuse the
        # handle registered by the st_worker fixture / standalone path.
        handle = getattr(type(self), "_st_l2_handle", None)
        if handle is None:
            handle = worker.register(callable_obj)
            type(self)._st_l2_handle = handle

        # Build args
        test_args = self.generate_args(params)
        chip_args, output_names = _build_chip_task_args(test_args, orch_sig)

        # Compute golden (unless skip_golden)
        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        # Save initial output tensor values for reset between rounds
        initial_outputs = {}
        if rounds > 1:
            for name in output_names:
                initial_outputs[name] = getattr(test_args, name).clone()

        # Execute rounds
        timings = []  # populated only when rounds > 1
        for round_idx in range(rounds):
            if round_idx > 0:
                for name, initial in initial_outputs.items():
                    getattr(test_args, name).copy_(initial)

            # enable_l2_swimlane / enable_dep_gen are already forced False by
            # the upstream gate in test_run / run_module when rounds > 1, so an
            # extra `and round_idx == 0` here is dead code; pass them through
            # verbatim. (If the upstream gate is ever relaxed, restore the
            # per-round masking here.)
            config = self._build_config(
                config_dict,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_args=enable_dump_args,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                enable_scope_stats=enable_scope_stats,
                output_prefix=output_prefix,
            )

            with _temporary_env(self._resolve_env()):
                timing = worker.run(handle, chip_args, config=config)
            if rounds > 1 and timing is not None:
                timings.append((timing.host_wall_us, timing.device_wall_us))

            if not skip_golden:
                _compare_outputs(test_args, golden_args, output_names, self.RTOL, self.ATOL)

        if timings:
            _log_round_timings(timings)

        # If a distributed-runtime swimlane was captured (PTO_DIST_SWIMLANE),
        # label its events with the incore function names from the spec.
        _inject_dist_swimlane_names(self.CALLABLE)

    def _run_and_validate_l3(  # noqa: PLR0913 -- threads CLI diagnostic flags + L3 ns context
        self,
        worker,
        compiled_callables,
        sub_handles,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=0,
        enable_dump_args=False,
        enable_pmu=0,
        enable_dep_gen=False,
        enable_scope_stats=False,
        output_prefix="",
    ):
        # Defensive belt-and-braces: the pytest dispatcher and run_module both
        # block --enable-l2-swimlane for L3 at the CLI boundary. Catch any code
        # path that reaches here with the flag on anyway (direct API use,
        # future refactors) so we fail loud rather than produce garbage perf
        # files. Lift once the runtime embeds device_id in the perf filename.
        if enable_l2_swimlane:
            raise NotImplementedError(
                "L3 profiling is not supported yet (multi-chip-process perf "
                "filename collision). Gate at the CLI level in "
                "conftest.pytest_collection_modifyitems / scene_test.run_module."
            )

        params = case.get("params", {})
        config_dict = case.get("config", {})

        # Build args
        test_args = self.generate_args(params)

        # Compute golden (unless skip_golden)
        golden_args = None
        if not skip_golden:
            golden_args = test_args.clone()
            self.compute_golden(golden_args, params)

        # Save initial tensor values for reset between rounds
        all_tensor_names = test_args.tensor_names()
        initial_tensors = {}
        if rounds > 1:
            for name in all_tensor_names:
                initial_tensors[name] = getattr(test_args, name).clone()

        # Build CallableNamespace: compiled ChipCallables + sub callable IDs
        ns = CallableNamespace({**compiled_callables, **sub_handles})

        # Get orch function (plain function from CALLABLE)
        orch_fn = self.CALLABLE["orchestration"]

        # Execute rounds
        timings = []
        for round_idx in range(rounds):
            if round_idx > 0:
                for name, initial in initial_tensors.items():
                    getattr(test_args, name).copy_(initial)

            # See _run_and_validate_l2: the per-round masking is dead code
            # under the existing upstream gate. Keep parity by passing through.
            config = self._build_config(
                config_dict,
                enable_l2_swimlane=enable_l2_swimlane,
                enable_dump_args=enable_dump_args,
                enable_pmu=enable_pmu,
                enable_dep_gen=enable_dep_gen,
                enable_scope_stats=enable_scope_stats,
                output_prefix=output_prefix,
            )

            # Orch fn signature: (orch, args, cfg) — inner fn forwards to
            # the user's scene orch which takes (orch, callables, task_args, config).
            def task_orch(orch, _args, _cfg, _ns=ns, _test_args=test_args, _config=config):
                orch_fn(orch, _ns, _test_args, _config)

            with _temporary_env(self._resolve_env()):
                timing = worker.run(task_orch)
            if rounds > 1 and timing is not None:
                # L3+ DAGs surface host wall only; device_wall_us is 0 because
                # per-task device cycles aren't aggregated up to Worker.run.
                timings.append((timing.host_wall_us, timing.device_wall_us))

            if not skip_golden:
                _compare_outputs(test_args, golden_args, all_tensor_names, self.RTOL, self.ATOL)

        if timings:
            _log_round_timings(timings)

        # If a distributed-runtime swimlane was captured (PTO_DIST_SWIMLANE),
        # label its events with the incore function names from the spec.
        _inject_dist_swimlane_names(self.CALLABLE)

    # ------------------------------------------------------------------
    # pytest auto test method
    # ------------------------------------------------------------------

    @staticmethod
    def _effective_enable_dep_gen(request, *, warn: bool = False) -> bool:
        """``--enable-dep-gen`` CLI value after applying the ``--rounds > 1``
        disable. Single source of truth so the framework's ``test_run`` loop
        and any subclass override (e.g. ``TestDepGenCapture``'s post-validate
        hook) can't drift on the gating rule. Pass ``warn=True`` from the
        framework's first call — it owns the user-facing "disabled because
        rounds > 1" message; subclass overrides leave ``warn`` off since
        ``super().test_run()`` already warned."""
        if not request.config.getoption("--enable-dep-gen", default=False):
            return False
        if request.config.getoption("--rounds", default=1) > 1:
            if warn:
                logger.warning("dep_gen disabled: --rounds > 1")
            return False
        return True

    def test_run(self, st_platform, st_worker, request):
        """Auto test method — runs matching cases for the current platform."""
        raw_selectors = request.config.getoption("--case", default=None) or []
        selectors = [_parse_case_selector(v) for v in raw_selectors]
        manual_mode = request.config.getoption("--manual", default="exclude")
        rounds = request.config.getoption("--rounds", default=1)
        skip_golden = request.config.getoption("--skip-golden", default=False)
        enable_l2_swimlane = request.config.getoption("--enable-l2-swimlane", default=0)
        enable_dump_args = request.config.getoption("--dump-args", default=0)
        enable_pmu = request.config.getoption("--enable-pmu", default=0)
        enable_dep_gen = self._effective_enable_dep_gen(request, warn=True)
        enable_scope_stats = request.config.getoption("--enable-scope-stats", default=False)
        # Sim-only trace-driven replay switch; consumed in _build_config (which
        # enforces the fully_distributed_within_core-only constraint and reads
        # the per-func durations from CALLABLE). Stashed on the instance so it
        # reaches _build_config without threading through every run variant.
        self._st_use_example_exec_time = 1 if request.config.getoption("--use-example-exec-time", default=False) else 0
        # With trace-driven replay on, incore kernels are skipped (busy-wait
        # only) so their outputs are never computed — force-skip the golden
        # comparison regardless of --skip-golden.
        if self._st_use_example_exec_time:
            skip_golden = True
        # device-log timing is cheap (PTO2_PROFILING markers, one block/round)
        # so unlike the heavy diagnostics it is NOT disabled when --rounds > 1.
        enable_device_log_timing = request.config.getoption("--enable-device-log-timing", default=False)
        enable_swimlane_overhead = request.config.getoption("--enable-swimlane-overhead", default=False)
        if rounds > 1:
            if enable_l2_swimlane:
                logger.warning("Profiling disabled: --rounds > 1")
                enable_l2_swimlane = 0
            if enable_dump_args:
                logger.warning("Dump args disabled: --rounds > 1")
                enable_dump_args = 0
            if enable_pmu:
                logger.warning("PMU disabled: --rounds > 1")
                enable_pmu = 0
            if enable_scope_stats:
                logger.warning("scope_stats disabled: --rounds > 1")
                enable_scope_stats = False

        cls_name = type(self).__name__
        callable_obj = self.build_callable(st_platform)
        sub_handles = getattr(type(self), "_st_sub_handles", {})
        # For L3, use registered chip handles instead of raw ChipCallable
        # objects.
        chip_handles = getattr(type(self), "_st_chip_handles", {})
        if self._st_level == 3 and chip_handles:
            callable_obj = {**chip_handles}

        matched = []
        for case in self.CASES:
            if st_platform not in case["platforms"]:
                continue
            if not _match_selectors(cls_name, case["name"], selectors):
                continue
            is_manual = bool(case.get("manual"))
            if manual_mode == "exclude" and is_manual:
                continue
            if manual_mode == "only" and not is_manual:
                continue
            matched.append(case)

        if not matched:
            import pytest  # noqa: PLC0415

            pytest.skip(f"No cases matched {cls_name} (platform={st_platform}, manual={manual_mode})")

        run_class_cases(
            st_worker,
            self,
            matched,
            callable_obj=callable_obj,
            sub_handles=sub_handles,
            rounds=rounds,
            skip_golden=skip_golden,
            enable_l2_swimlane=enable_l2_swimlane,
            enable_dump_args=enable_dump_args,
            enable_pmu=enable_pmu,
            enable_dep_gen=enable_dep_gen,
            enable_scope_stats=enable_scope_stats,
            enable_device_log_timing=enable_device_log_timing,
            enable_swimlane_overhead=enable_swimlane_overhead,
        )

    # ------------------------------------------------------------------
    # Standalone entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run_module(module_name):  # noqa: PLR0912, PLR0915 -- CLI parsing + dispatch; branches map to user-facing flags
        """Standalone entry: ``if __name__ == "__main__": SceneTestCase.run_module(__name__)``.

        Supports -d as either a single id or a range ("0-7"). When more than
        one device is provided (or any L3 case needs more than its single
        device), the outer invocation becomes a test dispatcher that spawns
        per-case subprocesses via ``parallel_scheduler``; each child re-enters
        this function in single-group mode via ``--runtime`` + ``--level``.
        """
        import argparse  # noqa: PLC0415

        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--platform", required=True)
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="0",
            help="Device id or range ('0', '4-7', '0,2,5')",
        )
        parser.add_argument(
            "--sanitizer",
            default="none",
            help=(
                "Run against sanitizer-built binaries (asan/ubsan/tsan or raw -fsanitize "
                "tokens). Must match the installed runtime's SIMPLER_SANITIZER and needs "
                "the runtime preloaded, e.g. LD_PRELOAD=$(g++ -print-file-name=libasan.so)."
            ),
        )
        parser.add_argument(
            "--case",
            action="append",
            default=None,
            help="Case selector; repeatable. Forms: 'Foo' (any class), 'ClassA::Foo', 'ClassA::'",
        )
        parser.add_argument(
            "--manual",
            choices=["exclude", "include", "only"],
            default="exclude",
            help="Manual case handling: exclude (default), include, only",
        )
        parser.add_argument("--rounds", type=int, default=1, help="Run each case N times (default: 1)")
        parser.add_argument("--skip-golden", action="store_true", help="Skip golden comparison (benchmark mode)")
        parser.add_argument(
            "--enable-l2-swimlane",
            nargs="?",
            const=4,
            default=0,
            type=int,
            metavar="PERF_LEVEL",
            help="Enable L2 swimlane. Bare flag=level 4 (full). "
            "1=AICore timing, 2=+dispatch/fanout, 3=+sched phases, 4=+orch phases",
        )
        parser.add_argument(
            "--enable-device-log-timing",
            action="store_true",
            help="After the run, parse the CANN device log and print per-round Total / Orch / "
            "Sched timing (from PTO2_PROFILING markers; no swimlane needed). Works with --rounds N "
            "(one row per round). Onboard only — ignored on sim and L3.",
        )
        parser.add_argument(
            "--dump-args",
            nargs="?",
            const=1,
            type=int,
            default=0,
            help="Dump per-task args at runtime. Level: 0=off, 1=partial (only "
            "tasks marked via Arg::dump(...), default when given without a value), "
            "2=full (all tasks), 3=full_json_only (all tasks, JSON metadata only, no .bin payload).",
        )
        parser.add_argument(
            "--enable-dep-gen",
            action="store_true",
            help="Enable dep_gen capture (SubmitTrace ring, first round only)",
        )
        parser.add_argument(
            "--enable-pmu",
            nargs="?",
            const=2,
            default=0,
            type=int,
            metavar="EVENT_TYPE",
            help="Enable PMU collection. Bare flag = PIPE_UTILIZATION(2). "
            "Pass event type to override (e.g. --enable-pmu 4)",
        )
        parser.add_argument(
            "--enable-scope-stats",
            action="store_true",
            default=False,
            help="Enable per-scope peak collection and emit <output_prefix>/scope_stats/scope_stats.jsonl "
            "(per-scope ring-fill peaks).",
        )
        parser.add_argument(
            "--enable-swimlane-overhead",
            action="store_true",
            default=False,
            help="Add the 8 Overhead Analysis counter tracks (per-engine "
            "idle/ready/overhead + system all/has overhead) to the swimlane "
            "JSON. Requires --enable-l2-swimlane + deps.json (re-run with "
            "--enable-dep-gen if absent).",
        )
        parser.add_argument(
            "--runtime",
            default=None,
            help="Only run classes with this _st_runtime (child-mode marker when combined with --level)",
        )
        parser.add_argument(
            "--level",
            type=int,
            choices=[2, 3],
            default=None,
            help="Only run classes with this _st_level (child-mode marker when combined with --runtime)",
        )
        parser.add_argument(
            "-x", "--exitfirst", action="store_true", help="Stop on first failing case (matches pytest -x)"
        )
        parser.add_argument(
            "--max-parallel",
            default="auto",
            help=(
                "Max in-flight subprocesses (make-style); decouples -d pool size from "
                "parallelism. 'auto' = min(nproc, len(-d)) on sim, len(-d) on hardware. "
                "Use e.g. '--max-parallel 2' to throttle sim on a CPU-constrained CI "
                "runner without shrinking -d. No short form — pytest reserves lowercase "
                "shorts; standalone mirrors that restriction for consistency."
            ),
        )
        parser.add_argument(
            "-c",
            "--pto-isa-commit",
            default=None,
            help=("Override the PTO-ISA revision before running. Default/latest: use the current checkout HEAD."),
        )
        parser.add_argument(
            "--clone-protocol",
            choices=["ssh", "https"],
            default="ssh",
            help="Git protocol for auto-cloning PTO-ISA when PTO_ISA_ROOT is not set. Default: ssh.",
        )
        parser.add_argument(
            "--log-level",
            choices=LOG_LEVEL_CHOICES,
            default=DEFAULT_LOG_LEVEL,
            help=f"Simpler logger level (debug/V0..V9/info/warn/error/null; default {DEFAULT_LOG_LEVEL})",
        )
        args = parser.parse_args()
        configure_logging(args.log_level)

        # Match the per-test kernel/orchestration compile to the runtime's
        # sanitizer, and require the runtime preloaded — same as conftest, since
        # the standalone path skips it.
        from . import sanitizers as _san  # noqa: PLC0415

        _san_tokens = _san.resolve(args.sanitizer)
        if _san_tokens:
            try:
                _san.validate(_san_tokens)
            except ValueError as e:
                parser.error(f"--sanitizer={args.sanitizer}: {e}")
            from .kernel_compiler import KernelCompiler  # noqa: PLC0415

            KernelCompiler._sanitizers = _san_tokens
            _lib = _san.preload_lib(_san_tokens)
            if _lib and not _san.is_runtime_loaded(_lib):
                parser.error(
                    f"--sanitizer={args.sanitizer} needs the {_lib} runtime preloaded. Re-run with:\n"
                    f"  {_san.preload_command(_san_tokens, args.platform)} python {module_name} ..."
                )

        os.environ["PTO_ISA_ROOT"] = ensure_pto_isa_root(
            commit=args.pto_isa_commit,
            clone_protocol=args.clone_protocol,
            update_if_exists=True,
            verbose=True,
        )

        if args.rounds > 1 and args.enable_l2_swimlane:
            logger.warning("Profiling disabled: --rounds > 1")
            args.enable_l2_swimlane = 0
        if args.rounds > 1 and args.enable_dep_gen:
            logger.warning("dep_gen disabled: --rounds > 1")
            args.enable_dep_gen = False
        if args.rounds > 1 and args.enable_scope_stats:
            logger.warning("scope_stats disabled: --rounds > 1")
            args.enable_scope_stats = False

        from .parallel_scheduler import default_max_parallel, device_range_to_list  # noqa: PLC0415

        device_ids = device_range_to_list(args.device)
        if not device_ids:
            print("ERROR: --device must be a non-empty id or range", file=sys.stderr)
            sys.exit(2)
        args.device_ids = device_ids
        # Keep ``args.device`` as an int for paths that expect a single id
        # (profiling snapshots, device-id binding inside one worker). In child
        # mode this is the single allocated id; in parent mode we use the first
        # slot but the dispatcher doesn't actually run tests here.
        args.device = device_ids[0]

        # Resolve -j (max parallel) — 'auto' is CPU-aware on sim, device-count on hardware.
        if args.max_parallel in (None, "", "auto"):
            args.max_parallel = default_max_parallel(args.platform, device_ids)
        else:
            try:
                args.max_parallel = int(args.max_parallel)
            except (TypeError, ValueError):
                print(f"ERROR: -j must be 'auto' or an integer, got {args.max_parallel!r}", file=sys.stderr)
                sys.exit(2)
            if args.max_parallel < 1:
                print(f"ERROR: -j must be >= 1, got {args.max_parallel}", file=sys.stderr)
                sys.exit(2)
        # Profiling + parallelism is safe: each test case sets its own
        # `output_prefix` on CallConfig (see run_class_cases) so diagnostic
        # artifacts land in distinct directories with no shared filenames.

        module = sys.modules[module_name]
        test_classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, SceneTestCase) and v is not SceneTestCase and hasattr(v, "CASES")
        ]

        # Apply --runtime/--level filters (child mode sets both; parent may also
        # use them when the user wants a narrow run).
        if args.runtime is not None:
            test_classes = [c for c in test_classes if getattr(c, "_st_runtime", None) == args.runtime]
        if args.level is not None:
            test_classes = [c for c in test_classes if getattr(c, "_st_level", None) == args.level]
        if not test_classes:
            print(
                f"No matching classes (runtime={args.runtime}, level={args.level})",
                file=sys.stderr,
            )
            sys.exit(0)

        selectors = [_parse_case_selector(v) for v in (args.case or [])]
        try:
            selected = _select_cases(test_classes, args.platform, selectors, args.manual)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

        selected_by_cls: dict[type, list[dict]] = {}
        for cls, case in selected:
            selected_by_cls.setdefault(cls, []).append(case)

        # L3 profiling not supported yet (multi-chip-process filename collision).
        # Mirror the pytest-side guard so standalone users get the same early-fail.
        if args.enable_l2_swimlane:
            l3_classes = sorted(cls.__name__ for cls in selected_by_cls if cls._st_level == 3)
            if l3_classes:
                print(
                    f"ERROR: --enable-l2-swimlane is not supported for L3 tests yet — "
                    f"multi-chip-process filename collision unresolved. "
                    f"L3 classes selected: {', '.join(l3_classes)}. "
                    f"Either drop --enable-l2-swimlane or scope to L2 with --level 2.",
                    file=sys.stderr,
                )
                sys.exit(2)

        # Child mode: both --runtime and --level set. Run inline without
        # spawning further subprocesses; this is the path dispatcher
        # children take after we re-enter run_module.
        child_mode = args.runtime is not None and args.level is not None

        if not child_mode:
            has_multi_dev_case = any(
                int(case.get("config", {}).get("device_count", 1)) > 1
                for cases in selected_by_cls.values()
                for case in cases
            )
            has_multiple_groups = len({(cls._st_runtime, cls._st_level) for cls in selected_by_cls}) > 1
            needs_orchestration = len(device_ids) > 1 or has_multi_dev_case or has_multiple_groups
            if needs_orchestration:
                ok = _dispatch_test_phases_standalone(module_name, selected_by_cls, args)
                sys.exit(0 if ok else 1)

        # ----- Inline execution (single group or child mode) -----
        by_rt_level: dict[tuple[str, int], list[type]] = {}
        for cls in selected_by_cls:
            by_rt_level.setdefault((cls._st_runtime, cls._st_level), []).append(cls)

        ok = True
        for (runtime, level), group in by_rt_level.items():
            print(f"\n=== Runtime: {runtime}  Level: {level} ===")
            worker, per_class_sub_handles, per_class_chip_handles = _create_standalone_worker(
                group, level, args, selected_by_cls
            )
            try:
                for cls in group:
                    inst = cls()
                    callable_obj = inst.build_callable(args.platform)
                    sub_handles = per_class_sub_handles.get(cls, {})
                    chip_handles = per_class_chip_handles.get(cls, {})
                    # For L3: merge chip handles into callable_obj (replacing
                    # ChipCallable objects with their registered handle).
                    if level == 3 and chip_handles:
                        callable_obj = {**chip_handles}
                    for case in selected_by_cls[cls]:
                        label = f"{cls.__name__}::{case['name']}"
                        print(f"  {label} ... ", end="", flush=True)
                        try:
                            run_class_cases(
                                worker,
                                inst,
                                [case],
                                callable_obj=callable_obj,
                                sub_handles=sub_handles,
                                rounds=args.rounds,
                                skip_golden=args.skip_golden,
                                enable_l2_swimlane=args.enable_l2_swimlane,
                                enable_dump_args=args.dump_args,
                                enable_pmu=args.enable_pmu,
                                enable_dep_gen=args.enable_dep_gen,
                                enable_scope_stats=args.enable_scope_stats,
                                enable_device_log_timing=args.enable_device_log_timing,
                                enable_swimlane_overhead=args.enable_swimlane_overhead,
                            )
                            print("PASSED")
                        except Exception as e:  # noqa: BLE001
                            print(f"FAILED: {e}")
                            ok = False
                            if args.exitfirst:
                                raise SystemExit(1) from None
            finally:
                worker.close()

        sys.exit(0 if ok else 1)


def _dispatch_test_phases_standalone(module_name, selected_by_cls, args):  # noqa: PLR0912 -- L3 + L2 phases + chunking + fail-fast
    """Parent-mode test dispatcher for run_module.

    L3 phase: one subprocess per class, scheduled by device count.
    L2 phase: per-runtime fanout — up to max_parallel concurrent subprocesses,
    each owning one device and running its round-robin chunk of classes.

    Returns True on full success, False if any child failed.
    """
    from .parallel_scheduler import Job, format_device_range, run_jobs  # noqa: PLC0415

    module = sys.modules[module_name]
    # Path to the user's test script — sys.argv[0] is the script they invoked.
    script = os.path.abspath(getattr(module, "__file__", sys.argv[0]))

    common = ["-p", args.platform, "--manual", args.manual, "--log-level", args.log_level]
    if args.sanitizer != "none":
        common += ["--sanitizer", args.sanitizer]
    if args.rounds != 1:
        common += ["--rounds", str(args.rounds)]
    if args.skip_golden:
        common.append("--skip-golden")
    if args.enable_l2_swimlane:
        common += ["--enable-l2-swimlane", str(args.enable_l2_swimlane)]
    if args.dump_args:
        common += ["--dump-args", str(args.dump_args)]
    if args.enable_dep_gen:
        common.append("--enable-dep-gen")
    if args.enable_scope_stats:
        common.append("--enable-scope-stats")
    if args.enable_device_log_timing:
        common.append("--enable-device-log-timing")
    if args.enable_swimlane_overhead:
        common.append("--enable-swimlane-overhead")

    # ----- L3 phase: one subprocess per class (not per case).
    # The child's _create_standalone_worker allocates max(cls.CASES.device_count)
    # for the whole class, so the scheduler must grant the class-level max,
    # otherwise a class with a 4-device case can't run any of its 1-device
    # cases when we dispatch them individually with --device <1>. Cases inside
    # a class still run serially in the child, reusing the L3 Worker.
    l3_jobs = []
    for cls, cases in selected_by_cls.items():
        if cls._st_level != 3:
            continue
        if not cases:
            continue
        class_dev_count = max(int(c.get("config", {}).get("device_count", 1)) for c in cases)
        label = f"L3 {cls.__name__} (rt={cls._st_runtime}, dev={class_dev_count})"

        def _build(ids, _cls=cls.__name__, _rt=cls._st_runtime):
            return [
                sys.executable,
                script,
                *common,
                "-d",
                format_device_range(ids),
                "--case",
                f"{_cls}::",
                "--runtime",
                _rt,
                "--level",
                "3",
            ]

        # Per-case output_prefix is chosen inside the child by run_class_cases,
        # so no env var is needed to scope concurrent jobs.
        l3_jobs.append(Job(label=label, device_count=class_dev_count, build_cmd=_build))

    l3_failed = False
    if l3_jobs:
        print(
            f"\n{'=' * 60}\n  L3 phase: {len(l3_jobs)} case(s), pool={args.device_ids}, "
            f"max_parallel={args.max_parallel}\n{'=' * 60}\n"
        )

        def _on_done(res):
            tag = "PASSED" if res.returncode == 0 else f"FAILED (rc={res.returncode})"
            print(f"  {res.label}: {tag} on devices {res.device_ids}", flush=True)

        try:
            results = run_jobs(
                l3_jobs,
                args.device_ids,
                max_parallel=args.max_parallel,
                fail_fast=args.exitfirst,
                on_job_done=_on_done,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return False
        l3_failed = any(r.returncode != 0 for r in results)
        if l3_failed and args.exitfirst:
            return False

    # ----- L2 phase: runtimes serial (CANN isolation); within a runtime, fan
    # out classes across device_ids as one subprocess per device. Each child
    # owns one ChipWorker and runs its chunk of classes back-to-back (layer-4
    # reuse). Single-device case reduces to one subprocess per runtime.
    l2_by_runtime: dict[str, list[type]] = {}
    for cls in selected_by_cls:
        if cls._st_level == 2:
            l2_by_runtime.setdefault(cls._st_runtime, []).append(cls)

    l2_failed = False
    for rt in sorted(l2_by_runtime):
        classes = l2_by_runtime[rt]
        # Chunk count = min(-j, number of classes). We intentionally do NOT
        # include len(device_ids) here: each chunk uses 1 device and at most
        # max_parallel chunks run concurrently, so a pool bigger than -j just
        # leaves unused ids. Fewer, larger chunks also amortize ChipWorker
        # init (layer-4 reuse) over more cases.
        n = min(args.max_parallel, len(classes))
        if n == 0:
            continue
        # Round-robin distribute classes to N children.
        chunks: list[list[type]] = [[] for _ in range(n)]
        for i, cls in enumerate(classes):
            chunks[i % n].append(cls)

        header = f"  L2 Runtime: {rt}" + (f"  [fanout n={n}]" if n > 1 else "")
        print(f"\n{'=' * 60}\n{header}\n{'=' * 60}\n")

        l2_jobs = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            dev = args.device_ids[i]
            case_filters: list[str] = []
            for cls in chunk:
                # User-supplied selectors still filter; we scope to this chunk's
                # classes using "ClassName::<case>" or "ClassName::" (whole class).
                for sel in args.case or []:
                    if "::" in sel:
                        sel_cls, sel_case = sel.split("::", 1)
                        if not sel_cls or sel_cls == cls.__name__:
                            case_filters.append(f"{cls.__name__}::{sel_case}" if sel_case else f"{cls.__name__}::")
                    else:
                        # Bare selector "Foo" = case name in any class — forward as-is,
                        # but still scope to this chunk's classes.
                        case_filters.append(f"{cls.__name__}::{sel}")
                if not args.case:
                    case_filters.append(f"{cls.__name__}::")
            label = f"L2 {rt} dev={dev} ({len(chunk)} class(es))"

            def _build(ids, _rt=rt, _dev=dev, _filters=tuple(case_filters)):
                cmd = [
                    sys.executable,
                    script,
                    *common,
                    "-d",
                    str(_dev),
                    "--runtime",
                    _rt,
                    "--level",
                    "2",
                ]
                for f in _filters:
                    cmd += ["--case", f]
                return cmd

            # device_count=1 for L2 fanout children (each child uses one slot).
            # Per-case output_prefix is chosen inside the child by run_class_cases,
            # so no env var is needed to scope concurrent jobs.
            l2_jobs.append(Job(label=label, device_count=1, build_cmd=_build))

        # Use the same scheduler: pool=device_ids, fail_fast=exitfirst. This
        # gives us automatic parallelism + SIGTERM on fail-fast.
        def _on_l2_done(res):
            tag = "PASSED" if res.returncode == 0 else f"FAILED (rc={res.returncode})"
            print(f"  {res.label}: {tag}", flush=True)

        try:
            results = run_jobs(
                l2_jobs,
                args.device_ids,
                max_parallel=args.max_parallel,
                fail_fast=args.exitfirst,
                on_job_done=_on_l2_done,
            )
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            l2_failed = True
            if args.exitfirst:
                break
            continue

        if any(r.returncode != 0 for r in results):
            l2_failed = True
            if args.exitfirst:
                break

    return not (l3_failed or l2_failed)


def _create_standalone_worker(group, level, args, selected_by_cls):
    """Create a Worker for a (runtime, level) group in run_module.

    ``level`` is passed explicitly by the caller; do not read it from
    ``group[0]._st_level`` because groups are now keyed on (runtime, level)
    and mixed-level files are allowed.

    ``selected_by_cls`` is the dict of the cases that will actually run (after
    ``--case`` / ``--manual`` / platform filtering). L3 ``max_devices`` /
    ``max_sub_workers`` must be computed from these, not from ``cls.CASES``:
    otherwise a manual case with a larger ``device_count`` inflates the
    allocation even when it isn't scheduled.

    Returns ``(worker, per_class_sub_handles, per_class_chip_handles)`` for both
    L2 and L3 so the caller can unpack uniformly. L2 has neither sub
    callables nor pre-registered chip callables, so both dicts are empty.
    """
    first_cls = group[0]
    if level == 2:
        return first_cls._create_worker(args.platform, args.device), {}, {}

    from simpler.worker import Worker  # noqa: PLC0415

    max_devices = max(
        (c.get("config", {}).get("device_count", 1) for cls in group for c in selected_by_cls.get(cls, [])),
        default=1,
    )
    max_subs = max(
        (c.get("config", {}).get("num_sub_workers", 0) for cls in group for c in selected_by_cls.get(cls, [])),
        default=0,
    )
    # Prefer the allocated list (dispatcher child mode), fall back to
    # contiguous range starting at args.device (legacy inline path).
    allocated = getattr(args, "device_ids", None)
    if allocated and len(allocated) >= max_devices:
        device_ids = allocated[:max_devices]
    else:
        device_ids = list(range(args.device, args.device + max_devices))
    worker = Worker(
        level=3,
        device_ids=device_ids,
        num_sub_workers=max_subs,
        platform=args.platform,
        runtime=first_cls._st_runtime,
    )
    # Prepare sub callables per-class to avoid name collisions.
    per_class_sub_handles: dict[type, dict] = {}
    # Also prepare ChipCallables here (before init) so the chip children
    # pre-warm them via _CTRL_PREPARE.
    per_class_chip_handles: dict[type, dict] = {}
    for cls in group:
        cls_sub_handles = {}
        cls_chip_handles = {}
        for entry in cls.CALLABLE.get("callables", []):
            if "callable" in entry:
                handle = worker.register(entry["callable"])
                cls_sub_handles[entry["name"]] = handle
            elif "orchestration" in entry:
                name = entry["name"]
                cache_key = (cls.__qualname__, name, args.platform, cls._st_runtime)
                chip = _compile_chip_callable_from_spec(entry, args.platform, cls._st_runtime, cache_key)
                handle = worker.register(chip)
                cls_chip_handles[name] = handle
                cls_chip_handles[f"{name}_sig"] = entry["orchestration"].get("signature", [])
        per_class_sub_handles[cls] = cls_sub_handles
        per_class_chip_handles[cls] = cls_chip_handles
    worker.init()
    return worker, per_class_sub_handles, per_class_chip_handles
