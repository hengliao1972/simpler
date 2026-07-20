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
from pathlib import Path
from types import SimpleNamespace

import pytest
from _task_interface import ArgDirection, ChipCallable  # pyright: ignore[reportMissingImports]

# ``simpler_setup/__init__.py`` re-exports the ``scene_test`` *decorator*,
# which shadows the submodule attribute when accessed via ``simpler_setup``.
# Importing the names directly from the submodule avoids that ambiguity.
from simpler_setup.scene_test import (
    _aicore_override_cache,
    _assert_fdwic_perf_clock_elf,
    _compile_cache,
    _fdwic_profile,
    _profiled_cache_key,
    clear_compile_cache,
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
    """Normal and perf-clock builds must never reuse one AICore override."""
    base = ("Case", "a5", "fully_distributed_within_core")

    monkeypatch.delenv("PTO_FDWIC_PROFILE", raising=False)
    assert _fdwic_profile() == "none"
    assert _profiled_cache_key(base) == (*base, "none")

    monkeypatch.setenv("PTO_FDWIC_PROFILE", "perf-clock")
    assert _fdwic_profile() == "perf-clock"
    assert _profiled_cache_key(base) == (*base, "perf-clock")


def test_fdwic_profile_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("PTO_FDWIC_PROFILE", "typo")

    with pytest.raises(ValueError, match="Unsupported PTO_FDWIC_PROFILE"):
        _fdwic_profile()


def test_perf_clock_elf_gate_accepts_required_symbol_without_observers(monkeypatch, tmp_path):
    """The final image must keep its marker and remove observer slow paths."""
    symbol_table = (
        "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 "
        "_Z30dist_perf_clock_expect_submitsj\n"
    )
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
            "12: 0000000000000000 0 FUNC GLOBAL DEFAULT UND "
            "_Z30dist_perf_clock_expect_submitsj\n",
            "missing defined perf-clock marker",
        ),
        (
            "37410: 0000000000001b54 68 FUNC WEAK DEFAULT 1 "
            "_Z30dist_perf_clock_expect_submitsj\n"
            "42: 0000000000000048 8 OBJECT LOCAL DEFAULT 17 g_fdwic_atomic_record_index\n",
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
