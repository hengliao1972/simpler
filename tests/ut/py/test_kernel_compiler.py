# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from types import SimpleNamespace

import pytest

from simpler_setup.kernel_compiler import KernelCompiler


def _compiler_with_captured_command(monkeypatch):
    compiler = object.__new__(KernelCompiler)
    compiler._sanitizers = ""
    captured: dict[str, list[str]] = {}

    def capture(cmd, _output_path, _label, **_kwargs):
        captured["cmd"] = cmd
        return b"orchestration"

    monkeypatch.setattr(compiler, "_compile_to_bytes", capture)
    toolchain = SimpleNamespace(
        cxx_path="test-g++",
        is_host=True,
        get_compile_flags=lambda: ["-shared"],
    )
    return compiler, toolchain, captured


def test_orchestration_shared_lib_forwards_each_compile_definition(monkeypatch, tmp_path):
    source = tmp_path / "orch.cpp"
    source.write_text("void Orchestration() {}\n")
    compiler, toolchain, captured = _compiler_with_captured_command(monkeypatch)

    result = compiler._compile_orchestration_shared_lib(
        str(source),
        toolchain,
        compile_definitions=["PTO_FDWIC_SHARED_MAP=1", "PTO_FDWIC_TRACE_ENABLED=0"],
    )

    assert result == b"orchestration"
    assert "-DPTO_FDWIC_SHARED_MAP=1" in captured["cmd"]
    assert "-DPTO_FDWIC_TRACE_ENABLED=0" in captured["cmd"]


def test_orchestration_shared_lib_omits_definitions_by_default(monkeypatch, tmp_path):
    source = tmp_path / "orch.cpp"
    source.write_text("void Orchestration() {}\n")
    compiler, toolchain, captured = _compiler_with_captured_command(monkeypatch)

    compiler._compile_orchestration_shared_lib(str(source), toolchain)

    assert not any(argument.startswith("-D") for argument in captured["cmd"])


@pytest.mark.parametrize(
    "definition",
    [
        "",
        " ",
        "\0",
        "1NAME=1",
        "=1",
        "-DNAME=1",
    ],
)
def test_orchestration_shared_lib_rejects_invalid_compile_definition(monkeypatch, tmp_path, definition):
    source = tmp_path / "orch.cpp"
    source.write_text("void Orchestration() {}\n")
    compiler, toolchain, _captured = _compiler_with_captured_command(monkeypatch)

    with pytest.raises(ValueError, match="compile definition"):
        compiler._compile_orchestration_shared_lib(
            str(source),
            toolchain,
            compile_definitions=[definition],
        )


def test_compile_orchestration_forwards_compile_definitions(monkeypatch):
    compiler = object.__new__(KernelCompiler)
    compiler.platform = "a5sim"
    compiler.host_gxx = object()
    monkeypatch.setattr(compiler, "get_orchestration_include_dirs", lambda _runtime: [])
    monkeypatch.setattr(compiler, "_get_orchestration_config", lambda _runtime: ([], []))
    monkeypatch.setattr(
        compiler,
        "_get_toolchain",
        lambda _mapping: __import__("simpler_setup.toolchain", fromlist=["ToolchainType"]).ToolchainType.HOST_GXX,
    )
    captured = {}

    def capture(*_args, **kwargs):
        captured.update(kwargs)
        return b"orchestration"

    monkeypatch.setattr(compiler, "_compile_orchestration_shared_lib", capture)

    assert (
        compiler.compile_orchestration(
            "fully_distributed_within_core",
            "unused.cpp",
            compile_definitions=["PTO_FDWIC_SHARED_MAP=1"],
        )
        == b"orchestration"
    )
    assert captured["compile_definitions"] == ["PTO_FDWIC_SHARED_MAP=1"]
