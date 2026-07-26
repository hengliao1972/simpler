# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for RuntimeBuilder class."""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# --- Discovery tests (no compilation needed) ---


class TestRuntimeBuilderDiscovery:
    """Test runtime discovery from src/runtime/ subdirectories."""

    def test_discovers_real_runtimes(self, default_test_platform):
        """RuntimeBuilder discovers host_build_graph from the real project tree."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        runtimes = builder.list_runtimes()
        assert "host_build_graph" in runtimes

    def test_runtime_dir_resolves_to_project_root(self, default_test_platform, test_arch):
        """runtime_dir resolves to src/{arch}/runtime/ under the project root."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.runtime_dir == builder.runtime_root / "src" / test_arch / "runtime"
        assert builder.runtime_dir.is_dir()

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_discovers_configs_in_runtime_dir(
        self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch
    ):
        """RuntimeBuilder discovers implementations in the runtime directory."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        # Set up fake runtime tree with architecture-specific structure
        rt_dir = tmp_path / "src" / test_arch / "runtime" / "my_runtime"
        rt_dir.mkdir(parents=True)
        (rt_dir / "build_config.py").write_text("BUILD_CONFIG = {}\n")

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["my_runtime"]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_ignores_dirs_without_build_config(
        self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch
    ):
        """Directories without build_config.py are not listed."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        rt_dir = tmp_path / "src" / test_arch / "runtime"
        (rt_dir / "has_config").mkdir(parents=True)
        (rt_dir / "has_config" / "build_config.py").write_text("BUILD_CONFIG = {}\n")
        (rt_dir / "no_config").mkdir(parents=True)
        # __pycache__ should also be ignored
        (rt_dir / "__pycache__").mkdir(parents=True)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["has_config"]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_empty_runtime_dir(self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """Empty src/{arch}/runtime/ directory yields no runtimes."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        (tmp_path / "src" / test_arch / "runtime").mkdir(parents=True)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == []

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_missing_runtime_dir(self, MockCompiler, tmp_path, monkeypatch, default_test_platform):
        """Non-existent src/{arch}/runtime/ directory yields no runtimes."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == []

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_multiple_runtimes_sorted(self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """Multiple implementations are returned in sorted order."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        rt_dir = tmp_path / "src" / test_arch / "runtime"
        for name in ["zeta", "alpha", "beta"]:
            d = rt_dir / name
            d.mkdir(parents=True)
            (d / "build_config.py").write_text("BUILD_CONFIG = {}\n")

        builder = RuntimeBuilder(platform=default_test_platform)
        assert builder.list_runtimes() == ["alpha", "beta", "zeta"]


# --- Error handling tests ---


class TestRuntimeBuilderErrors:
    """Test get_binaries() error handling without invoking real compilation."""

    def test_unknown_runtime_raises(self, default_test_platform):
        """get_binaries() raises ValueError for a non-existent runtime name."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match="is not available for platform"):
            builder.get_binaries("nonexistent_runtime", build=True)

    def test_unknown_runtime_lists_available(self, default_test_platform):
        """ValueError message includes available runtime names."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match="host_build_graph"):
            builder.get_binaries("nonexistent_runtime", build=True)

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_empty_registry_shows_none(self, MockCompiler, tmp_path, monkeypatch, default_test_platform, test_arch):
        """ValueError message shows '(none)' when no runtimes exist."""
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

        (tmp_path / "src" / test_arch / "runtime").mkdir(parents=True)
        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(ValueError, match=r"\(none\)"):
            builder.get_binaries("anything", build=True)


class TestFdwicTensorMapMode:
    """FDWIC TensorMap mode is an explicit, narrowly scoped build identity."""

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_defaults_to_private_and_reads_environment(self, MockCompiler, monkeypatch):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.delenv("PTO_FDWIC_TENSORMAP_MODE", raising=False)
        assert RuntimeBuilder(platform="a5").fdwic_tensormap_mode == "private"

        monkeypatch.setenv("PTO_FDWIC_TENSORMAP_MODE", "shared")
        assert RuntimeBuilder(platform="a5sim").fdwic_tensormap_mode == "shared"
        assert RuntimeBuilder(platform="a5", fdwic_tensormap_mode="private").fdwic_tensormap_mode == "private"

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    @pytest.mark.parametrize("mode", ["", "unknown", "private/shared"])
    def test_rejects_invalid_mode(self, MockCompiler, mode):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        with pytest.raises(ValueError, match="Invalid FDWIC TensorMap mode"):
            RuntimeBuilder(platform="a5", fdwic_tensormap_mode=mode)

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_rejects_shared_on_other_architecture(self, MockCompiler):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        with pytest.raises(ValueError, match="only valid for the a5/a5sim"):
            RuntimeBuilder(platform="a2a3sim", fdwic_tensormap_mode="shared")

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_rejects_shared_for_other_a5_runtime(self, MockCompiler):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform="a5", fdwic_tensormap_mode="shared")
        with pytest.raises(ValueError, match="runtime='host_build_graph'"):
            builder.get_binaries("host_build_graph")


class TestRuntimeBuilderPtoIsaValidation:
    """Test PTO-ISA compatibility validation is scoped to affected runtimes."""

    @pytest.mark.parametrize(
        ("platform", "should_validate"),
        [
            ("a2a3", True),
            ("a2a3sim", False),
            ("a5", False),
            ("a5sim", False),
        ],
    )
    def test_pto_isa_validation_only_runs_for_a2a3_onboard(self, tmp_path, monkeypatch, platform, should_validate):
        from simpler_setup import pto_isa  # noqa: PLC0415
        from simpler_setup.platform_info import TARGETS, parse_platform  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        class _Target:
            def __init__(self, name):
                self._name = name

            def get_binary_name(self):
                return f"lib{self._name}.so"

        calls = []

        def _fail_if_called(lib_dir):
            calls.append(lib_dir)
            raise RuntimeError("pto-isa validation called")

        monkeypatch.setattr(pto_isa, "validate_runtime_pto_isa_compatible", _fail_if_called)

        builder = RuntimeBuilder.__new__(RuntimeBuilder)
        builder.platform = platform
        builder._arch, builder._variant = parse_platform(platform)
        builder._LIB_DIR = tmp_path / "lib"
        builder._runtime_compiler = type(
            "Compiler",
            (),
            {f"{target}_target": _Target(target) for target in TARGETS},
        )()

        if should_validate:
            with pytest.raises(RuntimeError, match="pto-isa validation called"):
                builder._lookup_binaries("test_rt", tmp_path / "out")
            assert calls == [builder._LIB_DIR]
        else:
            with pytest.raises(FileNotFoundError, match="Pre-built runtime binaries not found"):
                builder._lookup_binaries("test_rt", tmp_path / "out")
            assert calls == []


# --- Build integration tests (mocked compilation) ---


class TestRuntimeBuilderGetBinaries:
    """Test get_binaries(build=True) logic with mocked RuntimeCompiler."""

    @pytest.fixture(autouse=True)
    def _patch_runtime_root(self, monkeypatch, tmp_path):
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)

    def _make_runtime(self, tmp_path, test_arch, name="test_rt"):
        """Create a fake runtime with a valid build_config.py."""
        rt_dir = tmp_path / "src" / test_arch / "runtime" / name
        for sub in ["aicore", "aicpu", "host", "runtime"]:
            (rt_dir / sub).mkdir(parents=True)

        config_content = textwrap.dedent("""\
            BUILD_CONFIG = {
                "aicore": {
                    "include_dirs": ["aicore", "runtime"],
                    "source_dirs": ["runtime"]
                },
                "aicpu": {
                    "include_dirs": ["runtime"],
                    "source_dirs": ["aicpu", "runtime"]
                },
                "host": {
                    "include_dirs": ["runtime"],
                    "source_dirs": ["host", "runtime"]
                }
            }
        """)
        (rt_dir / "build_config.py").write_text(config_content)
        return rt_dir

    @staticmethod
    def _set_build_roots(monkeypatch, tmp_path, RuntimeBuilder):
        monkeypatch.setattr(RuntimeBuilder, "_CACHE_DIR", tmp_path / "build" / "cache")
        monkeypatch.setattr(RuntimeBuilder, "_LIB_DIR", tmp_path / "build" / "lib")

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_returns_runtime_binaries(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """get_binaries(build=True) returns RuntimeBinaries with three paths."""
        from simpler_setup.runtime_builder import RuntimeBinaries, RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)

        # compile() returns Path when output_dir is set
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")

        builder = RuntimeBuilder(platform=default_test_platform)
        result = builder.get_binaries("test_rt", build=True)

        assert isinstance(result, RuntimeBinaries)
        assert result.host_path.name == "libhost.so"
        assert result.aicpu_path.name == "libaicpu.so"
        assert result.aicore_path.name == "libaicore.so"

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_calls_compiler_three_times(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """get_binaries(build=True) invokes compiler.compile() exactly 3 times."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")

        builder = RuntimeBuilder(platform=default_test_platform)
        builder.get_binaries("test_rt", build=True)

        assert mock_instance.compile.call_count == 3
        targets = sorted(call.args[0] for call in mock_instance.compile.call_args_list)
        assert targets == ["aicore", "aicpu", "host"]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_resolves_paths_relative_to_config(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """Include/source dirs are resolved relative to the build_config.py directory."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        rt_dir = self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: (Path(kw["output_dir"]) / f"lib{target}.so")

        builder = RuntimeBuilder(platform=default_test_platform)
        builder.get_binaries("test_rt", build=True)

        # Check any call: include_dirs should be resolved absolute paths
        for call in mock_instance.compile.call_args_list:
            include_dirs = call.args[1]
            for d in include_dirs:
                assert Path(d).is_absolute()
                assert str(rt_dir.resolve()) in d

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_propagates_compiler_error(self, MockCompiler, tmp_path, default_test_platform, test_arch):
        """If RuntimeCompiler.compile() raises, get_binaries() propagates the exception."""
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, test_arch)

        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = RuntimeError("cmake failed")

        builder = RuntimeBuilder(platform=default_test_platform)
        with pytest.raises(RuntimeError, match="cmake failed"):
            builder.get_binaries("test_rt", build=True)

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_fdwic_mode_macro_reaches_all_three_targets(self, MockCompiler, tmp_path, monkeypatch):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        runtime = "fully_distributed_within_core"
        self._make_runtime(tmp_path, "a5", runtime)
        self._set_build_roots(monkeypatch, tmp_path, RuntimeBuilder)
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: Path(kw["output_dir"]) / f"lib{target}.so"
        mock_instance.compile_simpler_log.return_value = tmp_path / "build" / "lib" / "libsimpler_log.so"

        builder = RuntimeBuilder(platform="a5", fdwic_tensormap_mode="shared")
        builder.get_binaries(runtime, build=True)

        runtime_calls = [
            call for call in mock_instance.compile.call_args_list if call.args[0] in {"host", "aicpu", "aicore"}
        ]
        assert len(runtime_calls) == 3
        assert {tuple(call.kwargs["compile_definitions"]) for call in runtime_calls} == {("PTO_FDWIC_SHARED_MAP=1",)}

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_private_mode_does_not_change_other_runtime_artifacts(self, MockCompiler, tmp_path, monkeypatch):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        self._make_runtime(tmp_path, "a5", "test_rt")
        self._set_build_roots(monkeypatch, tmp_path, RuntimeBuilder)
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: Path(kw["output_dir"]) / f"lib{target}.so"
        mock_instance.compile_simpler_log.return_value = tmp_path / "build" / "lib" / "libsimpler_log.so"

        RuntimeBuilder(platform="a5", fdwic_tensormap_mode="private").get_binaries("test_rt", build=True)

        runtime_calls = [
            call for call in mock_instance.compile.call_args_list if call.args[0] in {"host", "aicpu", "aicore"}
        ]
        assert {Path(call.kwargs["output_dir"]) for call in runtime_calls} == {
            tmp_path / "build" / "lib" / "a5" / "onboard" / "test_rt"
        }
        assert {Path(call.kwargs["build_dir"]) for call in runtime_calls} == {
            tmp_path / "build" / "cache" / "a5" / "onboard" / "test_rt"
        }
        assert all("compile_definitions" not in call.kwargs for call in runtime_calls)

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_fdwic_private_and_shared_use_separate_output_and_cache_paths(self, MockCompiler, tmp_path, monkeypatch):
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        runtime = "fully_distributed_within_core"
        self._make_runtime(tmp_path, "a5", runtime)
        self._set_build_roots(monkeypatch, tmp_path, RuntimeBuilder)
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: Path(kw["output_dir"]) / f"lib{target}.so"
        mock_instance.compile_simpler_log.return_value = tmp_path / "build" / "lib" / "libsimpler_log.so"

        for mode in ("private", "shared"):
            RuntimeBuilder(platform="a5", fdwic_tensormap_mode=mode).get_binaries(runtime, build=True)

        private_calls = [
            call for call in mock_instance.compile.call_args_list if Path(call.kwargs["output_dir"]).name == "private"
        ]
        shared_calls = [
            call for call in mock_instance.compile.call_args_list if Path(call.kwargs["output_dir"]).name == "shared"
        ]
        assert len(private_calls) == len(shared_calls) == 3
        assert {Path(call.kwargs["build_dir"]) for call in private_calls} == {
            tmp_path / "build" / "cache" / "a5" / "onboard" / runtime / "private"
        }
        assert {Path(call.kwargs["build_dir"]) for call in shared_calls} == {
            tmp_path / "build" / "cache" / "a5" / "onboard" / runtime / "shared"
        }
        assert {tuple(call.kwargs["compile_definitions"]) for call in private_calls} == {("PTO_FDWIC_SHARED_MAP=0",)}
        assert {tuple(call.kwargs["compile_definitions"]) for call in shared_calls} == {("PTO_FDWIC_SHARED_MAP=1",)}

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_fdwic_baseline_source_state_tracks_contents_and_mode(self, MockCompiler, tmp_path, monkeypatch):
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        runtime = "fully_distributed_within_core"
        rt_dir = self._make_runtime(tmp_path, "a5", runtime)
        source = rt_dir / "runtime" / "state.h"
        source.write_text("inline constexpr int kState = 1;\n")
        self._set_build_roots(monkeypatch, tmp_path, RuntimeBuilder)
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: Path(kw["output_dir"]) / f"lib{target}.so"
        mock_instance.compile_simpler_log.return_value = tmp_path / "build" / "lib" / "libsimpler_log.so"
        observed_states = []
        monkeypatch.setattr(
            rb_module,
            "_invalidate_cache_if_stale",
            lambda _cache_dir, current_state: observed_states.append(current_state),
        )

        RuntimeBuilder(platform="a5", fdwic_tensormap_mode="private").get_binaries(runtime, build=True)
        first_private_states = set(observed_states)
        observed_states.clear()
        source.write_text("inline constexpr int kState = 2;\n")
        RuntimeBuilder(platform="a5", fdwic_tensormap_mode="private").get_binaries(runtime, build=True)
        second_private_states = set(observed_states)
        observed_states.clear()
        RuntimeBuilder(platform="a5", fdwic_tensormap_mode="shared").get_binaries(runtime, build=True)
        shared_states = set(observed_states)

        assert all(state.startswith("source-v2:") for state in first_private_states)
        assert first_private_states != second_private_states
        assert second_private_states != shared_states


class TestFdwicAicoreExtraBuild:
    """Per-callable AICore images share the baseline FDWIC build identity."""

    @staticmethod
    def _make_runtime(tmp_path):
        runtime = "fully_distributed_within_core"
        rt_dir = tmp_path / "src" / "a5" / "runtime" / runtime
        for subdir in ("aicore", "runtime"):
            (rt_dir / subdir).mkdir(parents=True)
        (rt_dir / "build_config.py").write_text(
            textwrap.dedent("""\
                BUILD_CONFIG = {
                    "aicore": {
                        "include_dirs": ["aicore", "runtime"],
                        "source_dirs": ["runtime"]
                    }
                }
            """)
        )
        return runtime

    @patch("simpler_setup.kernel_compiler.KernelCompiler")
    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_mode_path_and_effective_definitions(self, MockCompiler, MockKernelCompiler, tmp_path, monkeypatch):
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(RuntimeBuilder, "_CACHE_DIR", tmp_path / "build" / "cache")
        monkeypatch.setattr(RuntimeBuilder, "_LIB_DIR", tmp_path / "build" / "lib")
        runtime = self._make_runtime(tmp_path)
        MockKernelCompiler.return_value.get_incore_include_dirs.return_value = []
        mock_instance = MockCompiler.get_instance.return_value
        mock_instance.compile.side_effect = lambda target, *a, **kw: Path(kw["output_dir"]) / "aicore.o"
        extra_source = tmp_path / "callable.cpp"
        extra_source.write_text("void callable() {}\n")

        result = RuntimeBuilder(platform="a5", fdwic_tensormap_mode="shared").build_aicore_with_extra_sources(
            runtime,
            [extra_source],
            "callable-key",
            compile_definitions=["PTO_FDWIC_PERF_CLOCK=1", "PTO_FDWIC_TRACE_ENABLED=0"],
        )

        call = mock_instance.compile.call_args
        expected_root = tmp_path / "build" / "lib" / "a5" / "onboard" / runtime / "shared"
        assert result == expected_root / "aicore-extra" / "callable-key" / "aicore.o"
        assert Path(call.kwargs["build_dir"]) == (
            tmp_path / "build" / "cache" / "a5" / "onboard" / runtime / "shared" / "aicore-extra" / "callable-key"
        )
        assert call.kwargs["compile_definitions"] == [
            "PTO_FDWIC_SHARED_MAP=1",
            "PTO_FDWIC_PERF_CLOCK=1",
            "PTO_FDWIC_TRACE_ENABLED=0",
        ]

    @patch("simpler_setup.runtime_builder.RuntimeCompiler")
    def test_rejects_conflicting_mode_definition(self, MockCompiler, tmp_path, monkeypatch):
        import simpler_setup.runtime_builder as rb_module  # noqa: PLC0415
        from simpler_setup.runtime_builder import RuntimeBuilder  # noqa: PLC0415

        monkeypatch.setattr(rb_module, "PROJECT_ROOT", tmp_path)
        runtime = self._make_runtime(tmp_path)
        builder = RuntimeBuilder(platform="a5", fdwic_tensormap_mode="shared")

        with pytest.raises(ValueError, match="Conflicting compile definitions for PTO_FDWIC_SHARED_MAP"):
            builder.build_aicore_with_extra_sources(
                runtime,
                [],
                "conflict",
                compile_definitions=["PTO_FDWIC_SHARED_MAP=0"],
            )


# --- _invalidate_cache_if_stale unit tests ---


class TestInvalidateCacheIfStale:
    """Test cmake cache invalidation behavior under varying git state."""

    def test_clears_when_commit_mismatches(self, tmp_path):
        from simpler_setup.runtime_builder import _GIT_COMMIT_FILE, _invalidate_cache_if_stale  # noqa: PLC0415

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / _GIT_COMMIT_FILE).write_text("old_sha\n")
        (cache_dir / "stale_artifact.o").write_text("stale")

        _invalidate_cache_if_stale(cache_dir, "new_sha")

        assert not (cache_dir / "stale_artifact.o").exists()
        assert (cache_dir / _GIT_COMMIT_FILE).read_text().strip() == "new_sha"

    def test_keeps_when_commit_matches(self, tmp_path):
        from simpler_setup.runtime_builder import _GIT_COMMIT_FILE, _invalidate_cache_if_stale  # noqa: PLC0415

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / _GIT_COMMIT_FILE).write_text("same_sha\n")
        artifact = cache_dir / "kept_artifact.o"
        artifact.write_text("fresh")

        _invalidate_cache_if_stale(cache_dir, "same_sha")

        assert artifact.exists()

    def test_clears_when_commit_unavailable(self, tmp_path):
        """No discoverable git HEAD should force a clean rebuild, not silently skip."""
        from simpler_setup.runtime_builder import _GIT_COMMIT_FILE, _invalidate_cache_if_stale  # noqa: PLC0415

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / _GIT_COMMIT_FILE).write_text("old_sha\n")
        (cache_dir / "stale_artifact.o").write_text("stale")

        _invalidate_cache_if_stale(cache_dir, "")

        assert not (cache_dir / "stale_artifact.o").exists()
        assert not (cache_dir / _GIT_COMMIT_FILE).exists()
        assert cache_dir.is_dir()


# --- Per-callable compile-definition tests ---


class TestBuildTargetCompileDefinitions:
    """Per-callable CCEC profiles must reach CMake as one stable list."""

    class _Toolchain:
        is_host = False

        @staticmethod
        def get_cmake_args():
            return []

    def test_forwards_compile_definitions_to_cmake(self, tmp_path):
        from simpler_setup.runtime_compiler import BuildTarget  # noqa: PLC0415

        target = BuildTarget(self._Toolchain(), str(tmp_path), "aicore_kernel.o")
        args = target.gen_cmake_args(
            [str(tmp_path / "include")],
            [str(tmp_path / "src")],
            compile_definitions=["PTO_FDWIC_PERF_CLOCK=1", "PTO_FDWIC_TRACE_ENABLED=0"],
        )

        assert ("-DCUSTOM_COMPILE_DEFINITIONS=PTO_FDWIC_PERF_CLOCK=1;PTO_FDWIC_TRACE_ENABLED=0") in args

    def test_omits_empty_compile_definitions(self, tmp_path):
        from simpler_setup.runtime_compiler import BuildTarget  # noqa: PLC0415

        target = BuildTarget(self._Toolchain(), str(tmp_path), "aicore_kernel.o")
        args = target.gen_cmake_args([str(tmp_path / "include")], [str(tmp_path / "src")])

        assert not any(arg.startswith("-DCUSTOM_COMPILE_DEFINITIONS=") for arg in args)


# --- Full integration tests (real compilation) ---


@pytest.mark.requires_hardware
class TestRuntimeBuilderIntegration:
    """Integration tests that actually compile all platform x runtime combinations.

    Test parametrization is handled dynamically by conftest.py based on:
    - Available platforms discovered from src/*/platform/{onboard,sim}/
    - Available runtimes per architecture from src/{arch}/runtime/*/build_config.py
    - --platform filter if specified on command line
    """

    @pytest.fixture(autouse=True)
    def _reset_compiler_singleton(self):
        """Reset RuntimeCompiler singleton-per-platform cache so each test gets fresh instances."""
        from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

        yield
        RuntimeCompiler._instances.clear()

    def test_get_binaries_returns_valid_paths(self, platform, runtime_name):
        """get_binaries(build=True) produces RuntimeBinaries with existing files."""
        from simpler_setup.runtime_builder import RuntimeBinaries, RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=platform)
        result = builder.get_binaries(runtime_name, build=True)

        assert isinstance(result, RuntimeBinaries)
        for label, path in [
            ("host", result.host_path),
            ("aicpu", result.aicpu_path),
            ("aicore", result.aicore_path),
        ]:
            assert path.is_file(), f"{label} binary not found: {path}"
            assert path.stat().st_size > 0, f"{label} binary is empty: {path}"
