"""Runtime-side assertions shared by the atomic minibench ST cases.

The minibench environment variables are diagnostic controls, not correctness
oracles by themselves. A runtime that ignores ``PTO_DIST_DEPSIG`` can
therefore pass the numerical golden while the feature under test is absent.
This mixin makes the diagnostics observable and fails the case when the
runtime does not report them.

The host simulator writes diagnostics to the captured process output.  On
hardware, AICore ``fprintf`` output is normally flushed to the per-device CANN
log, so the same contract also reads the bytes appended by this case.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass


_DEPSIG_RE = re.compile(r"\[dist\]\s+DEPSIG\s+mode=(\w+)\s+sig=([0-9a-fA-F]+)\s+edges=(\d+)")
_TMOPS_RE = re.compile(
    r"\[dist\]\s+TMOPS\s+mode=(\w+)\s+cores=(\d+)\s+"
    r"inserts=(\d+)\s+lookups=(\d+)\s+scans=(\d+)"
)


@contextmanager
def _capture_host_fds():
    """Capture C/C++ writes for the standalone SceneTestCase entry point.

    Pytest supplies ``capfd``. Direct-script runs do not, but the runtime
    diagnostics still need to be checked rather than silently skipped. This
    narrow fallback only surrounds one worker run.
    """

    captured = {"text": ""}
    with tempfile.TemporaryFile(mode="w+b") as output:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            yield captured
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            output.seek(0)
            captured["text"] = output.read().decode(errors="replace")


@dataclass(frozen=True)
class _DistRecord:
    case_name: str
    oracle_group: str
    platform: str
    text: str
    depsig: tuple[str, str, int] | None
    tmops: tuple[str, int, int, int, int] | None


class DistRuntimeContractMixin:
    """Add runtime diagnostic checks to a ``SceneTestCase``.

    The subclass may put a ``runtime_env`` mapping in a case. It is merged
    with the class-level ``RUNTIME_ENV`` for that case only. MB-7 uses this to
    sweep the run-ahead bound without relying on shell-side environment.
    """

    # MB-5 needs TMOPS on the host simulator.  TMOPS is intentionally a
    # simulator-only counter in the runtime, so hardware cases only require
    # DEPSIG.
    DIST_REQUIRE_TMOPS_ON_SIM = False
    DIST_DEFAULT_MODE = "private"
    DIST_LOG_POLL_SECONDS = 5.0

    def _resolve_env(self):
        # SceneTestCase performs the normal relative path handling for the
        # class-level mapping.  Minibench case overrides contain scalar knobs,
        # so updating that result keeps the framework behavior unchanged.
        env = super()._resolve_env()
        env.update({str(k): str(v) for k, v in getattr(self, "_dist_case_env", {}).items()})
        return env

    @staticmethod
    def _snapshot_device_logs(device_id: int) -> dict[str, int]:
        from simpler_setup.scene_test import _get_device_log_dir  # noqa: PLC0415

        log_dir = _get_device_log_dir(device_id)
        offsets: dict[str, int] = {}
        if not log_dir.is_dir():
            return offsets
        for path in log_dir.glob("*.log"):
            try:
                offsets[str(path)] = path.stat().st_size
            except FileNotFoundError:
                continue
        return offsets

    @staticmethod
    def _read_new_device_logs(device_id: int, offsets: dict[str, int]) -> str:
        from simpler_setup.scene_test import _get_device_log_dir  # noqa: PLC0415

        log_dir = _get_device_log_dir(device_id)
        if not log_dir.is_dir():
            return ""
        chunks: list[str] = []
        for path in sorted(log_dir.glob("*.log")):
            try:
                size = path.stat().st_size
                start = offsets.get(str(path), 0)
                if start > size:
                    # CANN rotated or truncated an existing file.
                    start = 0
                with path.open("rb") as f:
                    f.seek(start)
                    chunks.append(f.read().decode(errors="replace"))
            except (FileNotFoundError, OSError):
                continue
        return "".join(chunks)

    def _collect_case_text(self, platform: str, device_id: int, offsets: dict[str, int], host_text: str) -> str:
        if platform.endswith("sim"):
            return host_text

        # Device logs are asynchronous.  Poll only until a diagnostic marker
        # appears or the short watchdog expires; a missing marker should fail
        # this test, not be silently treated as a logging race.
        deadline = time.monotonic() + self.DIST_LOG_POLL_SECONDS
        device_text = ""
        while time.monotonic() < deadline:
            device_text = self._read_new_device_logs(device_id, offsets)
            if "[dist] DEPSIG" in device_text:
                break
            time.sleep(0.25)
        return host_text + device_text

    @staticmethod
    def _parse_record(case_name: str, oracle_group: str, platform: str, text: str) -> _DistRecord:
        deps = list(_DEPSIG_RE.finditer(text))
        tmops = list(_TMOPS_RE.finditer(text))
        depsig = None
        if deps:
            m = deps[-1]
            depsig = (m.group(1), m.group(2).lower(), int(m.group(3)))
        tmops_value = None
        if tmops:
            m = tmops[-1]
            tmops_value = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
        return _DistRecord(case_name, oracle_group, platform, text, depsig, tmops_value)

    def _run_and_validate(self, worker, callable_obj, case, *args, **kwargs):
        if not hasattr(self, "_dist_records"):
            self._dist_records = []

        platform = str(worker._config.get("platform", "<unknown>"))
        device_id = int(getattr(worker, "_st_device_id", worker._config.get("device_id", 0)))
        offsets = {} if platform.endswith("sim") else self._snapshot_device_logs(device_id)
        self._dist_case_env = dict(case.get("runtime_env", {}))
        run_error = False
        result = None
        host_text = ""
        capfd = getattr(self, "_dist_capfd", None)
        try:
            if capfd is None:
                with _capture_host_fds() as captured:
                    result = super()._run_and_validate(worker, callable_obj, case, *args, **kwargs)
                host_text = captured["text"]
            else:
                result = super()._run_and_validate(worker, callable_obj, case, *args, **kwargs)
        except BaseException:
            run_error = True
            raise
        finally:
            if capfd is not None:
                captured = capfd.readouterr()
                host_text = captured.out + captured.err
            text = self._collect_case_text(platform, device_id, offsets, host_text)
            record = self._parse_record(case["name"], case.get("oracle_group", case["name"]), platform, text)
            self._dist_records.append(record)
            self._dist_case_env = {}

            if not run_error:
                expected_mode = self.DIST_DEFAULT_MODE
                if record.depsig is None:
                    raise AssertionError(
                        f"{type(self).__name__}/{case['name']} did not emit the runtime DEPSIG oracle; "
                        f"PTO_DIST_DEPSIG is being ignored or diagnostics were not captured.\n"
                        f"Captured output tail:\n{text[-4000:]}"
                    )
                if record.depsig[0] != expected_mode:
                    raise AssertionError(
                        f"{type(self).__name__}/{case['name']} requested TensorMap mode {expected_mode!r} "
                        f"but runtime reported {record.depsig[0]!r}: {text[-2000:]}"
                    )
                if record.tmops is not None and record.tmops[0] != record.depsig[0]:
                    raise AssertionError(
                        f"{type(self).__name__}/{case['name']} reported inconsistent diagnostic modes: "
                        f"DEPSIG={record.depsig[0]!r}, TMOPS={record.tmops[0]!r}"
                    )
                if platform.endswith("sim") and self.DIST_REQUIRE_TMOPS_ON_SIM and record.tmops is None:
                    raise AssertionError(
                        f"{type(self).__name__}/{case['name']} did not emit TMOPS on the simulator; "
                        f"PTO_DIST_OVERHEAD is being ignored.\nCaptured output tail:\n{text[-4000:]}"
                    )
        if capfd is None and host_text:
            print(host_text, end="", flush=True)
        return result

    def _assert_dist_records(self) -> None:
        if not self._dist_records:
            raise AssertionError("minibench runtime contract collected no case records")

        # Scheduling/throttling may change the winner, but not the dependency
        # graph. Group by the reported artifact mode so repeated private
        # run-ahead settings remain comparable.
        by_group_mode: dict[str, dict[str, set[tuple[str, int]]]] = {}
        for record in self._dist_records:
            assert record.depsig is not None
            mode, sig, edges = record.depsig
            by_group_mode.setdefault(record.oracle_group, {}).setdefault(mode, set()).add((sig, edges))
        inconsistent = {
            group: {mode: values for mode, values in modes.items() if len(values) != 1}
            for group, modes in by_group_mode.items()
            if any(len(values) != 1 for values in modes.values())
        }
        if inconsistent:
            raise AssertionError(f"DEPSIG changed across identical minibench graphs: {inconsistent}")
        # If a private minibench requests TMOPS, require positive work. Shared
        # PA uses its dedicated schema-v5 tests and is not compared to this
        # legacy private counter model.
        for record in self._dist_records:
            if record.tmops is None:
                continue
            _mode, _cores, inserts, lookups, scans = record.tmops
            if inserts <= 0 or lookups <= 0 or scans <= 0:
                raise AssertionError(f"non-positive TensorMap counters in {record.case_name}: {record.tmops}")

    def test_run(self, st_platform, st_worker, request, capfd):
        self._dist_records = []
        self._dist_capfd = capfd
        try:
            super().test_run(st_platform, st_worker, request)
            self._assert_dist_records()
        finally:
            self._dist_capfd = None
