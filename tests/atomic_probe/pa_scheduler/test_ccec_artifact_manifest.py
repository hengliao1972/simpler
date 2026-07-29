#!/usr/bin/env python3

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
RUN_SH = HERE / "run.sh"


class CcecArtifactManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_sh = self.root / "run.sh"
        shutil.copy2(RUN_SH, self.run_sh)
        self.run_sh.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish_artifacts(
        self, mode: str, variant: str, phase: str = "none"
    ) -> tuple[Path, list[str]]:
        if variant == "submit-pmu":
            build_dir = (
                self.root / "build" / "ccec" / mode /
                "submit-pmu" / phase
            )
            artifacts = [
                "pa_scheduler_host",
                "pa_scheduler_kernel.o",
                "libpa_scheduler_pmu_owner_aicpu.so",
                "libpa_scheduler_pmu_owner_dispatcher.so",
            ]
        else:
            build_dir = (
                self.root / "build" / "ccec" / mode / variant
            )
            artifacts = [
                "pa_scheduler_host",
                "pa_scheduler_kernel.o",
            ]
        # 同一测试方法会依次破坏多个字段；每个 subTest 都从这里重新
        # 发布完整基线，覆盖上一个 subTest 留下的 manifest。
        build_dir.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            path = build_dir / artifact
            if artifact == "pa_scheduler_host":
                path.write_text(
                    "#!/usr/bin/env bash\nexit 0\n",
                    encoding="utf-8",
                )
                path.chmod(0o755)
            else:
                path.write_bytes(
                    f"{mode}:{variant}:{phase}:{artifact}\n".encode()
                )

        if mode == "shared":
            submit_claim_bytes = 32
            records_per_core = 28416
            if variant == "swimlane":
                generic_bytes = 16
                worker_stride = 593920
            else:
                generic_bytes = 32
                worker_stride = 1048576
            mode_id = 1
        else:
            generic_bytes = 32
            submit_claim_bytes = 0
            records_per_core = 65536
            worker_stride = 2097152
            mode_id = 0

        phase_ids = {
            "none": 0,
            "claim": 1,
            "efdrain": 2,
            "materialize": 4,
            "register": 5,
        }
        lines = [
            "# schema=pa_scheduler_artifacts/v4",
            f"# tensormap_mode={mode}",
            f"# tensormap_mode_id={mode_id}",
            "# tensormap_ring_cap=128",
            "# shared_insert_turn_groups=1",
            f"# generic_record_bytes={generic_bytes}",
            f"# submit_claim_record_bytes={submit_claim_bytes}",
            f"# records_per_core={records_per_core}",
            f"# worker_stride_bytes={worker_stride}",
            f"# variant={variant}",
            f"# phase={phase}",
            f"# phase_id={phase_ids[phase]}",
        ]
        for artifact in artifacts:
            data = (build_dir / artifact).read_bytes()
            lines.append(
                f"{hashlib.sha256(data).hexdigest()}  {artifact}"
            )
        (build_dir / "pa_scheduler_artifacts.manifest").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        return build_dir, lines

    def _run(
        self, mode: str, variant: str, phase: str = "none"
    ) -> subprocess.CompletedProcess[str]:
        if variant == "swimlane":
            arguments = [
                str(self.run_sh), "run", "ccec",
                "--tensormap", mode,
            ]
        elif variant == "perf-clock":
            arguments = [
                str(self.run_sh), "perf-clock", "ccec",
                "--tensormap", mode,
            ]
        else:
            arguments = [
                str(self.run_sh), "submit-pmu", "ccec", phase,
                "--tensormap", mode,
            ]
        return subprocess.run(
            arguments,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_accepts_each_unique_layout(self) -> None:
        cases = [
            ("private", "swimlane", "none"),
            ("shared", "swimlane", "none"),
            ("shared", "perf-clock", "none"),
            ("shared", "submit-pmu", "register"),
        ]
        for mode, variant, phase in cases:
            with self.subTest(
                mode=mode, variant=variant, phase=phase
            ):
                self._publish_artifacts(mode, variant, phase)
                completed = self._run(mode, variant, phase)
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stdout + completed.stderr,
                )
                self.assertIn(
                    "CCEC artifact manifest verified",
                    completed.stdout,
                )

    def test_rejects_schema_and_each_layout_field_mutation(self) -> None:
        mutations = {
            "old schema": (0, "# schema=pa_scheduler_artifacts/v3"),
            "generic bytes": (5, "# generic_record_bytes=32"),
            "submit claim bytes": (
                6, "# submit_claim_record_bytes=16"
            ),
            "record count": (7, "# records_per_core=65536"),
            "worker stride": (8, "# worker_stride_bytes=1048576"),
        }
        for label, (index, replacement) in mutations.items():
            with self.subTest(label=label):
                build_dir, lines = self._publish_artifacts(
                    "shared", "swimlane"
                )
                lines[index] = replacement
                (
                    build_dir /
                    "pa_scheduler_artifacts.manifest"
                ).write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )
                completed = self._run("shared", "swimlane")
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "trace layout",
                    completed.stderr,
                )

    def test_rejects_reordered_or_extra_identity_lines(self) -> None:
        for label in ("reordered", "extra"):
            with self.subTest(label=label):
                build_dir, lines = self._publish_artifacts(
                    "shared", "swimlane"
                )
                if label == "reordered":
                    lines[5], lines[6] = lines[6], lines[5]
                else:
                    lines.insert(12, "# unexpected=1")
                (
                    build_dir /
                    "pa_scheduler_artifacts.manifest"
                ).write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )
                completed = self._run("shared", "swimlane")
                self.assertNotEqual(completed.returncode, 0)

    def test_rejects_artifact_checksum_mismatch(self) -> None:
        build_dir, _ = self._publish_artifacts(
            "shared", "swimlane"
        )
        with (build_dir / "pa_scheduler_kernel.o").open("ab") as output:
            output.write(b"tampered\n")
        completed = self._run("shared", "swimlane")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("SHA256", completed.stderr)


if __name__ == "__main__":
    unittest.main()
