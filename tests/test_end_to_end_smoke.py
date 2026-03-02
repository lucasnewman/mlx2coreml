import os
import json
import subprocess
import sys
import unittest
from pathlib import Path


class EndToEndSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MX2MIL_RUN_E2E") == "1",
        "Set MX2MIL_RUN_E2E=1 to run end-to-end smoke test.",
    )
    def test_smoke_pipeline_emits_model_and_compiled_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        artifacts_dir = repo_root / "artifacts" / "test_e2e"
        cmd = [
            sys.executable,
            "scripts/smoke_mlx_to_coreml.py",
            "--mock",
            "--eval-compiled",
            "--artifacts-dir",
            str(artifacts_dir),
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

        self.assertTrue((artifacts_dir / "smoke.mlpackage").exists())
        self.assertTrue((artifacts_dir / "smoke.mlmodelc").exists())
        self.assertTrue((artifacts_dir / "smoke_report.json").exists())
        self.assertTrue((artifacts_dir / "run_context.json").exists())

        report = json.loads((artifacts_dir / "smoke_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["run_kind"], "smoke")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["stage_status"]["normalize"], "ok")
        self.assertEqual(report["stage_status"]["type_infer"], "ok")
        self.assertEqual(report["stage_status"]["support_check"], "ok")
        self.assertEqual(report["stage_status"]["lower"], "ok")
        self.assertEqual(report["stage_status"]["convert"], "ok")


if __name__ == "__main__":
    unittest.main()
