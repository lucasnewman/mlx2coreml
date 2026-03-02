import os
import json
import subprocess
import sys
import unittest
from pathlib import Path


class EndToEndLiveCaptureSubsetTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MX2MIL_RUN_ZOO_LIVE_CAPTURE") == "1"
        or os.environ.get("MX2MIL_RUN_ZOO_LIVE_PARSE") == "1",
        "Set MX2MIL_RUN_ZOO_LIVE_CAPTURE=1 to run live-capture subset end-to-end test.",
    )
    def test_model_zoo_live_capture_subset(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        artifacts_dir = repo_root / "artifacts" / "test_zoo_live_capture"
        cmd = [
            sys.executable,
            "scripts/run_model_zoo.py",
            "--models",
            "linear_relu,arithmetic_chain,mlp_2layer",
            "--capture-mode",
            "live",
            "--skip-compile",
            "--artifacts-dir",
            str(artifacts_dir),
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

        summary_path = artifacts_dir / "summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(len(summary), 3)
        for item in summary:
            self.assertEqual(item["status"], "ok")
            self.assertEqual(item["stage_status"]["capture"], "ok")
            self.assertIn("capture_dot", item["artifacts"])
            self.assertTrue(item["artifacts"]["capture_dot"])


if __name__ == "__main__":
    unittest.main()
