import os
import json
import subprocess
import sys
import unittest
from pathlib import Path


class EndToEndModelZooTransformerLiveTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MX2MIL_RUN_ZOO_TRANSFORMER") == "1",
        "Set MX2MIL_RUN_ZOO_TRANSFORMER=1 to run transformer_block live-capture zoo test.",
    )
    def test_model_zoo_transformer_block_live_capture(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        artifacts_dir = repo_root / "artifacts" / "test_zoo_transformer_live"
        cmd = [
            sys.executable,
            "scripts/run_model_zoo.py",
            "--models",
            "transformer_block",
            "--capture-mode",
            "live",
            "--eval-compiled",
            "--artifacts-dir",
            str(artifacts_dir),
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

        model_dir = artifacts_dir / "transformer_block"
        self.assertTrue((model_dir / "model.mlpackage").exists())
        self.assertTrue((model_dir / "model.mlmodelc").exists())
        self.assertTrue((model_dir / "capture_graph.dot").exists())
        self.assertTrue((model_dir / "report.json").exists())

        summary = json.loads((artifacts_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["model"], "transformer_block")
        self.assertEqual(summary[0]["status"], "ok")
        self.assertEqual(summary[0]["stage_status"]["capture"], "ok")
        self.assertEqual(summary[0]["stage_status"]["support_check"], "ok")
        self.assertEqual(summary[0]["stage_status"]["lower"], "ok")
        self.assertEqual(summary[0]["stage_status"]["convert"], "ok")
        self.assertEqual(summary[0]["stage_status"]["compile"], "ok")
        self.assertEqual(summary[0]["stage_status"]["eval"], "ok")


if __name__ == "__main__":
    unittest.main()
