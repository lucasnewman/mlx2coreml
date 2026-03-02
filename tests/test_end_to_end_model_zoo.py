import os
import json
import subprocess
import sys
import unittest
from pathlib import Path


class EndToEndModelZooTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MX2MIL_RUN_ZOO") == "1",
        "Set MX2MIL_RUN_ZOO=1 to run model-zoo end-to-end test.",
    )
    def test_model_zoo_runner(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        artifacts_dir = repo_root / "artifacts" / "test_zoo"
        cmd = [
            sys.executable,
            "scripts/run_model_zoo.py",
            "--models",
            "linear_relu,arithmetic_chain,reduction_suite,shape_helpers,indexing_transforms,creation_helpers,mlp_2layer,broadcast_tensordot,numeric_sanity,diagonal_trace,tri_band,logical_checks,meshgrid_kron,p0_math_pack,stats_divmod,conv_block",
            "--eval-compiled",
            "--artifacts-dir",
            str(artifacts_dir),
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

        for model_name in (
            "linear_relu",
            "arithmetic_chain",
            "reduction_suite",
            "shape_helpers",
            "indexing_transforms",
            "creation_helpers",
            "mlp_2layer",
            "broadcast_tensordot",
            "numeric_sanity",
            "diagonal_trace",
            "tri_band",
            "logical_checks",
            "meshgrid_kron",
            "p0_math_pack",
            "stats_divmod",
            "conv_block",
        ):
            model_dir = artifacts_dir / model_name
            self.assertTrue((model_dir / "model.mlpackage").exists())
            self.assertTrue((model_dir / "model.mlmodelc").exists())
            self.assertTrue((model_dir / "report.md").exists())
            self.assertTrue((model_dir / "report.json").exists())
        summary_path = artifacts_dir / "summary.json"
        run_context_path = artifacts_dir / "run_context.json"
        self.assertTrue(summary_path.exists())
        self.assertTrue(run_context_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
        self.assertEqual(run_context["run_kind"], "zoo")
        self.assertEqual(len(summary), 16)
        for item in summary:
            self.assertIn("stage_status", item)
            self.assertEqual(item["status"], "ok")
            self.assertIn("schema_version", item)
            self.assertEqual(item["stage_status"]["normalize"], "ok")
            self.assertEqual(item["stage_status"]["type_infer"], "ok")
            self.assertEqual(item["stage_status"]["support_check"], "ok")
            self.assertIn("artifacts", item)
            self.assertIn("capture_dot", item["artifacts"])


if __name__ == "__main__":
    unittest.main()
