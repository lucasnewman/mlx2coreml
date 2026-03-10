import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mlx2coreml.conversion import CapturedMLXGraph
from mlx2coreml.ir import Graph, Node, TensorSpec


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "convert_mlx_to_coreml.py"
    spec = importlib.util.spec_from_file_location("convert_mlx_to_coreml", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConvertMlxToCoreMlScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_script_module()

    def test_load_callable_resolves_dotted_attribute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            module_name = "tmp_convert_mlx_callable"
            module_path = temp_path / f"{module_name}.py"
            module_path.write_text(
                "\n".join(
                    [
                        "class Wrapper:",
                        "    def __call__(self, x):",
                        "        return x",
                        "",
                        "class Root:",
                        "    pass",
                        "",
                        "root = Root()",
                        "root.inner = Wrapper()",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            sys.path.insert(0, str(temp_path))
            importlib.invalidate_caches()
            try:
                fn = self.module._load_callable(module_name, "root.inner")
                self.assertTrue(callable(fn))
            finally:
                sys.path.remove(str(temp_path))
                sys.modules.pop(module_name, None)

    def test_main_writes_reports_for_generic_callable_when_skip_lower(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            module_name = "tmp_convert_mlx_identity"
            module_path = temp_path / f"{module_name}.py"
            module_path.write_text(
                "\n".join(
                    [
                        "def identity(x):",
                        "    return x",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            inputs_path = temp_path / "inputs.npz"
            np.savez(inputs_path, x=np.asarray([[1.0, 2.0]], dtype=np.float32))

            graph = Graph(
                inputs=[TensorSpec("x", (1, 2), "fp32")],
                nodes=[Node("add", ("x", "x"), "out")],
                outputs=["out"],
            )
            expected = {"out": np.asarray([[2.0, 4.0]], dtype=np.float32)}
            normalized_inputs = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
            artifacts_dir = temp_path / "artifacts"

            sys.path.insert(0, str(temp_path))
            importlib.invalidate_caches()
            try:
                argv = [
                    "convert_mlx_to_coreml.py",
                    "--module",
                    module_name,
                    "--function",
                    "identity",
                    "--inputs",
                    str(inputs_path),
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--run-name",
                    "generic_identity",
                    "--skip-lower",
                ]
                with patch.object(self.module, "capture_mlx_graph") as capture_mock:
                    capture_mock.return_value = CapturedMLXGraph(
                        graph=graph,
                        normalized_inputs=normalized_inputs,
                        expected_outputs=expected,
                    )
                    with patch.object(sys, "argv", argv):
                        self.module.main()

                run_dir = artifacts_dir / "generic_identity"
                self.assertTrue((run_dir / "graph.json").exists())
                self.assertTrue((run_dir / "inputs.npz").exists())
                self.assertTrue((run_dir / "expected_outputs.npz").exists())
                self.assertTrue((run_dir / "report.json").exists())
                self.assertTrue((run_dir / "run_context.json").exists())

                report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["run_kind"], "mlx_callable_convert")
                self.assertEqual(report["module"], module_name)
                self.assertEqual(report["function"], "identity")
                self.assertEqual(report["status"], "ok")
                self.assertEqual(report["stage_status"]["load_callable"], "ok")
                self.assertEqual(report["stage_status"]["load_inputs"], "ok")
                self.assertEqual(report["stage_status"]["capture"], "ok")
                self.assertEqual(report["stage_status"]["normalize"], "ok")
                self.assertEqual(report["stage_status"]["type_infer"], "ok")
                self.assertEqual(report["stage_status"]["support_check"], "ok")
                self.assertEqual(report["stage_status"]["lower"], "skipped")
                self.assertEqual(report["stage_status"]["convert"], "skipped")
                self.assertEqual(report["main_function"], "main")
                self.assertEqual(report["artifacts"]["source_inputs"], str(inputs_path.resolve()))
                self.assertIn("main", report["artifacts"]["functions"])
            finally:
                sys.path.remove(str(temp_path))
                sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
