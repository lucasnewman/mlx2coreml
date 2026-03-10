import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_analyze_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "analyze_ane_placement.py"
    spec = importlib.util.spec_from_file_location("analyze_ane_placement", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalyzeAnePlacementHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_analyze_module()

    def test_resolve_output_path_defaults_to_base_dir(self) -> None:
        base_dir = Path("/tmp/fake")
        resolved = self.module._resolve_output_path(None, base_dir=base_dir, filename="report.json")
        self.assertEqual(resolved, base_dir / "report.json")

    def test_write_markdown_emits_core_summary_fields(self) -> None:
        report = {
            "input_model_path": "/tmp/in.mlmodelc",
            "compiled_model_path": "/tmp/in.mlmodelc",
            "analysis": {
                "compute_units": "all",
                "total_operations": 10,
                "fallback_operation_count": 3,
                "fallback_ratio": 0.3,
                "preferred_device_counts": {"ANE": 7, "CPU": 3},
                "fallback_cost_ratio": 0.25,
                "top_ops": [["ios18.matmul", 5]],
                "top_fallback_ops": [["ios18.cast", 2]],
                "fallback_samples": [
                    {
                        "function": "main",
                        "operator": "ios18.cast",
                        "preferred_device": "CPU",
                        "supported_devices": ["ANE", "CPU"],
                        "outputs": ["x"],
                        "estimated_cost": 0.1,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            md_path = Path(temp_dir) / "placement.md"
            self.module._write_markdown(md_path, report)
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("ANE Placement Report", text)
            self.assertIn("Fallback operations", text)
            self.assertIn("Top fallback ops", text)


if __name__ == "__main__":
    unittest.main()
