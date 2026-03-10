import importlib.util
import unittest
from pathlib import Path

import coremltools as ct
import numpy as np


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "benchmark_compute_units.py"
    spec = importlib.util.spec_from_file_location("benchmark_compute_units", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkComputeUnitHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_benchmark_module()

    def test_parse_compute_unit_aliases(self) -> None:
        name, unit = self.module._parse_compute_unit("cpu_gpu")
        self.assertEqual(name, "cpu_and_gpu")
        self.assertEqual(unit, ct.ComputeUnit.CPU_AND_GPU)

        name, unit = self.module._parse_compute_unit("cpu_and_ne")
        self.assertEqual(name, "cpu_and_ne")
        self.assertEqual(unit, ct.ComputeUnit.CPU_AND_NE)

        with self.assertRaises(ValueError):
            self.module._parse_compute_unit("unsupported")

    def test_latency_stats_contains_expected_keys(self) -> None:
        stats = self.module._latency_stats([2.0, 1.0, 5.0, 3.0])
        self.assertEqual(set(stats.keys()), {"min_ms", "max_ms", "mean_ms", "p50_ms", "p95_ms"})
        self.assertEqual(stats["min_ms"], 1.0)
        self.assertEqual(stats["max_ms"], 5.0)
        self.assertGreater(stats["mean_ms"], 0.0)

    def test_compare_outputs_detects_mismatch(self) -> None:
        baseline = {"out": np.asarray([1.0, 2.0], dtype=np.float32)}
        candidate = {"out": np.asarray([1.0, 4.0], dtype=np.float32)}
        result = self.module._compare_outputs(baseline, candidate, atol=1e-6, rtol=1e-6)
        self.assertFalse(result["ok"])
        self.assertIn("out", result["outputs"])
        self.assertFalse(result["outputs"]["out"]["ok"])


if __name__ == "__main__":
    unittest.main()
