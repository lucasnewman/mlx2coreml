import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mlx2coreml.conversion import CapturedMLXGraph
from mlx2coreml.ir import Graph, Node, TensorSpec
import mlx2coreml.convert_mlx_audio as convert_mlx_audio


class _DummyAudioModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            processor_config=SimpleNamespace(
                sampling_rate=16000,
                max_audio_seconds=8,
            )
        )

    def eval(self):
        return self

    def prepare_input_features(self, audio, sample_rate=None):
        return np.zeros((80, 800), dtype=np.float32)

    def __call__(self, input_features):
        return input_features


class ConvertMlxAudioTests(unittest.TestCase):
    def test_main_writes_reports_for_audio_model_when_skip_lower(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "smart_turn_coreml"
            graph = Graph(
                inputs=[TensorSpec("input_features", (80, 800), "fp32")],
                nodes=[Node("sigmoid", ("input_features",), "out")],
                outputs=["out"],
            )
            expected = {"out": np.zeros((80, 800), dtype=np.float32)}
            normalized_inputs = {"input_features": np.zeros((80, 800), dtype=np.float32)}

            argv = [
                "convert_mlx_audio.py",
                "--model-id",
                "mlx-community/smart-turn-v3",
                "--output",
                str(run_dir),
                "--audio-seconds",
                "1.0",
                "--skip-lower",
            ]

            with patch("mlx_audio.vad.load", return_value=_DummyAudioModel()):
                with patch.object(convert_mlx_audio, "capture_mlx_graph") as capture_mock:
                    capture_mock.return_value = CapturedMLXGraph(
                        graph=graph,
                        normalized_inputs=normalized_inputs,
                        expected_outputs=expected,
                    )
                    with patch.object(sys, "argv", argv):
                        convert_mlx_audio.main()

            self.assertTrue((run_dir / "graph.json").exists())
            self.assertTrue((run_dir / "inputs.npz").exists())
            self.assertTrue((run_dir / "expected_outputs.npz").exists())
            self.assertTrue((run_dir / "report.json").exists())
            self.assertTrue((run_dir / "run_context.json").exists())

            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["run_kind"], "mlx_audio_vad_weighted_probe")
            self.assertEqual(report["model_id"], "mlx-community/smart-turn-v3")
            self.assertEqual(report["audio_source"], "generated_silence")
            self.assertEqual(report["input_feature_shape"], [80, 800])
            self.assertEqual(report["input_feature_frames"], 800)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["stage_status"]["load_model"], "ok")
            self.assertEqual(report["stage_status"]["prepare_features"], "ok")
            self.assertEqual(report["stage_status"]["capture"], "ok")
            self.assertEqual(report["stage_status"]["normalize"], "ok")
            self.assertEqual(report["stage_status"]["type_infer"], "ok")
            self.assertEqual(report["stage_status"]["support_check"], "ok")
            self.assertEqual(report["stage_status"]["lower"], "skipped")
            self.assertEqual(report["stage_status"]["convert"], "skipped")
            self.assertEqual(report["main_function"], "main")
            self.assertIn("main", report["artifacts"]["functions"])


if __name__ == "__main__":
    unittest.main()
