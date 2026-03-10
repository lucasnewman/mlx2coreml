import importlib.util
import tempfile
import unittest
from pathlib import Path

import coremltools as ct
import numpy as np

from mlx2coreml.ir import Graph, Node, TensorSpec


def _load_convert_script_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "convert_mlx_lm_to_coreml.py"
    spec = importlib.util.spec_from_file_location("convert_mlx_lm_to_coreml", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlexInputShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_convert_script_module()

    def test_parse_flex_lengths_auto_includes_one_and_seq_len(self) -> None:
        parsed = self.module._parse_flex_lengths("auto", seq_len=16)
        self.assertEqual(parsed, [1, 16])

    def test_parse_flex_lengths_enforces_positive_unique_values(self) -> None:
        parsed = self.module._parse_flex_lengths("4,8,4", seq_len=8)
        self.assertEqual(parsed, [1, 4, 8])
        with self.assertRaises(ValueError):
            self.module._parse_flex_lengths("0,8", seq_len=8)

    def test_build_conversion_inputs_applies_enumerated_shapes_to_target_inputs(self) -> None:
        specs = [
            TensorSpec(name="input_ids", shape=(1, 16), dtype="int32"),
            TensorSpec(name="position_ids", shape=(1, 16), dtype="int32"),
            TensorSpec(name="hidden", shape=(1, 16, 64), dtype="fp16"),
        ]
        converted, applied = self.module._build_conversion_inputs(
            specs,
            flex_input_lens=[1, 16, 32],
            flex_input_names={"input_ids"},
        )
        assert converted is not None
        self.assertEqual(set(applied.keys()), {"input_ids"})
        self.assertEqual(applied["input_ids"], [[1, 16], [1, 1], [1, 32]])

        input_ids_tensor = converted[0]
        self.assertIsInstance(input_ids_tensor.shape, ct.EnumeratedShapes)
        actual_shapes = [
            tuple(int(v) for v in getattr(shape, "shape", ()))
            for shape in input_ids_tensor.shape.shapes
        ]
        self.assertEqual(actual_shapes, [(1, 16), (1, 1), (1, 32)])

        position_ids_tensor = converted[1]
        position_shape = tuple(int(v) for v in getattr(position_ids_tensor.shape, "shape", ()))
        self.assertEqual(position_shape, (1, 16))

        hidden_tensor = converted[2]
        hidden_shape = tuple(int(v) for v in getattr(hidden_tensor.shape, "shape", ()))
        self.assertEqual(hidden_shape, (1, 16, 64))

    def test_apply_argmax_output_reduction_index_and_value(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1, 2, 3), dtype="fp32")],
            nodes=[Node(op="identity", inputs=("x",), output="logits")],
            outputs=["logits"],
        )
        expected = {"logits": np.asarray([[[1.0, 4.0, 3.0], [2.0, 1.0, 5.0]]], dtype=np.float32)}
        reduced_graph, reduced_expected, meta = self.module._apply_argmax_output_reduction(
            graph,
            expected,
            mode="index_and_value",
            axis=-1,
        )
        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["mode"], "index_and_value")
        self.assertEqual(meta["axis"], 2)
        self.assertEqual([node.op for node in reduced_graph.nodes[-2:]], ["argmax", "max"])
        self.assertEqual(len(reduced_graph.outputs), 2)
        self.assertEqual(
            list(reduced_expected[reduced_graph.outputs[0]].tolist()),
            [[1, 2]],
        )
        self.assertEqual(
            list(reduced_expected[reduced_graph.outputs[1]].tolist()),
            [[4.0, 5.0]],
        )

    def test_apply_argmax_output_reduction_rejects_bad_axis(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1, 2, 3), dtype="fp32")],
            nodes=[Node(op="identity", inputs=("x",), output="logits")],
            outputs=["logits"],
        )
        expected = {"logits": np.zeros((1, 2, 3), dtype=np.float32)}
        with self.assertRaises(ValueError):
            self.module._apply_argmax_output_reduction(
                graph,
                expected,
                mode="index",
                axis=4,
            )

    def test_normalize_graph_for_function_uses_shared_pipeline(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1, 2, 3), dtype="fp32")],
            nodes=[Node(op="identity", inputs=("x",), output="logits")],
            outputs=["logits"],
        )
        expected = {"logits": np.zeros((1, 2, 3), dtype=np.float32)}
        normalized_graph, reduced_expected, argmax_meta, top_ops = self.module._normalize_graph_for_function(
            graph,
            expected,
            argmax_mode="off",
            argmax_axis=-1,
        )
        self.assertEqual(normalized_graph.outputs, ["logits"])
        self.assertIn("logits", reduced_expected)
        self.assertFalse(argmax_meta["enabled"])
        self.assertEqual(top_ops[0][0], "identity")

    def test_load_state_specs_parses_list_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = Path(temp_dir) / "states.json"
            spec_path.write_text(
                '[{"name":"k_cache","shape":[1,2,3,4],"dtype":"fp16"}]\n',
                encoding="utf-8",
            )
            specs = self.module._load_state_specs(spec_path)
        assert specs is not None
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "k_cache")
        self.assertEqual(specs[0].shape, (1, 2, 3, 4))
        self.assertEqual(specs[0].dtype, "fp16")

    def test_load_state_specs_rejects_bad_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec_path = Path(temp_dir) / "states.json"
            spec_path.write_text('{"states":"not-a-list"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module._load_state_specs(spec_path)

    def test_temporary_capture_training_mode_toggles_and_restores(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.training = False
                self.is_training = False
                self.train_calls = 0
                self.eval_calls = 0

            def train(self) -> None:
                self.training = True
                self.train_calls += 1

            def eval(self) -> None:
                self.training = False
                self.eval_calls += 1

        model = DummyModel()
        with self.module._temporary_capture_training_mode(model, enabled=True):
            self.assertTrue(model.training)
            self.assertTrue(model.is_training)

        self.assertFalse(model.training)
        self.assertFalse(model.is_training)
        self.assertEqual(model.train_calls, 1)
        self.assertEqual(model.eval_calls, 1)

    def test_temporary_capture_training_mode_noop_when_disabled(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.training = False
                self.is_training = False
                self.train_calls = 0
                self.eval_calls = 0

            def train(self) -> None:
                self.training = True
                self.train_calls += 1

            def eval(self) -> None:
                self.training = False
                self.eval_calls += 1

        model = DummyModel()
        with self.module._temporary_capture_training_mode(model, enabled=False):
            self.assertFalse(model.training)
            self.assertFalse(model.is_training)

        self.assertFalse(model.training)
        self.assertFalse(model.is_training)
        self.assertEqual(model.train_calls, 0)
        self.assertEqual(model.eval_calls, 0)


if __name__ == "__main__":
    unittest.main()
