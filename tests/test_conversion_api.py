import unittest
from unittest.mock import patch

import coremltools as ct
import numpy as np

from mlx2coreml.conversion import ConversionConfig, convert_mlx_to_coreml, prepare_mlx_conversion
from mlx2coreml.ir import Graph, Node, TensorSpec


class ConversionApiTests(unittest.TestCase):
    def test_prepare_mlx_conversion_defaults_to_calling_target(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1, 2), dtype="fp32")],
            nodes=[Node(op="add", inputs=("x", "x"), output="out")],
            outputs=["out"],
        )
        normalized_inputs = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
        expected_outputs = {"out": np.asarray([[2.0, 4.0]], dtype=np.float32)}

        class DummyTarget:
            def __call__(self, x):
                return x

        target = DummyTarget()
        with patch("mlx2coreml.conversion.capture_graph_from_mlx_function") as capture_mock:
            capture_mock.return_value = (graph, normalized_inputs, expected_outputs)
            prepare_mlx_conversion(target, normalized_inputs)

        self.assertIs(capture_mock.call_args.kwargs["function"], target)

    def test_prepare_mlx_conversion_collects_graph_metadata(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1, 2), dtype="fp32")],
            nodes=[Node(op="add", inputs=("x", "x"), output="out")],
            outputs=["out"],
        )
        normalized_inputs = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
        expected_outputs = {"out": np.asarray([[2.0, 4.0]], dtype=np.float32)}

        with patch("mlx2coreml.conversion.capture_graph_from_mlx_function") as capture_mock:
            capture_mock.return_value = (graph, normalized_inputs, expected_outputs)
            prepared = prepare_mlx_conversion(lambda x: x, normalized_inputs)

        self.assertEqual(prepared.graph.outputs, ["out"])
        self.assertEqual(prepared.normalized_graph.outputs, ["out"])
        self.assertEqual(prepared.top_ops[0], ["add", 1])
        self.assertEqual(prepared.extra_input_names, [])
        self.assertTrue(prepared.weights_captured_as_constants)
        self.assertEqual(prepared.inference_summary["with_shape"], 2)
        self.assertEqual(prepared.inference_summary["with_dtype"], 2)
        self.assertEqual(prepared.unsupported_details, [])
        self.assertEqual(capture_mock.call_args.kwargs["capture_mode"], "callback")
        self.assertTrue(capture_mock.call_args.kwargs["allow_unknown_sources"])

    def test_prepare_mlx_conversion_passes_target_for_training_mode_toggle(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1, 2), dtype="fp32")],
            nodes=[Node(op="add", inputs=("x", "x"), output="out")],
            outputs=["out"],
        )
        normalized_inputs = {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)}
        expected_outputs = {"out": np.asarray([[2.0, 4.0]], dtype=np.float32)}

        class DummyTarget:
            def __call__(self, x):
                return x

        target = DummyTarget()
        capture_function = lambda x: x

        with patch("mlx2coreml.conversion.capture_graph_from_mlx_function") as capture_mock:
            with patch("mlx2coreml.conversion.temporary_capture_training_mode") as mode_mock:
                capture_mock.return_value = (graph, normalized_inputs, expected_outputs)
                prepare_mlx_conversion(
                    target,
                    normalized_inputs,
                    config=ConversionConfig(capture_is_training=True),
                    capture_function=capture_function,
                )

        self.assertEqual(mode_mock.call_args.kwargs["enabled"], True)
        self.assertIs(mode_mock.call_args.args[0], target)
        self.assertIs(capture_mock.call_args.kwargs["function"], capture_function)

    def test_convert_mlx_to_coreml_runs_full_pipeline(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1, 4), dtype="fp32")],
            nodes=[Node(op="add", inputs=("x", "x"), output="out")],
            outputs=["out"],
        )
        normalized_inputs = {"x": np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)}
        expected_outputs = {"out": normalized_inputs["x"] * 2.0}
        fake_model = object()
        config = ConversionConfig(
            deployment_target="iOS18",
            flex_input_lens=[1, 4, 8],
            flex_input_names={"x"},
        )

        with patch("mlx2coreml.conversion.capture_graph_from_mlx_function") as capture_mock:
            with patch("mlx2coreml.conversion.build_mil_program") as build_program_mock:
                with patch("mlx2coreml.conversion.convert_program_to_model") as convert_mock:
                    capture_mock.return_value = (graph, normalized_inputs, expected_outputs)
                    build_program_mock.return_value = "program"
                    convert_mock.return_value = fake_model
                    result = convert_mlx_to_coreml(lambda x: x, normalized_inputs, config=config)

        self.assertEqual(result.program, "program")
        self.assertIs(result.model, fake_model)
        self.assertEqual(result.flex_input_shapes, {"x": [[1, 4], [1, 1], [1, 8]]})
        self.assertEqual(build_program_mock.call_args.kwargs["deployment_target"], ct.target.iOS18)
        self.assertFalse(build_program_mock.call_args.kwargs["normalize"])
        self.assertEqual(convert_mock.call_args.kwargs["deployment_target"], ct.target.iOS18)
        self.assertEqual(convert_mock.call_args.kwargs["compute_precision"], "auto")
        self.assertEqual(convert_mock.call_args.kwargs["compute_units"], "all")
        self.assertEqual(convert_mock.call_args.kwargs["convert_to"], "mlprogram")
        converted_inputs = convert_mock.call_args.kwargs["inputs"]
        self.assertIsNotNone(converted_inputs)
        self.assertIsInstance(converted_inputs[0].shape, ct.EnumeratedShapes)
