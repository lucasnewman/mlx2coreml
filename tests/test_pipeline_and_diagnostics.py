import unittest

import numpy as np

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.op_registry import UnsupportedOpsError, ensure_supported, unsupported_op_details
from mlx2coreml.passes import infer_graph_specs, normalize_graph, summarize_inference


class PipelineAndDiagnosticsTests(unittest.TestCase):
    def test_normalize_graph_canonicalizes_and_eliminates_identity(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
            nodes=[
                Node("Add", ("x", "y"), "a0"),
                Node("StopGradient", ("a0",), "a1"),
                Node("Multiply", ("a1", "y"), "out"),
            ],
            outputs=["out"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        self.assertEqual([node.op for node in normalized.nodes], ["add", "multiply"])
        self.assertEqual(normalized.nodes[1].inputs[0], "a0")

    def test_unsupported_diagnostics_include_status_and_source(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (4,), "fp32")],
            nodes=[Node("cummax", ("x",), "out", source="unit:test-node")],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(UnsupportedOpsError) as ctx:
            ensure_supported(graph)

        err = ctx.exception
        self.assertEqual(err.first_op, "cummax")
        self.assertIn("cummax", err.all_ops)
        self.assertEqual(err.details[0]["status"], "not_yet_implemented")
        self.assertEqual(err.details[0]["source"], "unit:test-node")

    def test_unsupported_diagnostics_aggregate_count_and_primitive_context(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (4,), "fp32")],
            nodes=[
                Node("BitwiseAnd", ("x", "x"), "y0", source="mlx_export:21:BitwiseAnd"),
                Node("BitwiseAnd", ("y0", "x"), "y1", source="mlx_export:22:BitwiseAnd"),
            ],
            outputs=["y1"],
        )
        graph.validate()

        details = unsupported_op_details(graph)
        self.assertEqual(len(details), 1)
        detail = details[0]
        self.assertEqual(detail["op"], "bitwiseand")
        self.assertEqual(detail["count"], 2)
        self.assertEqual(detail["source_kind"], "mlx_export")
        self.assertEqual(detail["primitive"], "BitwiseAnd")
        self.assertEqual(detail["primitive_index"], "21")
        self.assertGreaterEqual(len(detail["sample_sources"]), 1)
        self.assertGreaterEqual(len(detail["sample_contexts"]), 1)
        self.assertEqual(detail["sample_contexts"][0]["output"], "y0")

        with self.assertRaises(UnsupportedOpsError) as ctx:
            ensure_supported(graph)
        message = str(ctx.exception)
        self.assertIn("count=2", message)
        self.assertIn("primitive=BitwiseAnd", message)

    def test_normalization_keeps_identity_when_it_is_graph_output(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32")],
            nodes=[Node("StopGradient", ("x",), "out")],
            outputs=["out"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        self.assertEqual(len(normalized.nodes), 1)
        self.assertEqual(normalized.nodes[0].op, "stop_gradient")
        self.assertEqual(normalized.nodes[0].output, "out")

    def test_normalization_canonicalizes_input_dtypes_and_tensor_names(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x:0", (2, 3), "float32"), TensorSpec("1y", (2, 3), "int")],
            nodes=[Node("Add", ("x:0", "1y"), "out-0")],
            outputs=["out-0"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        self.assertEqual(normalized.inputs[0].dtype, "fp32")
        self.assertEqual(normalized.inputs[1].dtype, "int32")
        self.assertEqual(normalized.inputs[0].name, "x_0")
        self.assertEqual(normalized.inputs[1].name, "t_1y")
        self.assertEqual(normalized.nodes[0].inputs, ("x_0", "t_1y"))
        self.assertEqual(normalized.nodes[0].output, "out_0")
        self.assertEqual(normalized.outputs, ["out_0"])

    def test_normalization_canonicalizes_constant_attrs(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2,), "fp32")],
            nodes=[
                Node("Const", tuple(), "c0", attrs={"val": np.array([1.0, 2.0], dtype=np.float32)}),
                Node("Add", ("x", "c0"), "out"),
            ],
            outputs=["out"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        self.assertEqual(normalized.nodes[0].op, "constant")
        self.assertIn("value", normalized.nodes[0].attrs)
        self.assertEqual(normalized.nodes[0].attrs["value"], [1.0, 2.0])

    def test_normalization_keeps_large_constant_arrays_uninlined(self) -> None:
        graph = Graph(
            inputs=[],
            nodes=[
                Node(
                    "Const",
                    tuple(),
                    "c0",
                    attrs={"val": np.arange(129, dtype=np.float32)},
                )
            ],
            outputs=["c0"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        self.assertEqual(normalized.nodes[0].op, "constant")
        self.assertIsInstance(normalized.nodes[0].attrs["value"], np.ndarray)
        self.assertEqual(normalized.nodes[0].attrs["value"].shape, (129,))

    def test_inference_tracks_shape_and_dtype(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
            nodes=[
                Node("Add", ("x", "y"), "s"),
                Node("Mean", ("s",), "m", attrs={"axes": [1], "keep_dims": False}),
                Node("Argmax", ("s",), "a", attrs={"axis": 1, "keep_dims": False}),
            ],
            outputs=["m", "a"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        inferred = infer_graph_specs(normalized)
        summary = summarize_inference(inferred)
        self.assertEqual(inferred["s"].shape, (2, 3))
        self.assertEqual(inferred["s"].dtype, "fp32")
        self.assertEqual(inferred["m"].shape, (2,))
        self.assertEqual(inferred["a"].dtype, "int32")
        self.assertGreaterEqual(summary["with_shape"], 4)

    def test_sdpa_mask_attrs_are_canonicalized(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 4, 8), "fp32"),
                TensorSpec("k", (1, 2, 4, 8), "fp32"),
                TensorSpec("v", (1, 2, 4, 8), "fp32"),
                TensorSpec("m", (1, 1, 4, 4), "fp32"),
            ],
            nodes=[
                Node(
                    "ScaledDotProductAttention",
                    ("q", "k", "v", "m"),
                    "out",
                    attrs={"do_causal": 1, "scale": "0.5", "mask_mode": "BOOL"},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        attrs = normalized.nodes[0].attrs
        self.assertEqual(normalized.nodes[0].op, "scaled_dot_product_attention")
        self.assertEqual(attrs["do_causal"], True)
        self.assertEqual(attrs["scale"], 0.5)
        self.assertEqual(attrs["mask_mode"], "bool")

    def test_sdpa_without_mask_sets_mask_mode_none(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 4, 8), "fp32"),
                TensorSpec("k", (1, 2, 4, 8), "fp32"),
                TensorSpec("v", (1, 2, 4, 8), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v"),
                    "out",
                    attrs={"do_causal": False},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        self.assertEqual(normalized.nodes[0].attrs["mask_mode"], "none")


if __name__ == "__main__":
    unittest.main()
