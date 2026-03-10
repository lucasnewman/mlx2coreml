import unittest
from unittest.mock import patch

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringTensorCreationAndFastHelpersTests(unittest.TestCase):
    def test_compare_and_select_ops_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("y", (2, 3), "fp32"),
                TensorSpec("cond", (2, 3), "bool"),
            ],
            nodes=[
                Node("greater", ("x", "y"), "gt"),
                Node("less", ("x", "y"), "lt"),
                Node("select", ("cond", "x", "y"), "sel"),
            ],
            outputs=["gt", "lt", "sel"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("greater(", "less(", "select("):
            self.assertIn(token, text)

    def test_rope_op_lowers_to_trig_rotate_pattern(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (1, 2, 3, 8), "fp32"),
                TensorSpec("offset", tuple(), "int32"),
                TensorSpec("freqs", (4,), "fp32"),
            ],
            nodes=[
                Node(
                    "rope",
                    ("x", "offset", "freqs"),
                    "out",
                    attrs={"dims": 8, "traditional": False, "base": None, "scale": 1.0},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("range_1d(", "cos(", "sin(", "slice_by_index(", "concat("):
            self.assertIn(token, text)

    def test_softmax_sigmoid_rmsnorm_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3, 8), "fp32"),
                TensorSpec("w", (8,), "fp32"),
            ],
            nodes=[
                Node("sigmoid", ("x",), "sig"),
                Node("softmax", ("x",), "sm"),
                Node("rmsnorm", ("x", "w"), "rn", attrs={"eps": 1e-5}),
            ],
            outputs=["sig", "sm", "rn"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("sigmoid(", "softmax(", "reduce_mean(", "rsqrt("):
            self.assertIn(token, text)

    def test_reshape_transpose_broadcast_lower(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (1, 2), "fp32")],
            nodes=[
                Node("broadcast", ("x",), "b0", attrs={"shape": [3, 2]}),
                Node("reshape", ("b0",), "r0", attrs={"shape": [2, 3]}),
                Node("transpose", ("r0",), "t0", attrs={"perm": [1, 0]}),
            ],
            outputs=["t0"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("tile(", "reshape(", "transpose("):
            self.assertIn(token, text)

    def test_bitwisebinary_logical_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("a", (4,), "bool"),
                TensorSpec("b", (4,), "bool"),
            ],
            nodes=[
                Node("bitwisebinary", ("a", "b"), "and_out", attrs={"mode": 0}),
                Node("bitwisebinary", ("a", "b"), "or_out", attrs={"mode": 1}),
            ],
            outputs=["and_out", "or_out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("logical_and(", "logical_or("):
            self.assertIn(token, text)

    def test_sdpa_fused_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 4, 8), "fp32"),
                TensorSpec("k", (1, 2, 4, 8), "fp32"),
                TensorSpec("v", (1, 2, 4, 8), "fp32"),
                TensorSpec("m", (1, 1, 4, 4), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v", "m"),
                    "out",
                    attrs={"scale": 1.0 / (8.0**0.5), "do_causal": False},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        self.assertIn("scaled_dot_product_attention(", str(program))

    def test_sdpa_grouped_query_repeats_kv_heads(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 4, 8, 8), "fp32"),
                TensorSpec("k", (1, 2, 8, 8), "fp32"),
                TensorSpec("v", (1, 2, 8, 8), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v"),
                    "out",
                    attrs={"scale": 1.0 / (8.0**0.5), "do_causal": True},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("gather(", text)
        self.assertIn("scaled_dot_product_attention(", text)

    def test_sdpa_fused_causal_mask_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 8, 8), "fp32"),
                TensorSpec("k", (1, 2, 8, 8), "fp32"),
                TensorSpec("v", (1, 2, 8, 8), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v"),
                    "out",
                    attrs={"scale": 1.0 / (8.0**0.5), "do_causal": True},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("scaled_dot_product_attention(", text)
        self.assertIn("causal_mask", text)

    def test_sdpa_fused_causal_plus_explicit_additive_mask_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 8, 8), "fp32"),
                TensorSpec("k", (1, 2, 8, 8), "fp32"),
                TensorSpec("v", (1, 2, 8, 8), "fp32"),
                TensorSpec("m", (1, 1, 8, 8), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v", "m"),
                    "out",
                    attrs={"scale": 1.0 / (8.0**0.5), "do_causal": True},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("scaled_dot_product_attention(", text)
        self.assertIn("mask_combined", text)

    def test_sdpa_decomposed_causal_plus_explicit_mask_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 8, 8), "fp32"),
                TensorSpec("k", (1, 2, 8, 8), "fp32"),
                TensorSpec("v", (1, 2, 8, 8), "fp32"),
                TensorSpec("m", (1, 1, 8, 8), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v", "m"),
                    "out",
                    attrs={"scale": 0.75, "do_causal": True},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("causal_mask", text)
        self.assertIn("scores_masked", text)
        self.assertIn("softmax(", text)
        self.assertNotIn("scaled_dot_product_attention(", text)

    def test_sdpa_symbolic_shape_path_supports_explicit_scale_without_causal(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 8, 8), "fp32"),
                TensorSpec("k", (1, 2, 8, 8), "fp32"),
                TensorSpec("v", (1, 2, 8, 8), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v"),
                    "out",
                    attrs={"scale": 0.75, "do_causal": False},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        with patch("mlx2coreml.lower_to_mil._shape_list_if_static", return_value=None):
            program = build_mil_program(graph)
        text = str(program)
        self.assertIn("scores_scaled", text)
        self.assertIn("softmax(", text)

    def test_sdpa_symbolic_shape_path_rejects_causal_mask(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("q", (1, 2, 8, 8), "fp32"),
                TensorSpec("k", (1, 2, 8, 8), "fp32"),
                TensorSpec("v", (1, 2, 8, 8), "fp32"),
            ],
            nodes=[
                Node(
                    "scaled_dot_product_attention",
                    ("q", "k", "v"),
                    "out",
                    attrs={"scale": 0.75, "do_causal": True},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        with patch("mlx2coreml.lower_to_mil._shape_list_if_static", return_value=None):
            with self.assertRaisesRegex(ValueError, "requires static shape"):
                build_mil_program(graph)

    def test_gather_and_squeeze_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (16, 8), "fp32"),
                TensorSpec("idx", (1, 4), "int32"),
            ],
            nodes=[
                Node("gather", ("x", "idx"), "g", attrs={"axis": 0, "shape": [1, 4, 1, 8]}),
                Node("squeeze", ("g",), "out", attrs={"axes": [2]}),
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("gather(", text)
        self.assertIn("squeeze(", text)


if __name__ == "__main__":
    unittest.main()
