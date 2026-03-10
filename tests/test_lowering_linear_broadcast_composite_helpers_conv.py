import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringLinearBroadcastCompositeHelpersConvTests(unittest.TestCase):
    def test_conv_family_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x1", (1, 2, 8), "fp32"),
                TensorSpec("w1", (3, 2, 3), "fp32"),
                TensorSpec("b1", (3,), "fp32"),
                TensorSpec("x2", (1, 2, 8, 8), "fp32"),
                TensorSpec("w2", (4, 2, 3, 3), "fp32"),
                TensorSpec("b2", (4,), "fp32"),
                TensorSpec("x3", (1, 1, 6, 6, 6), "fp32"),
                TensorSpec("w3", (2, 1, 3, 3, 3), "fp32"),
                TensorSpec("b3", (2,), "fp32"),
                TensorSpec("xt1", (1, 2, 4), "fp32"),
                TensorSpec("wt1", (2, 3, 3), "fp32"),
                TensorSpec("bt1", (3,), "fp32"),
                TensorSpec("xt2", (1, 2, 4, 4), "fp32"),
                TensorSpec("wt2", (2, 3, 3, 3), "fp32"),
                TensorSpec("bt2", (3,), "fp32"),
                TensorSpec("xt3", (1, 1, 3, 3, 3), "fp32"),
                TensorSpec("wt3", (1, 2, 2, 2, 2), "fp32"),
                TensorSpec("bt3", (2,), "fp32"),
            ],
            nodes=[
                Node(
                    "conv1d",
                    ("x1", "w1", "b1"),
                    "c1",
                    attrs={"strides": [2], "padding": [1], "pad_type": "custom"},
                ),
                Node("conv2d", ("x2", "w2", "b2"), "c2", attrs={"strides": [1, 1], "pad_type": "same"}),
                Node("conv3d", ("x3", "w3", "b3"), "c3", attrs={"pad_type": "valid"}),
                Node("conv_general", ("x2", "w2", "b2"), "cg", attrs={"stride": [1, 1], "padding": [1, 1]}),
                Node(
                    "conv_transpose1d",
                    ("xt1", "wt1", "bt1"),
                    "t1",
                    attrs={"strides": [2], "pad_type": "valid"},
                ),
                Node(
                    "conv_transpose2d",
                    ("xt2", "wt2", "bt2"),
                    "t2",
                    attrs={"strides": [2, 2], "pad_type": "valid", "output_shape": [1, 3, 9, 9]},
                ),
                Node("conv_transpose3d", ("xt3", "wt3", "bt3"), "t3", attrs={"pad_type": "same"}),
            ],
            outputs=["c1", "c2", "c3", "cg", "t1", "t2", "t3"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("conv(", text)
        self.assertIn("conv_transpose(", text)

    def test_conv_groups_must_be_positive(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (1, 2, 8, 8), "fp32"), TensorSpec("w", (4, 2, 3, 3), "fp32")],
            nodes=[Node("conv2d", ("x", "w"), "out", attrs={"groups": 0})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_convolution_channels_last_depthwise_pattern_lowers(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (1, 9, 4), "fp32"),
                TensorSpec("w", (4, 3, 1), "fp32"),
            ],
            nodes=[
                Node(
                    "convolution",
                    ("x", "w"),
                    "out",
                    attrs={"strides": [1], "padding": [0], "dilations": [1], "groups": 4},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("conv(", text)
        self.assertIn("transpose(", text)


if __name__ == "__main__":
    unittest.main()
