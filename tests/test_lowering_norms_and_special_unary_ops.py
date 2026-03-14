import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringNormsAndSpecialUnaryOpsTests(unittest.TestCase):
    def test_tanh_and_erf_lower(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32")],
            nodes=[
                Node("tanh", ("x",), "t"),
                Node("erf", ("t",), "out"),
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("tanh(", text)
        self.assertIn("erf(", text)

    def test_layernorm_lowers_with_scale_and_bias(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 4), "fp32"),
                TensorSpec("weight", (4,), "fp32"),
                TensorSpec("bias", (4,), "fp32"),
            ],
            nodes=[
                Node(
                    "layernorm",
                    ("x", "weight", "bias"),
                    "out",
                    attrs={"epsilon": 1e-5},
                )
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("reduce_mean(", "sub(", "rsqrt(", "mul(", "add("):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
