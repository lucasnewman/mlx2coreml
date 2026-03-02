import unittest

import numpy as np

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program


class LoweringMatmulLinearizationTests(unittest.TestCase):
    def test_matmul_with_const_rhs_lowers_to_linear(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32")],
            nodes=[
                Node("const", tuple(), "w", attrs={"value": np.ones((3, 4), dtype=np.float32)}),
                Node("matmul", ("x", "w"), "y"),
            ],
            outputs=["y"],
        )
        graph.validate()
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("linear(", text)

    def test_matmul_with_dynamic_rhs_stays_matmul(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("w", (3, 4), "fp32")],
            nodes=[Node("matmul", ("x", "w"), "y")],
            outputs=["y"],
        )
        graph.validate()
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("matmul(", text)


if __name__ == "__main__":
    unittest.main()
