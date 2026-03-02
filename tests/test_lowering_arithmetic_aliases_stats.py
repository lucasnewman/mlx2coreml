import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringArithmeticAliasesStatsTests(unittest.TestCase):
    def test_var_std_divmod_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3, 4), "fp32"),
                TensorSpec("a", (2, 3), "int32"),
                TensorSpec("b", (2, 3), "int32"),
            ],
            nodes=[
                Node("var", ("x",), "v", attrs={"axes": [1, 2], "keep_dims": False}),
                Node("std", ("x",), "s", attrs={"axes": [0, 2], "keep_dims": False, "ddof": 1}),
                Node("divmod", ("a", "b"), "q", attrs={"output": "quotient"}),
                Node("divmod", ("a", "b"), "r", attrs={"output": "remainder"}),
            ],
            outputs=["v", "s", "q", "r"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("reduce_mean(", "sqrt(", "floor_div(", "mod("):
            self.assertIn(token, text)

    def test_divmod_invalid_selector_raises(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("a", (2, 3), "int32"), TensorSpec("b", (2, 3), "int32")],
            nodes=[Node("divmod", ("a", "b"), "out", attrs={"output": "bad"})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_std_invalid_correction_raises(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32")],
            nodes=[Node("std", ("x",), "s", attrs={"axes": [1], "ddof": 4})],
            outputs=["s"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)


if __name__ == "__main__":
    unittest.main()
