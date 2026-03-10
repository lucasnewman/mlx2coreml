import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringArithmeticAliasesExtendedTests(unittest.TestCase):
    def test_math_and_concat_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("u", (2, 3), "fp32"),
                TensorSpec("p", (2, 3), "fp32"),
                TensorSpec("c0", (2, 2), "fp32"),
                TensorSpec("c1", (2, 1), "fp32"),
                TensorSpec("ia", (2, 3), "int32"),
                TensorSpec("ib", (2, 3), "int32"),
            ],
            nodes=[
                Node("arccos", ("u",), "acos"),
                Node("arcsin", ("u",), "asin"),
                Node("arctan", ("u",), "atan"),
                Node("arctanh", ("u",), "atanh"),
                Node("negative", ("u",), "neg"),
                Node("degrees", ("u",), "deg"),
                Node("radians", ("deg",), "rad"),
                Node("expm1", ("u",), "ex1"),
                Node("log1p", ("p",), "l1p"),
                Node("log2", ("p",), "l2"),
                Node("log10", ("p",), "l10"),
                Node("floor_divide", ("ia", "ib"), "fd"),
                Node("concatenate", ("c0", "c1"), "cat", attrs={"axis": 1}),
                Node("logsumexp", ("cat",), "lse", attrs={"axes": [1], "keep_dims": False}),
            ],
            outputs=["acos", "asin", "atan", "atanh", "neg", "rad", "ex1", "l1p", "l2", "l10", "fd", "lse"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in (
            "acos(",
            "asin(",
            "atan(",
            "atanh(",
            "concat(",
            "floor_div(",
            "reduce_log_sum_exp(",
            "exp(",
            "log(",
        ):
            self.assertIn(token, text)

    def test_concatenate_requires_input(self) -> None:
        graph = Graph(inputs=[], nodes=[Node("concatenate", tuple(), "out")], outputs=["out"])
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_exp_and_greaterequal_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("y", (2, 3), "fp32"),
            ],
            nodes=[
                Node("exp", ("x",), "ex"),
                Node("greaterequal", ("ex", "y"), "ge"),
            ],
            outputs=["ex", "ge"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("exp(", text)
        self.assertIn("greater_equal(", text)


if __name__ == "__main__":
    unittest.main()
