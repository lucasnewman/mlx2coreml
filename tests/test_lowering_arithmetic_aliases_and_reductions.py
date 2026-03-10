import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringArithmeticAliasesAndReductionsTests(unittest.TestCase):
    def test_arithmetic_alias_ops_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("a", (2, 4), "fp32"),
                TensorSpec("b", (2, 4), "fp32"),
                TensorSpec("c", (2, 4), "fp32"),
                TensorSpec("d", (2, 4), "fp32"),
                TensorSpec("e", (2, 4), "fp32"),
                TensorSpec("f", (2, 4), "fp32"),
            ],
            nodes=[
                Node("subtract", ("a", "b"), "t0"),
                Node("multiply", ("t0", "c"), "t1"),
                Node("divide", ("t1", "d"), "t2"),
                Node("power", ("t2", "e"), "t3"),
                Node("remainder", ("t3", "c"), "t4"),
                Node("reciprocal", ("f",), "t5"),
                Node("add", ("t4", "t5"), "out"),
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("sub(", "mul(", "real_div(", "pow(", "mod(", "inverse("):
            self.assertIn(token, text)

    def test_reduction_ops_lower(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3, 4), "fp32")],
            nodes=[
                Node("sum", ("x",), "s", attrs={"axes": [2], "keep_dims": False}),
                Node("mean", ("x",), "m", attrs={"axes": [2], "keep_dims": False}),
                Node("min", ("x",), "n", attrs={"axes": [2], "keep_dims": False}),
                Node("max", ("x",), "xmax", attrs={"axes": [2], "keep_dims": False}),
                Node("prod", ("x",), "p", attrs={"axes": [2], "keep_dims": False}),
                Node("argmax", ("x",), "am", attrs={"axis": 2, "keep_dims": False}),
                Node("argmin", ("x",), "ai", attrs={"axis": 1, "keep_dims": True}),
            ],
            outputs=["s", "m", "n", "xmax", "p", "am", "ai"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in (
            "reduce_sum(",
            "reduce_mean(",
            "reduce_min(",
            "reduce_max(",
            "reduce_prod(",
            "reduce_argmax(",
            "reduce_argmin(",
        ):
            self.assertIn(token, text)

    def test_arg_reduction_requires_axis(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3, 4), "fp32")],
            nodes=[Node("argmax", ("x",), "am")],
            outputs=["am"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_reduce_primitive_mode_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3, 4), "fp32"),
                TensorSpec("b", (2, 3, 4), "bool"),
            ],
            nodes=[
                Node("reduce", ("x",), "sum_k", attrs={"mode": 2, "axes": [2], "keep_dims": True}),
                Node("reduce", ("x",), "max_k", attrs={"mode": 5, "axes": [2], "keep_dims": True}),
                Node("reduce", ("b",), "all_k", attrs={"mode": 0, "axes": [2], "keep_dims": True}),
            ],
            outputs=["sum_k", "max_k", "all_k"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("reduce_sum(", text)
        self.assertIn("reduce_max(", text)
        self.assertIn("reduce_prod(", text)


if __name__ == "__main__":
    unittest.main()
