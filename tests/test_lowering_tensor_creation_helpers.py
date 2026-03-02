import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringTensorCreationHelpersTests(unittest.TestCase):
    def test_creation_helper_ops_lower(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("condf", (2, 3), "fp32")],
            nodes=[
                Node("zeros", tuple(), "z", attrs={"shape": [2, 3], "value": 0.0}),
                Node("ones", tuple(), "o", attrs={"shape": [2, 3], "value": 1.0}),
                Node("full", tuple(), "f", attrs={"shape": [2, 3], "value": 2.5}),
                Node("zeros_like", ("x",), "zl"),
                Node("ones_like", ("x",), "ol"),
                Node("full_like", ("x",), "fl", attrs={"value": -1.0}),
                Node("arange", tuple(), "ar", attrs={"start": 0, "end": 6, "step": 1}),
                Node(
                    "linspace",
                    tuple(),
                    "ls",
                    attrs={"start": 0.0, "stop": 1.0, "num": 6, "endpoint": True, "dtype": "fp32"},
                ),
                Node("astype", ("x",), "xi", attrs={"dtype": "int32"}),
                Node("astype", ("condf",), "condb", attrs={"dtype": "bool"}),
                Node("where", ("condb", "o", "z"), "w"),
                Node("number_of_elements", ("x",), "n"),
                Node("stop_gradient", ("f",), "sg"),
            ],
            outputs=["z", "o", "f", "zl", "ol", "fl", "ar", "ls", "xi", "w", "n", "sg"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("fill(", "range_1d(", "cast(", "select(", "shape(", "reduce_prod("):
            self.assertIn(token, text)

    def test_full_requires_value(self) -> None:
        graph = Graph(
            inputs=[],
            nodes=[Node("full", tuple(), "f", attrs={"shape": [2, 3]})],
            outputs=["f"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_cast_requires_dtype(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32")],
            nodes=[Node("astype", ("x",), "c")],
            outputs=["c"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)


if __name__ == "__main__":
    unittest.main()
