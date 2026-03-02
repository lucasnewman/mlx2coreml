import unittest

from mlx2coreml.from_mlx import make_mock_smoke_graph, parse_mlx_dot_to_graph
from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported
from mlx2coreml.passes import normalize_graph


class LoweringSmokeTests(unittest.TestCase):
    def test_mock_graph_is_supported_and_lowers(self) -> None:
        graph = make_mock_smoke_graph()
        ensure_supported(graph)
        program = build_mil_program(graph)
        self.assertIn("main", program.functions)

    def test_unsupported_op_fails_fast(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(1,), dtype="fp32")],
            nodes=[Node(op="unknown_op", inputs=("x",), output="y")],
            outputs=["y"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            ensure_supported(graph)

    def test_parse_simple_dot(self) -> None:
        dot = """digraph {
{ rank=source; "x"; }
{ rank=source; "w"; }
{ 1 [label ="Matmul", shape=rectangle]; }
"x" -> 1
"w" -> 1
{ "m0"; }
1 -> "m0"
{ rank=sink; "m0"; }
}"""
        specs = [
            TensorSpec(name="x", shape=(2, 3), dtype="fp32"),
            TensorSpec(name="w", shape=(3, 4), dtype="fp32"),
        ]
        graph = parse_mlx_dot_to_graph(dot, input_specs=specs)
        self.assertEqual([node.op for node in graph.nodes], ["matmul"])
        self.assertEqual(graph.outputs, ["m0"])

    def test_constant_node_normalizes_and_lowers(self) -> None:
        graph = Graph(
            inputs=[TensorSpec(name="x", shape=(2,), dtype="fp32")],
            nodes=[
                Node(op="Const", inputs=tuple(), output="c", attrs={"val": [1.0, 2.0]}),
                Node(op="Add", inputs=("x", "c"), output="out"),
            ],
            outputs=["out"],
        )
        graph.validate()
        normalized = normalize_graph(graph)
        ensure_supported(normalized)
        program = build_mil_program(normalized, normalize=False)
        text = str(program)
        self.assertIn("add(", text)
        self.assertIn("[1.0, 2.0]", text)


if __name__ == "__main__":
    unittest.main()
