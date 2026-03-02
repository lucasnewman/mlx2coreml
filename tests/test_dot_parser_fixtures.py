import unittest
from pathlib import Path

from mlx2coreml.from_mlx import parse_mlx_dot_to_graph
from mlx2coreml.ir import TensorSpec

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dot"


def _read_fixture(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8")


class DotParserFixtureTests(unittest.TestCase):
    def test_parse_linear_smoke_fixture(self) -> None:
        graph = parse_mlx_dot_to_graph(
            _read_fixture("linear_smoke.dot"),
            input_specs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("w", (3, 4), "fp32"),
                TensorSpec("b", (2, 4), "fp32"),
                TensorSpec("z", (2, 4), "fp32"),
            ],
        )
        self.assertEqual([spec.name for spec in graph.inputs], ["x", "w", "b", "z"])
        self.assertEqual([node.op for node in graph.nodes], ["matmul", "add", "maximum"])
        self.assertEqual(graph.outputs, ["out"])
        self.assertEqual(graph.nodes[0].source, "mlx_dot:10:Matmul")

    def test_toposort_handles_out_of_order_op_definitions(self) -> None:
        graph = parse_mlx_dot_to_graph(
            _read_fixture("out_of_order.dot"),
            input_specs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("y", (2, 3), "fp32"),
                TensorSpec("z", (2, 3), "fp32"),
            ],
        )
        self.assertEqual([node.op for node in graph.nodes], ["add", "multiply"])
        self.assertEqual(graph.nodes[0].output, "t0")
        self.assertEqual(graph.nodes[1].output, "out")

    def test_parser_uses_last_node_output_when_sink_is_missing(self) -> None:
        graph = parse_mlx_dot_to_graph(
            _read_fixture("no_sink.dot"),
            input_specs=[
                TensorSpec("a", (2, 3), "fp32"),
                TensorSpec("b", (2, 3), "fp32"),
            ],
        )
        self.assertEqual(graph.outputs, ["t1"])
        self.assertEqual([node.op for node in graph.nodes], ["add", "subtract"])

    def test_parser_rejects_missing_input_specs(self) -> None:
        with self.assertRaisesRegex(ValueError, "source nodes without input specs: w"):
            parse_mlx_dot_to_graph(
                _read_fixture("linear_smoke.dot"),
                input_specs=[
                    TensorSpec("x", (2, 3), "fp32"),
                    TensorSpec("b", (2, 4), "fp32"),
                    TensorSpec("z", (2, 4), "fp32"),
                ],
            )

    def test_parser_can_keep_unknown_sources_when_enabled(self) -> None:
        graph = parse_mlx_dot_to_graph(
            _read_fixture("linear_smoke.dot"),
            input_specs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("b", (2, 4), "fp32"),
                TensorSpec("z", (2, 4), "fp32"),
            ],
            allow_unknown_sources=True,
        )
        self.assertIn("w", [spec.name for spec in graph.inputs])

    def test_parser_rejects_cyclic_dependencies(self) -> None:
        cyclic_dot = """
digraph {
{ rank=source; "x"; }
{ rank=source; "y"; }
{ 1 [label ="Add", shape=rectangle]; }
"b" -> 1
"x" -> 1
{ "a"; }
1 -> "a"
{ 2 [label ="Multiply", shape=rectangle]; }
"a" -> 2
"y" -> 2
{ "b"; }
2 -> "b"
}
""".strip()
        with self.assertRaisesRegex(ValueError, "unresolved/cyclic"):
            parse_mlx_dot_to_graph(
                cyclic_dot,
                input_specs=[
                    TensorSpec("x", (2, 3), "fp32"),
                    TensorSpec("y", (2, 3), "fp32"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
