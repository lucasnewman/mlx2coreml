import unittest

from mlx2coreml.ir import Graph, Node, StateSpec, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringStatefulPrimitivesTests(unittest.TestCase):
    def test_read_and_write_state_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("kv_cache", (1, 2, 4), "fp16"),
                TensorSpec("delta", (1, 2, 4), "fp16"),
            ],
            nodes=[
                Node("read_state", ("kv_cache",), "curr"),
                Node("add", ("curr", "delta"), "next"),
                Node("write_state", ("kv_cache", "next"), "out"),
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(
            graph,
            normalize=False,
            shared_state_specs=[StateSpec("kv_cache", (1, 2, 4), "fp16")],
        )
        text = str(program)
        self.assertIn("read_state(", text)
        self.assertIn("coreml_update_state(", text)

    def test_state_update_masked_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("kv_cache", (1, 2, 4), "fp16"),
                TensorSpec("write_value", (1, 2, 4), "fp16"),
                TensorSpec("mask", (1, 2, 4), "bool"),
            ],
            nodes=[
                Node("state_update_masked", ("kv_cache", "write_value", "mask"), "out"),
            ],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(
            graph,
            normalize=False,
            shared_state_specs=[StateSpec("kv_cache", (1, 2, 4), "fp16")],
        )
        text = str(program)
        self.assertIn("read_state(", text)
        self.assertIn("select(", text)
        self.assertIn("coreml_update_state(", text)


if __name__ == "__main__":
    unittest.main()
