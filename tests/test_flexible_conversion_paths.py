import unittest

import coremltools as ct
import numpy as np

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program, convert_program_to_model


class FlexibleConversionPathTests(unittest.TestCase):
    def _convert_with_flex_x(
        self,
        graph: Graph,
        *,
        x_shapes: list[tuple[int, ...]],
        extra_inputs: list[ct.TensorType] | None = None,
    ) -> None:
        graph.validate()
        program = build_mil_program(graph)
        model_inputs: list[ct.TensorType] = [
            ct.TensorType(
                name="x",
                shape=ct.EnumeratedShapes(shapes=x_shapes),
                dtype=np.float32,
            )
        ]
        if extra_inputs:
            model_inputs.extend(extra_inputs)
        model = convert_program_to_model(
            program,
            deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            inputs=model_inputs,
        )
        self.assertIsNotNone(model)

    def test_flexible_reshape_converts(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (1, 8, 4), "fp32")],
            nodes=[Node("reshape", ("x",), "out", attrs={"shape": [1, -1, 4]})],
            outputs=["out"],
        )
        self._convert_with_flex_x(graph, x_shapes=[(1, 1, 4), (1, 8, 4)])

    def test_flexible_slice_converts(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (1, 8, 4), "fp32")],
            nodes=[
                Node(
                    "slice",
                    ("x",),
                    "out",
                    attrs={"begin": [0, 0, 0], "end": [1, 1, 4], "stride": [1, 1, 1]},
                )
            ],
            outputs=["out"],
        )
        self._convert_with_flex_x(graph, x_shapes=[(1, 1, 4), (1, 8, 4)])

    def test_flexible_gather_converts(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (1, 8, 4), "fp32"), TensorSpec("idx", (1,), "int32")],
            nodes=[Node("gather", ("x", "idx"), "out", attrs={"axis": 1})],
            outputs=["out"],
        )
        idx_input = ct.TensorType(name="idx", shape=(1,), dtype=np.int32)
        self._convert_with_flex_x(
            graph,
            x_shapes=[(1, 1, 4), (1, 8, 4)],
            extra_inputs=[idx_input],
        )


if __name__ == "__main__":
    unittest.main()
