import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringShapeIndexTransformsTests(unittest.TestCase):
    def test_shape_transform_ops_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("v", (4,), "fp32"),
                TensorSpec("m", (2, 3), "fp32"),
                TensorSpec("t", (2, 3, 4), "fp32"),
            ],
            nodes=[
                Node("flatten", ("t",), "flat", attrs={"shape": [24]}),
                Node("unflatten", ("flat",), "restored", attrs={"shape": [2, 3, 4]}),
                Node("atleast_1d", ("v",), "v1"),
                Node("atleast_2d", ("v",), "v2"),
                Node("atleast_3d", ("m",), "m3"),
            ],
            outputs=["flat", "restored", "v1", "v2", "m3"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("reshape(", text)
        self.assertIn("expand_dims(", text)

    def test_indexing_transform_ops_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3, 4), "fp32"),
                TensorSpec("indices", (2,), "int32"),
                TensorSpec("indices_along", (2, 3, 2), "int32"),
            ],
            nodes=[
                Node("moveaxis", ("x",), "mx", attrs={"source": 2, "destination": 0}),
                Node("swapaxes", ("mx",), "sw", attrs={"axis1": 1, "axis2": 2}),
                Node(
                    "slice",
                    ("sw",),
                    "sl",
                    attrs={"begin": [0, 0, 0], "end": [4, 2, 2], "stride": [1, 1, 1]},
                ),
                Node("take", ("sw", "indices"), "tk", attrs={"axis": 1}),
                Node("take_along_axis", ("x", "indices_along"), "tka", attrs={"axis": 2}),
            ],
            outputs=["sl", "tk", "tka"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("transpose(", text)
        self.assertIn("slice_by_index(", text)
        self.assertIn("gather(", text)
        self.assertIn("gather_along_axis(", text)


if __name__ == "__main__":
    unittest.main()
