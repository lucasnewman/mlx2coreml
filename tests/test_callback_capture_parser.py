import unittest

import numpy as np

from mlx2coreml.from_mlx import parse_mlx_export_events_to_graph
from mlx2coreml.ir import TensorSpec


class CallbackCaptureParserTests(unittest.TestCase):
    def test_shape_op_attrs_are_recovered_from_primitive_arguments(self) -> None:
        events = [
            {"type": "inputs", "inputs": [("A", (1, 2), "mlx.core.float32")]},
            {"type": "keyword_inputs", "keywords": [("x", "A")]},
            {"type": "outputs", "outputs": [("D", (3, 2), "mlx.core.float32")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "Broadcast",
                "inputs": [("A", (1, 2), "mlx.core.float32")],
                "outputs": [("B", (3, 2), "mlx.core.float32")],
                "arguments": [(3, 2)],
            },
            {
                "type": "primitive",
                "name": "Reshape",
                "inputs": [("B", (3, 2), "mlx.core.float32")],
                "outputs": [("C", (2, 3), "mlx.core.float32")],
                "arguments": [(2, 3)],
            },
            {
                "type": "primitive",
                "name": "Transpose",
                "inputs": [("C", (2, 3), "mlx.core.float32")],
                "outputs": [("D", (3, 2), "mlx.core.float32")],
                "arguments": [[1, 0]],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[TensorSpec(name="x", shape=(1, 2), dtype="fp32")],
            allow_unknown_sources=False,
        )

        self.assertEqual([spec.name for spec in graph.inputs], ["x"])
        self.assertEqual(graph.outputs, ["D"])
        self.assertEqual([node.op for node in graph.nodes], ["broadcast", "reshape", "transpose"])
        self.assertEqual(graph.nodes[0].attrs["shape"], [3, 2])
        self.assertEqual(graph.nodes[1].attrs["shape"], [2, 3])
        self.assertEqual(graph.nodes[2].attrs["perm"], [1, 0])
        self.assertEqual(graph.nodes[0].inputs, ("x",))

    def test_constants_are_injected_as_const_nodes(self) -> None:
        const_val = np.arange(6, dtype=np.float32).reshape(2, 3)
        events = [
            {"type": "inputs", "inputs": [("A", (2, 3), "mlx.core.float32")]},
            {"type": "keyword_inputs", "keywords": [("x", "A")]},
            {"type": "outputs", "outputs": [("B", (2, 3), "mlx.core.float32")]},
            {"type": "constants", "constants": [("C", const_val)]},
            {
                "type": "primitive",
                "name": "Add",
                "inputs": [
                    ("A", (2, 3), "mlx.core.float32"),
                    ("C", (2, 3), "mlx.core.float32"),
                ],
                "outputs": [("B", (2, 3), "mlx.core.float32")],
                "arguments": [],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[TensorSpec(name="x", shape=(2, 3), dtype="fp32")],
            allow_unknown_sources=False,
        )

        self.assertEqual([spec.name for spec in graph.inputs], ["x"])
        self.assertEqual([node.op for node in graph.nodes], ["const", "add"])
        self.assertEqual(graph.nodes[1].inputs, ("x", "C"))
        self.assertTrue(np.array_equal(graph.nodes[0].attrs["value"], const_val))

    def test_astype_and_softmax_callback_arguments_are_parsed_safely(self) -> None:
        events = [
            {"type": "inputs", "inputs": [("A", (2, 3), "mlx.core.float32")]},
            {"type": "keyword_inputs", "keywords": [("x", "A")]},
            {"type": "outputs", "outputs": [("C", (2, 3), "mlx.core.float16")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "AsType",
                "inputs": [("A", (2, 3), "mlx.core.float32")],
                "outputs": [("B", (2, 3), "mlx.core.float16")],
                "arguments": [],
            },
            {
                "type": "primitive",
                "name": "Softmax",
                "inputs": [("B", (2, 3), "mlx.core.float16")],
                "outputs": [("C", (2, 3), "mlx.core.float16")],
                "arguments": [True],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[TensorSpec(name="x", shape=(2, 3), dtype="fp32")],
            allow_unknown_sources=False,
        )

        self.assertEqual([node.op for node in graph.nodes], ["astype", "softmax"])
        self.assertEqual(graph.nodes[0].attrs["dtype"], "fp16")
        self.assertNotIn("axis", graph.nodes[1].attrs)
        self.assertEqual(graph.nodes[1].attrs["precise"], True)

    def test_sdpa_primitive_state_arguments_are_parsed(self) -> None:
        events = [
            {
                "type": "inputs",
                "inputs": [
                    ("Q", (1, 2, 4, 8), "mlx.core.float32"),
                    ("K", (1, 2, 4, 8), "mlx.core.float32"),
                    ("V", (1, 2, 4, 8), "mlx.core.float32"),
                ],
            },
            {
                "type": "keyword_inputs",
                "keywords": [("q", "Q"), ("k", "K"), ("v", "V")],
            },
            {"type": "outputs", "outputs": [("O", (1, 2, 4, 8), "mlx.core.float32")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "ScaledDotProductAttention",
                "inputs": [
                    ("Q", (1, 2, 4, 8), "mlx.core.float32"),
                    ("K", (1, 2, 4, 8), "mlx.core.float32"),
                    ("V", (1, 2, 4, 8), "mlx.core.float32"),
                ],
                "outputs": [("O", (1, 2, 4, 8), "mlx.core.float32")],
                "arguments": [None, 0.35355338, False, False, False],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[
                TensorSpec(name="q", shape=(1, 2, 4, 8), dtype="fp32"),
                TensorSpec(name="k", shape=(1, 2, 4, 8), dtype="fp32"),
                TensorSpec(name="v", shape=(1, 2, 4, 8), dtype="fp32"),
            ],
            allow_unknown_sources=False,
        )

        self.assertEqual([node.op for node in graph.nodes], ["scaled_dot_product_attention"])
        attrs = graph.nodes[0].attrs
        self.assertAlmostEqual(float(attrs["scale"]), 0.35355338, places=6)
        self.assertEqual(attrs["do_causal"], False)
        self.assertEqual(attrs["has_sinks"], False)
        self.assertEqual(attrs["output_logsumexp"], False)

    def test_slice_update_arguments_are_parsed(self) -> None:
        events = [
            {
                "type": "inputs",
                "inputs": [
                    ("X", (1, 8, 256, 128), "mlx.core.float16"),
                    ("U", (1, 8, 8, 128), "mlx.core.float16"),
                ],
            },
            {"type": "keyword_inputs", "keywords": [("x", "X"), ("u", "U")]},
            {"type": "outputs", "outputs": [("Y", (1, 8, 256, 128), "mlx.core.float16")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "SliceUpdate",
                "inputs": [
                    ("X", (1, 8, 256, 128), "mlx.core.float16"),
                    ("U", (1, 8, 8, 128), "mlx.core.float16"),
                ],
                "outputs": [("Y", (1, 8, 256, 128), "mlx.core.float16")],
                "arguments": [(0, 0, 0, 0), (1, 8, 8, 128), (1, 1, 1, 1)],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[
                TensorSpec(name="x", shape=(1, 8, 256, 128), dtype="fp16"),
                TensorSpec(name="u", shape=(1, 8, 8, 128), dtype="fp16"),
            ],
            allow_unknown_sources=False,
        )

        self.assertEqual([node.op for node in graph.nodes], ["slice_update"])
        attrs = graph.nodes[0].attrs
        self.assertEqual(attrs["begin"], [0, 0, 0, 0])
        self.assertEqual(attrs["end"], [1, 8, 8, 128])
        self.assertEqual(attrs["stride"], [1, 1, 1, 1])
        self.assertEqual(graph.nodes[0].inputs, ("x", "u"))



    def test_bfloat16_dtype_widens_to_fp32(self) -> None:
        events = [
            {"type": "inputs", "inputs": [("A", (2, 3), "mlx.core.bfloat16")]},
            {"type": "keyword_inputs", "keywords": [("x", "A")]},
            {"type": "outputs", "outputs": [("B", (2, 3), "mlx.core.bfloat16")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "Identity",
                "inputs": [("A", (2, 3), "mlx.core.bfloat16")],
                "outputs": [("B", (2, 3), "mlx.core.bfloat16")],
                "arguments": [],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[TensorSpec(name="x", shape=(2, 3), dtype="fp32")],
            allow_unknown_sources=False,
        )

        self.assertEqual(graph.inputs[0].dtype, "fp32")

    def test_rmsnorm_eps_argument_is_parsed(self) -> None:
        events = [
            {
                "type": "inputs",
                "inputs": [
                    ("X", (1, 4, 8), "mlx.core.float16"),
                    ("W", (8,), "mlx.core.float16"),
                ],
            },
            {"type": "keyword_inputs", "keywords": [("x", "X"), ("w", "W")]},
            {"type": "outputs", "outputs": [("Y", (1, 4, 8), "mlx.core.float16")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "RMSNorm",
                "inputs": [
                    ("X", (1, 4, 8), "mlx.core.float16"),
                    ("W", (8,), "mlx.core.float16"),
                ],
                "outputs": [("Y", (1, 4, 8), "mlx.core.float16")],
                "arguments": [1e-5],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[
                TensorSpec(name="x", shape=(1, 4, 8), dtype="fp16"),
                TensorSpec(name="w", shape=(8,), dtype="fp16"),
            ],
            allow_unknown_sources=False,
        )

        self.assertEqual([node.op for node in graph.nodes], ["rmsnorm"])
        self.assertAlmostEqual(float(graph.nodes[0].attrs["eps"]), 1e-5, places=8)

    def test_gather_and_squeeze_arguments_are_parsed(self) -> None:
        events = [
            {
                "type": "inputs",
                "inputs": [
                    ("A", (16, 8), "mlx.core.float32"),
                    ("I", (1, 4), "mlx.core.int32"),
                ],
            },
            {"type": "keyword_inputs", "keywords": [("x", "A"), ("indices", "I")]},
            {"type": "outputs", "outputs": [("C", (1, 4, 8), "mlx.core.float32")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "Gather",
                "inputs": [
                    ("A", (16, 8), "mlx.core.float32"),
                    ("I", (1, 4), "mlx.core.int32"),
                ],
                "outputs": [("B", (1, 4, 1, 8), "mlx.core.float32")],
                "arguments": [[0], (1, 8)],
            },
            {
                "type": "primitive",
                "name": "Squeeze",
                "inputs": [("B", (1, 4, 1, 8), "mlx.core.float32")],
                "outputs": [("C", (1, 4, 8), "mlx.core.float32")],
                "arguments": [[2]],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[
                TensorSpec(name="x", shape=(16, 8), dtype="fp32"),
                TensorSpec(name="indices", shape=(1, 4), dtype="int32"),
            ],
            allow_unknown_sources=False,
        )

        self.assertEqual([node.op for node in graph.nodes], ["gather", "squeeze"])
        self.assertEqual(graph.nodes[0].attrs["axis"], 0)
        self.assertEqual(graph.nodes[0].attrs["shape"], [1, 4, 1, 8])
        self.assertEqual(graph.nodes[1].attrs["axes"], [2])

    def test_split_expanddims_and_convolution_arguments_are_parsed(self) -> None:
        events = [
            {
                "type": "inputs",
                "inputs": [
                    ("A", (1, 9, 4), "mlx.core.float16"),
                    ("W", (4, 3, 1), "mlx.core.float16"),
                ],
            },
            {"type": "keyword_inputs", "keywords": [("x", "A"), ("w", "W")]},
            {"type": "outputs", "outputs": [("E", (1, 7, 1, 2), "mlx.core.float16")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "Convolution",
                "inputs": [
                    ("A", (1, 9, 4), "mlx.core.float16"),
                    ("W", (4, 3, 1), "mlx.core.float16"),
                ],
                "outputs": [("B", (1, 7, 4), "mlx.core.float16")],
                "arguments": [[1], [0], [0], [1], [1], 4, False],
            },
            {
                "type": "primitive",
                "name": "Split",
                "inputs": [("B", (1, 7, 4), "mlx.core.float16")],
                "outputs": [
                    ("C0", (1, 7, 2), "mlx.core.float16"),
                    ("C1", (1, 7, 2), "mlx.core.float16"),
                ],
                "arguments": [(2,), 2],
            },
            {
                "type": "primitive",
                "name": "ExpandDims",
                "inputs": [("C0", (1, 7, 2), "mlx.core.float16")],
                "outputs": [("D", (1, 7, 1, 2), "mlx.core.float16")],
                "arguments": [[2]],
            },
            {
                "type": "primitive",
                "name": "Exp",
                "inputs": [("D", (1, 7, 1, 2), "mlx.core.float16")],
                "outputs": [("E", (1, 7, 1, 2), "mlx.core.float16")],
                "arguments": [],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[
                TensorSpec(name="x", shape=(1, 9, 4), dtype="fp16"),
                TensorSpec(name="w", shape=(4, 3, 1), dtype="fp16"),
            ],
            allow_unknown_sources=False,
        )

        self.assertEqual(
            [node.op for node in graph.nodes],
            ["convolution", "split", "split", "expanddims", "exp"],
        )

        conv_attrs = graph.nodes[0].attrs
        self.assertEqual(conv_attrs["strides"], [1])
        self.assertEqual(conv_attrs["padding"], [0])
        self.assertEqual(conv_attrs["dilations"], [1])
        self.assertEqual(conv_attrs["groups"], 4)
        self.assertEqual(conv_attrs["transpose"], False)

        split_attrs_0 = graph.nodes[1].attrs
        split_attrs_1 = graph.nodes[2].attrs
        self.assertEqual(split_attrs_0["split_indices"], [2])
        self.assertEqual(split_attrs_0["axis"], 2)
        self.assertEqual(split_attrs_0["output_index"], 0)
        self.assertEqual(split_attrs_1["output_index"], 1)
        self.assertEqual(split_attrs_0["num_outputs"], 2)

        self.assertEqual(graph.nodes[3].attrs["axes"], [2])
        self.assertEqual(graph.nodes[4].attrs, {})

    def test_reduce_arguments_are_parsed(self) -> None:
        events = [
            {"type": "inputs", "inputs": [("A", (2, 3, 4), "mlx.core.float32")]},
            {"type": "keyword_inputs", "keywords": [("x", "A")]},
            {"type": "outputs", "outputs": [("B", (2, 3, 1), "mlx.core.float32")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "Reduce",
                "inputs": [("A", (2, 3, 4), "mlx.core.float32")],
                "outputs": [("B", (2, 3, 1), "mlx.core.float32")],
                "arguments": [2, [2]],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[TensorSpec(name="x", shape=(2, 3, 4), dtype="fp32")],
            allow_unknown_sources=False,
        )

        self.assertEqual([node.op for node in graph.nodes], ["reduce"])
        attrs = graph.nodes[0].attrs
        self.assertEqual(attrs["mode"], 2)
        self.assertEqual(attrs["axes"], [2])
        self.assertEqual(attrs["keep_dims"], True)

    def test_slice_arguments_are_parsed(self) -> None:
        events = [
            {"type": "inputs", "inputs": [("A", (1, 96, 16), "mlx.core.float16")]},
            {"type": "keyword_inputs", "keywords": [("x", "A")]},
            {"type": "outputs", "outputs": [("B", (1, 1, 16), "mlx.core.float16")]},
            {"type": "constants", "constants": []},
            {
                "type": "primitive",
                "name": "Slice",
                "inputs": [("A", (1, 96, 16), "mlx.core.float16")],
                "outputs": [("B", (1, 1, 16), "mlx.core.float16")],
                "arguments": [(0, 2, 0), (1, 3, 16), (1, 1, 1)],
            },
        ]

        graph = parse_mlx_export_events_to_graph(
            events=events,
            input_specs=[TensorSpec(name="x", shape=(1, 96, 16), dtype="fp16")],
            allow_unknown_sources=False,
        )

        self.assertEqual([node.op for node in graph.nodes], ["slice"])
        attrs = graph.nodes[0].attrs
        self.assertEqual(attrs["begin"], [0, 2, 0])
        self.assertEqual(attrs["end"], [1, 3, 16])
        self.assertEqual(attrs["stride"], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
