import unittest

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program
from mlx2coreml.op_registry import ensure_supported


class LoweringLinearBroadcastCompositeHelpers1Tests(unittest.TestCase):
    def test_linear_broadcast_and_tensor_helpers_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("y", (3, 4), "fp32"),
                TensorSpec("b", (2, 4), "fp32"),
                TensorSpec("s", (1, 3), "fp32"),
                TensorSpec("a", (2, 1), "fp32"),
                TensorSpec("bb", (1, 3), "fp32"),
                TensorSpec("v", (3,), "fp32"),
                TensorSpec("w", (5,), "fp32"),
                TensorSpec("p", (2, 3), "fp32"),
                TensorSpec("q", (4, 3), "fp32"),
                TensorSpec("tx", (2, 3, 4), "fp32"),
                TensorSpec("ty", (5, 4, 3), "fp32"),
            ],
            nodes=[
                Node("addmm", ("b", "x", "y"), "mm"),
                Node("broadcast_to", ("s",), "bcast", attrs={"shape": [2, 3]}),
                Node("broadcast_arrays", ("a", "bb"), "a_bcast", attrs={"input_index": 0}),
                Node("broadcast_arrays", ("a", "bb"), "bb_bcast", attrs={"input_index": 1}),
                Node("outer", ("v", "w"), "outer_out"),
                Node("inner", ("p", "q"), "inner_out"),
                Node("tensordot", ("tx", "ty"), "td", attrs={"axes": [[1, 2], [2, 1]]}),
            ],
            outputs=["mm", "bcast", "a_bcast", "bb_bcast", "outer_out", "inner_out", "td"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("tile(", "matmul(", "reshape(", "add(", "transpose("):
            self.assertIn(token, text)

    def test_broadcast_to_rejects_incompatible_shape(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32")],
            nodes=[Node("broadcast_to", ("x",), "out", attrs={"shape": [2, 2]})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_addmm_swaps_matmul_inputs_when_capture_orders_weight_before_activation(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("bias", (400, 384), "fp32"),
                TensorSpec("weight", (384, 384), "fp32"),
                TensorSpec("x", (400, 384), "fp32"),
            ],
            nodes=[Node("addmm", ("bias", "weight", "x"), "out")],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("matmul(", text)
        self.assertIn("add(", text)

    def test_addmm_accepts_capture_order_with_bias_last(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (400, 384), "fp32"),
                TensorSpec("weight", (384, 1536), "fp32"),
                TensorSpec("bias", (400, 1536), "fp32"),
            ],
            nodes=[Node("addmm", ("x", "weight", "bias"), "out")],
            outputs=["out"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("matmul(", text)
        self.assertIn("add(", text)

    def test_inner_requires_matching_last_dimension(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (4, 2), "fp32")],
            nodes=[Node("inner", ("x", "y"), "out")],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_broadcast_arrays_requires_valid_input_index(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("a", (2, 1), "fp32"), TensorSpec("b", (1, 3), "fp32")],
            nodes=[Node("broadcast_arrays", ("a", "b"), "out", attrs={"input_index": 2})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_tensordot_requires_matching_contract_dims(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3, 4), "fp32"), TensorSpec("y", (5, 4, 2), "fp32")],
            nodes=[Node("tensordot", ("x", "y"), "out", attrs={"axes": [[1, 2], [2, 1]]})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)


class LoweringLinearBroadcastCompositeHelpers2Tests(unittest.TestCase):
    def test_closeness_and_nan_helpers_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("y", (2, 3), "fp32"),
            ],
            nodes=[
                Node("isclose", ("x", "y"), "c", attrs={"rtol": 1e-4, "atol": 1e-6, "equal_nan": True}),
                Node("allclose", ("x", "y"), "ac", attrs={"rtol": 1e-4, "atol": 1e-6, "equal_nan": True}),
                Node("nan_to_num", ("x",), "xn", attrs={"nan": 0.5, "posinf": 9.0, "neginf": -9.0}),
            ],
            outputs=["c", "ac", "xn"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("less_equal(", "logical_or(", "not_equal(", "select(", "equal("):
            self.assertIn(token, text)

    def test_nan_to_num_int_input_is_passthrough(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "int32")],
            nodes=[Node("nan_to_num", ("x",), "out")],
            outputs=["out"],
        )
        graph.validate()
        program = build_mil_program(graph)
        self.assertIn("identity(", str(program))


class LoweringLinearBroadcastCompositeHelpers3Tests(unittest.TestCase):
    def test_diag_diagonal_trace_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("v", (4,), "fp32"),
                TensorSpec("m", (4, 4), "fp32"),
                TensorSpec("t", (2, 3, 4), "fp32"),
            ],
            nodes=[
                Node("diag", ("v",), "dv", attrs={"k": 1}),
                Node("diag", ("m",), "dm", attrs={"k": -1}),
                Node("diagonal", ("t",), "dt", attrs={"axis1": 1, "axis2": 2, "offset": 1}),
                Node("trace", ("t",), "tr", attrs={"axis1": 1, "axis2": 2, "offset": 0}),
            ],
            outputs=["dv", "dm", "dt", "tr"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("scatter_nd(", "gather(", "reduce_sum(", "reshape("):
            self.assertIn(token, text)

    def test_diag_rejects_rank_gt2(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3, 4), "fp32")],
            nodes=[Node("diag", ("x",), "out")],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_diagonal_rejects_same_axes(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3, 4), "fp32")],
            nodes=[Node("diagonal", ("x",), "out", attrs={"axis1": 1, "axis2": 1})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)


class LoweringLinearBroadcastCompositeHelpers4Tests(unittest.TestCase):
    def test_tri_tril_triu_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 4, 5), "fp32"),
                TensorSpec("m", (4, 5), "fp32"),
            ],
            nodes=[
                Node("tri", tuple(), "t", attrs={"n": 4, "m": 5, "k": -1, "dtype": "fp32"}),
                Node("tril", ("x",), "xl", attrs={"k": 0}),
                Node("triu", ("x",), "xu", attrs={"k": 1}),
                Node("tril", ("m",), "ml", attrs={"k": -1}),
                Node("triu", ("m",), "mu", attrs={"k": 0}),
            ],
            outputs=["t", "xl", "xu", "ml", "mu"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("range_1d(", "less_equal(", "greater_equal(", "select(", "tile("):
            self.assertIn(token, text)

    def test_tri_requires_n_attr(self) -> None:
        graph = Graph(
            inputs=[],
            nodes=[Node("tri", tuple(), "out", attrs={"m": 3})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_tril_requires_rank_at_least_2(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (4,), "fp32")],
            nodes=[Node("tril", ("x",), "out")],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)


class LoweringLinearBroadcastCompositeHelpers5Tests(unittest.TestCase):
    def test_logical_compare_ops_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 3), "fp32"),
                TensorSpec("y", (2, 3), "fp32"),
                TensorSpec("i", (2, 3), "int32"),
            ],
            nodes=[
                Node("isnan", ("x",), "n"),
                Node("isinf", ("x",), "inf"),
                Node("isfinite", ("x",), "fin"),
                Node("isneginf", ("x",), "ninf"),
                Node("isposinf", ("x",), "pinf"),
                Node("all", ("i",), "all_i"),
                Node("any", ("i",), "any_i"),
                Node("array_equal", ("x", "y"), "eq_xy"),
            ],
            outputs=["n", "inf", "fin", "ninf", "pinf", "all_i", "any_i", "eq_xy"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in ("not_equal(", "equal(", "logical_not(", "reduce_prod(", "reduce_sum(", "greater("):
            self.assertIn(token, text)

    def test_array_equal_shape_mismatch_returns_false(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (1, 3), "fp32")],
            nodes=[Node("array_equal", ("x", "y"), "eq")],
            outputs=["eq"],
        )
        graph.validate()
        program = build_mil_program(graph)
        text = str(program)
        self.assertIn("fill(", text)
        self.assertIn("squeeze(", text)

    def test_any_requires_single_input(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
            nodes=[Node("any", ("x", "y"), "out")],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)


class LoweringLinearBroadcastCompositeHelpers6Tests(unittest.TestCase):
    def test_eye_meshgrid_kron_logaddexp_lower(self) -> None:
        graph = Graph(
            inputs=[
                TensorSpec("x", (3,), "fp32"),
                TensorSpec("y", (2,), "fp32"),
                TensorSpec("a", (2, 2), "fp32"),
                TensorSpec("b", (2, 1), "fp32"),
                TensorSpec("p", (2, 3), "fp32"),
                TensorSpec("q", (2, 3), "fp32"),
            ],
            nodes=[
                Node("meshgrid", ("x", "y"), "gx", attrs={"input_index": 0, "indexing": "xy"}),
                Node("meshgrid", ("x", "y"), "gy", attrs={"input_index": 1, "indexing": "xy"}),
                Node("kron", ("a", "b"), "k"),
                Node("eye", tuple(), "e", attrs={"n": 3, "m": 4, "k": 1, "dtype": "fp32"}),
                Node("logaddexp", ("p", "q"), "l"),
            ],
            outputs=["gx", "gy", "k", "e", "l"],
        )
        graph.validate()
        ensure_supported(graph)
        program = build_mil_program(graph)
        text = str(program)
        for token in (
            "range_1d(",
            "equal(",
            "tile(",
            "reshape(",
            "minimum(",
            "maximum(",
            "exp(",
            "log(",
        ):
            self.assertIn(token, text)

    def test_eye_requires_n_attr(self) -> None:
        graph = Graph(
            inputs=[],
            nodes=[Node("eye", tuple(), "out", attrs={"m": 4})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)

    def test_meshgrid_requires_vector_inputs(self) -> None:
        graph = Graph(
            inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2,), "fp32")],
            nodes=[Node("meshgrid", ("x", "y"), "out", attrs={"input_index": 0})],
            outputs=["out"],
        )
        graph.validate()
        with self.assertRaises(ValueError):
            build_mil_program(graph)


if __name__ == "__main__":
    unittest.main()
