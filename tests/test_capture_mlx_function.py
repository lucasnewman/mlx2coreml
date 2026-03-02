import unittest
from pathlib import Path

import numpy as np

from mlx2coreml.from_mlx import capture_graph_from_mlx_function
from mlx2coreml.op_registry import unsupported_op_details
from mlx2coreml.passes import normalize_graph


class CaptureMlxFunctionTests(unittest.TestCase):
    def test_capture_swiglu_gated_silu_graph(self) -> None:
        import mlx.core as mx

        def silu(x):
            return x * mx.sigmoid(x)

        def swiglu(gate, x):
            return silu(gate) * x

        rng = np.random.default_rng(0)
        inputs = {
            "gate": rng.standard_normal((2, 4), dtype=np.float32),
            "x": rng.standard_normal((2, 4), dtype=np.float32),
        }

        graph, _, expected = capture_graph_from_mlx_function(
            dot_output_path=Path("artifacts/test_capture_swiglu/swiglu.dot"),
            inputs=inputs,
            function=swiglu,
        )

        self.assertEqual([node.op for node in graph.nodes], ["sigmoid", "multiply", "multiply"])
        self.assertEqual(len(graph.outputs), 1)

        output_name = graph.outputs[0]
        captured = expected[output_name]
        reference = (inputs["gate"] * (1.0 / (1.0 + np.exp(-inputs["gate"])))) * inputs["x"]
        self.assertTrue(np.allclose(captured, reference, atol=1e-6, rtol=1e-6))

    def test_capture_full_mlp_with_swiglu(self) -> None:
        from dataclasses import dataclass

        import mlx.core as mx
        import mlx.nn as nn

        @dataclass
        class ModelArgs:
            hidden_size: int
            intermediate_size: int
            mlp_bias: bool = False

        def silu(x):
            return x * mx.sigmoid(x)

        def swiglu(gate, x):
            return silu(gate) * x

        class MLP(nn.Module):
            def __init__(self, args: ModelArgs):
                super().__init__()
                dim = args.hidden_size
                hidden_dim = args.intermediate_size
                mlp_bias = args.mlp_bias if hasattr(args, "mlp_bias") else False
                self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
                self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
                self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

            def __call__(self, x) -> mx.array:
                return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

        args = ModelArgs(hidden_size=8, intermediate_size=16, mlp_bias=False)
        model = MLP(args)
        rng = np.random.default_rng(0)
        inputs = {"x": rng.standard_normal((2, 8), dtype=np.float32)}

        graph, _, expected = capture_graph_from_mlx_function(
            dot_output_path=Path("artifacts/test_capture_mlp/mlp.dot"),
            inputs=inputs,
            function=lambda x: model(x),
            allow_unknown_sources=True,
        )

        self.assertEqual(len(graph.outputs), 1)
        output_name = graph.outputs[0]
        self.assertEqual(expected[output_name].shape, (2, 8))

        op_set = {node.op for node in graph.nodes}
        self.assertIn("sigmoid", op_set)
        self.assertIn("matmul", op_set)
        self.assertIn("split", op_set)
        self.assertIn("reshape", op_set)

        split_nodes = [node for node in graph.nodes if node.op == "split"]
        self.assertGreater(len(split_nodes), 0)
        self.assertTrue(any("output_index" in node.attrs for node in split_nodes))
        self.assertTrue(any("num_outputs" in node.attrs for node in split_nodes))

    def test_capture_rope_free_attention_fast_sdpa_graph(self) -> None:
        from dataclasses import dataclass

        import mlx.core as mx
        import mlx.nn as nn

        @dataclass
        class ModelArgs:
            hidden_size: int
            num_attention_heads: int
            num_key_value_heads: int
            head_dim: int | None = None

        class Attention(nn.Module):
            def __init__(self, args: ModelArgs):
                super().__init__()
                dim = args.hidden_size
                self.n_heads = n_heads = args.num_attention_heads
                self.n_kv_heads = n_kv_heads = args.num_key_value_heads
                self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
                self.scale = head_dim**-0.5
                self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
                self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
                self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

            def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
                bsz, seqlen, _ = x.shape
                queries = self.q_proj(x)
                keys = self.k_proj(x)
                values = self.v_proj(x)
                queries = queries.reshape(bsz, seqlen, self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                output = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=self.scale,
                    mask=mask,
                )
                output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
                return self.o_proj(output)

        def _collect_arrays(tree):
            arrays = []
            if isinstance(tree, dict):
                for value in tree.values():
                    arrays.extend(_collect_arrays(value))
                return arrays
            if isinstance(tree, (list, tuple)):
                for value in tree:
                    arrays.extend(_collect_arrays(value))
                return arrays
            return [tree]

        args = ModelArgs(hidden_size=32, num_attention_heads=4, num_key_value_heads=4, head_dim=8)
        model = Attention(args)
        model_params = _collect_arrays(model.parameters())
        if model_params:
            mx.eval(*model_params)

        rng = np.random.default_rng(0)
        seqlen = 16
        inputs = {
            "x": rng.standard_normal((2, seqlen, 32), dtype=np.float32),
            "mask": np.triu(np.full((1, 1, seqlen, seqlen), -1e9, dtype=np.float32), 1),
        }

        graph, _, expected = capture_graph_from_mlx_function(
            dot_output_path=Path("artifacts/test_capture_attention_no_rope/attention.dot"),
            inputs=inputs,
            function=lambda x, mask: model(x, mask=mask),
            allow_unknown_sources=True,
        )

        self.assertEqual(len(graph.outputs), 1)
        output_name = graph.outputs[0]
        self.assertEqual(expected[output_name].shape, (2, seqlen, 32))
        self.assertTrue(np.all(np.isfinite(expected[output_name])))

        ops = [node.op for node in graph.nodes]
        self.assertEqual(ops.count("softmax"), 1)
        self.assertIn("multiply", ops)

        softmax_node = next(node for node in graph.nodes if node.op == "softmax")
        producer_by_output = {node.output: node for node in graph.nodes}
        softmax_input = softmax_node.inputs[0]
        self.assertIn(softmax_input, producer_by_output)

        add_node = producer_by_output[softmax_input]
        self.assertEqual(add_node.op, "add")
        add_input_ops = {
            producer_by_output[name].op if name in producer_by_output else "source"
            for name in add_node.inputs
        }
        self.assertIn("matmul", add_input_ops)
        self.assertIn("broadcast", add_input_ops)

        softmax_consumers = [node for node in graph.nodes if softmax_node.output in node.inputs]
        self.assertEqual(len(softmax_consumers), 1)
        self.assertEqual(softmax_consumers[0].op, "matmul")

    def test_capture_llama3_rope_graph_with_and_without_materialized_freqs(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        class Llama3RoPE(nn.Module):
            def __init__(
                self,
                dims: int,
                max_position_embeddings: int = 2048,
                traditional: bool = False,
                base: float = 10000.0,
                scaling_config: dict | None = None,
            ):
                super().__init__()
                self.dims = dims
                self.max_position_embeddings = max_position_embeddings
                self.traditional = traditional

                cfg = scaling_config or {
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                }
                factor = cfg["factor"]
                low_freq_factor = cfg.get("low_freq_factor", 1.0)
                high_freq_factor = cfg.get("high_freq_factor", 4.0)
                old_context_len = cfg.get("original_max_position_embeddings", 8192)

                low_freq_period = old_context_len / low_freq_factor
                high_freq_period = old_context_len / high_freq_factor

                freqs = base ** (mx.arange(0, dims, 2) / dims)
                periods = 2 * mx.pi * freqs

                freqs = mx.where(periods > low_freq_period, freqs * factor, freqs)
                is_medium_freq = (periods > high_freq_period) & (periods < low_freq_period)
                smooth_factors = (old_context_len / periods - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                smooth_freqs = freqs / ((1.0 - smooth_factors) / factor + smooth_factors)
                self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

            def __call__(self, x, offset: int = 0):
                return mx.fast.rope(
                    x,
                    self.dims,
                    traditional=self.traditional,
                    base=None,
                    scale=1.0,
                    offset=offset,
                    freqs=self._freqs,
                )

        rng = np.random.default_rng(0)
        inputs = {
            "x": rng.standard_normal((2, 4, 16, 64), dtype=np.float32),
            "offset": np.array(0, dtype=np.int32),
        }

        def _capture(model: Llama3RoPE, dot_path: Path):
            graph, _, expected = capture_graph_from_mlx_function(
                dot_output_path=dot_path,
                inputs=inputs,
                function=lambda x, offset: model(x, offset=int(offset)),
                allow_unknown_sources=True,
                # This callable performs Python int() on an MLX scalar input, which is
                # not transform-safe under export_function callback capture.
                capture_mode="dot",
            )
            normalized = normalize_graph(graph)
            return normalized, expected

        rope_lazy = Llama3RoPE(dims=64, traditional=False)
        lazy_graph, lazy_expected = _capture(
            rope_lazy, Path("artifacts/test_capture_llama3_rope/llama3_rope_lazy.dot")
        )
        lazy_ops = {node.op for node in lazy_graph.nodes}
        self.assertIn("rope", lazy_ops)
        self.assertTrue({"arange", "power", "greater", "less", "select", "bitwiseand"}.issubset(lazy_ops))

        lazy_missing = {detail["op"] for detail in unsupported_op_details(lazy_graph)}
        self.assertNotIn("bitwisebinary", lazy_missing)
        self.assertNotIn("broadcast", lazy_missing)
        self.assertNotIn("greater", lazy_missing)
        self.assertNotIn("less", lazy_missing)
        self.assertNotIn("select", lazy_missing)
        self.assertNotIn("rope", lazy_missing)

        lazy_output_name = lazy_graph.outputs[0]
        self.assertEqual(lazy_expected[lazy_output_name].shape, (2, 4, 16, 64))
        self.assertTrue(np.all(np.isfinite(lazy_expected[lazy_output_name])))

        rope_materialized = Llama3RoPE(dims=64, traditional=False)
        mx.eval(rope_materialized._freqs)
        mat_graph, mat_expected = _capture(
            rope_materialized,
            Path("artifacts/test_capture_llama3_rope/llama3_rope_materialized.dot"),
        )
        mat_ops = {node.op for node in mat_graph.nodes}
        self.assertEqual(mat_ops, {"rope"})

        mat_missing = {detail["op"] for detail in unsupported_op_details(mat_graph)}
        self.assertEqual(mat_missing, set())

        mat_output_name = mat_graph.outputs[0]
        self.assertEqual(mat_expected[mat_output_name].shape, (2, 4, 16, 64))
        self.assertTrue(np.all(np.isfinite(mat_expected[mat_output_name])))

    def test_capture_transformer_block_callback_and_audit_supported(self) -> None:
        from dataclasses import dataclass

        import mlx.core as mx
        import mlx.nn as nn

        @dataclass
        class ModelArgs:
            hidden_size: int
            intermediate_size: int
            num_attention_heads: int
            num_key_value_heads: int
            rms_norm_eps: float
            head_dim: int | None = None
            rope_theta: float = 10000.0
            rope_traditional: bool = False

        def swiglu(gate, x):
            return (gate * mx.sigmoid(gate)) * x

        class Llama3RoPE(nn.Module):
            def __init__(self, dims: int, traditional: bool = False, base: float = 10000.0):
                super().__init__()
                self.dims = dims
                self.traditional = traditional

                factor = 8.0
                low_freq_factor = 1.0
                high_freq_factor = 4.0
                old_context_len = 8192
                low_freq_period = old_context_len / low_freq_factor
                high_freq_period = old_context_len / high_freq_factor

                freqs = base ** (mx.arange(0, dims, 2) / dims)
                periods = 2 * mx.pi * freqs
                freqs = mx.where(periods > low_freq_period, freqs * factor, freqs)
                is_medium_freq = (periods > high_freq_period) & (periods < low_freq_period)
                smooth_factors = (old_context_len / periods - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                smooth_freqs = freqs / ((1.0 - smooth_factors) / factor + smooth_factors)
                self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

            def __call__(self, x):
                return mx.fast.rope(
                    x,
                    self.dims,
                    traditional=self.traditional,
                    base=None,
                    scale=1.0,
                    offset=0,
                    freqs=self._freqs,
                )

        class Attention(nn.Module):
            def __init__(self, args: ModelArgs):
                super().__init__()
                dim = args.hidden_size
                self.n_heads = n_heads = args.num_attention_heads
                self.n_kv_heads = n_kv_heads = args.num_key_value_heads
                self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
                self.scale = head_dim**-0.5

                self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
                self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
                self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
                self.rope = Llama3RoPE(self.head_dim, traditional=args.rope_traditional, base=args.rope_theta)

            def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
                bsz, seqlen, _ = x.shape
                queries = self.q_proj(x)
                keys = self.k_proj(x)
                values = self.v_proj(x)
                queries = queries.reshape(bsz, seqlen, self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(bsz, seqlen, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                queries = self.rope(queries)
                keys = self.rope(keys)
                output = mx.fast.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    scale=self.scale,
                    mask=mask,
                )
                output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
                return self.o_proj(output)

        class MLP(nn.Module):
            def __init__(self, args: ModelArgs):
                super().__init__()
                dim = args.hidden_size
                hidden_dim = args.intermediate_size
                self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
                self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
                self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

            def __call__(self, x) -> mx.array:
                return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

        class TransformerBlock(nn.Module):
            def __init__(self, args: ModelArgs):
                super().__init__()
                self.self_attn = Attention(args)
                self.mlp = MLP(args)
                self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
                self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

            def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
                r = self.self_attn(self.input_layernorm(x), mask)
                h = x + r
                r = self.mlp(self.post_attention_layernorm(h))
                return h + r

        def _collect_arrays(tree):
            arrays = []
            if isinstance(tree, dict):
                for value in tree.values():
                    arrays.extend(_collect_arrays(value))
                return arrays
            if isinstance(tree, (list, tuple)):
                for value in tree:
                    arrays.extend(_collect_arrays(value))
                return arrays
            return [tree]

        args = ModelArgs(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            head_dim=8,
        )
        block = TransformerBlock(args)
        model_params = _collect_arrays(block.parameters())
        if model_params:
            mx.eval(*model_params)

        rng = np.random.default_rng(0)
        seqlen = 8
        inputs = {
            "x": rng.standard_normal((2, seqlen, args.hidden_size), dtype=np.float32),
            "mask": np.triu(np.full((1, 1, seqlen, seqlen), -1e9, dtype=np.float32), 1),
        }

        graph, _, expected = capture_graph_from_mlx_function(
            dot_output_path=Path("artifacts/test_capture_transformer_block/transformer_block.dot"),
            inputs=inputs,
            function=lambda x, mask: block(x, mask=mask),
            allow_unknown_sources=True,
        )
        normalized = normalize_graph(graph)

        output_name = normalized.outputs[0]
        self.assertEqual(expected[output_name].shape, (2, seqlen, args.hidden_size))
        self.assertTrue(np.all(np.isfinite(expected[output_name])))

        op_set = {node.op for node in normalized.nodes}
        self.assertTrue({"reshape", "transpose", "broadcast", "rmsnorm", "softmax", "sigmoid", "rope"}.issubset(op_set))

        unsupported = unsupported_op_details(normalized)
        self.assertEqual(unsupported, [])


if __name__ == "__main__":
    unittest.main()
