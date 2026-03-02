import os
import unittest
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np

from mlx2coreml.from_mlx import capture_graph_from_mlx_function
from mlx2coreml.lower_to_mil import (
    build_mil_program,
    compile_model_artifact,
    convert_program_to_model,
)
from mlx2coreml.op_registry import ensure_supported
from mlx2coreml.passes import normalize_graph


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


class EndToEndTransformerBlockParityTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MX2MIL_RUN_E2E_TRANSFORMER") == "1",
        "Set MX2MIL_RUN_E2E_TRANSFORMER=1 to run TransformerBlock CoreML-vs-MLX parity test.",
    )
    def test_static_transformer_block_parity_compiled(self) -> None:
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

        mx.random.seed(0)
        args = ModelArgs(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            head_dim=8,
        )
        block = TransformerBlock(args)
        params = _collect_arrays(block.parameters())
        if params:
            mx.eval(*params)

        rng = np.random.default_rng(0)
        seqlen = 8
        inputs = {
            "x": rng.standard_normal((2, seqlen, args.hidden_size), dtype=np.float32),
            "mask": np.triu(np.full((1, 1, seqlen, seqlen), -1e9, dtype=np.float32), 1),
        }

        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts" / "test_transformer_block_parity"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        graph, _, expected = capture_graph_from_mlx_function(
            dot_output_path=artifacts_dir / "transformer_block.dot",
            inputs=inputs,
            function=lambda x, mask: block(x, mask=mask),
            allow_unknown_sources=True,
            capture_mode="callback",
        )
        normalized = normalize_graph(graph)
        ensure_supported(normalized)

        program = build_mil_program(normalized, deployment_target=ct.target.iOS18, normalize=False)
        model = convert_program_to_model(
            program,
            deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        model_path = artifacts_dir / "transformer_block.mlpackage"
        model.save(str(model_path))
        compiled_path = compile_model_artifact(model_path, artifacts_dir / "transformer_block.mlmodelc")

        compiled = ct.models.CompiledMLModel(str(compiled_path))
        outputs = compiled.predict(inputs)
        if len(outputs) != 1:
            self.fail(f"Expected one output, got keys: {list(outputs.keys())}")

        y_coreml = np.asarray(next(iter(outputs.values())))
        y_mlx = np.asarray(next(iter(expected.values())))
        self.assertEqual(y_coreml.shape, y_mlx.shape)

        atol = 5e-2
        rtol = 1e-2
        if not np.allclose(y_coreml, y_mlx, atol=atol, rtol=rtol):
            abs_err = np.abs(y_coreml - y_mlx)
            self.fail(
                "TransformerBlock parity check failed: "
                f"max_abs={float(np.max(abs_err))}, "
                f"mean_abs={float(np.mean(abs_err))}, "
                f"p99_abs={float(np.quantile(abs_err, 0.99))}, "
                f"atol={atol}, rtol={rtol}"
            )


if __name__ == "__main__":
    unittest.main()
