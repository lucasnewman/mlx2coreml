import os
import unittest
from pathlib import Path

import coremltools as ct
import numpy as np

from mlx2coreml.ir import Graph, Node, TensorSpec
from mlx2coreml.lower_to_mil import build_mil_program, convert_program_to_model


def _build_llama3_freqs(dims: int) -> np.ndarray:
    base = 10000.0
    factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192.0

    low_freq_period = old_context_len / low_freq_factor
    high_freq_period = old_context_len / high_freq_factor

    freqs = base ** (np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims))
    periods = 2.0 * np.pi * freqs

    freqs = np.where(periods > low_freq_period, freqs * factor, freqs)
    is_medium_freq = np.logical_and(periods > high_freq_period, periods < low_freq_period)
    smooth_factors = (old_context_len / periods - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth_freqs = freqs / ((1.0 - smooth_factors) / factor + smooth_factors)
    return np.where(is_medium_freq, smooth_freqs, freqs).astype(np.float32)


class EndToEndRoPEParityTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MX2MIL_RUN_E2E_ROPE") == "1",
        "Set MX2MIL_RUN_E2E_ROPE=1 to run RoPE CoreML-vs-MLX parity test.",
    )
    def test_llama3_rope_custom_freqs_parity(self) -> None:
        import mlx.core as mx

        dims = 64
        graph = Graph(
            inputs=[
                TensorSpec("x", (2, 4, 16, dims), "fp32"),
                TensorSpec("offset", tuple(), "int32"),
                TensorSpec("freqs", (dims // 2,), "fp32"),
            ],
            nodes=[
                Node(
                    "rope",
                    ("x", "offset", "freqs"),
                    "out",
                    attrs={"dims": dims, "traditional": False, "base": None, "scale": 1.0},
                )
            ],
            outputs=["out"],
        )
        graph.validate()

        program = build_mil_program(graph, deployment_target=ct.target.iOS18, normalize=False)
        model = convert_program_to_model(
            program,
            deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts" / "test_rope_parity"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / "rope_parity.mlpackage"
        model.save(str(model_path))

        rng = np.random.default_rng(7)
        x = (0.5 * rng.standard_normal((2, 4, 16, dims), dtype=np.float32)).astype(np.float32)
        freqs = _build_llama3_freqs(dims)
        offset = np.array([0], dtype=np.int32)

        y_mlx = np.asarray(
            mx.fast.rope(
                mx.array(x),
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=mx.array(freqs),
            )
        )

        pred = model.predict({"x": x, "offset": offset, "freqs": freqs})
        _, y_coreml = next(iter(pred.items()))
        y_coreml = np.asarray(y_coreml)

        atol = 2e-2
        rtol = 5e-3
        if not np.allclose(y_coreml, y_mlx, atol=atol, rtol=rtol):
            abs_err = np.abs(y_coreml - y_mlx)
            self.fail(
                "RoPE parity check failed: "
                f"max_abs={float(np.max(abs_err))}, "
                f"mean_abs={float(np.mean(abs_err))}, "
                f"p99_abs={float(np.quantile(abs_err, 0.99))}, "
                f"atol={atol}, rtol={rtol}"
            )


if __name__ == "__main__":
    unittest.main()
