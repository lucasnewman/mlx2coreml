import argparse
import json
import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.compilation import compile_mlmodel
from mlx2coreml.reporting import build_run_context, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run numeric parity between MLX and a converted weighted Core ML model artifact "
            "for the same fixed input_ids batch."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/llama32_1b_package"),
        help="Artifact directory from scripts/convert_mlx_lm_to_coreml.py.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override model id. Defaults to run report model_id.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt used when inputs.npz is missing.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Optional seq len used when inputs.npz is missing.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision passed to mlx_lm.load().",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Pass lazy=True to mlx_lm.load().",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Do not compile the model artifact; requires an existing model.mlmodelc.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
        help="Absolute tolerance for allclose check.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-2,
        help="Relative tolerance for allclose check.",
    )
    return parser.parse_args()


def _tokenize_prompt(tokenizer: Any, prompt: str, seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError(f"--seq-len must be positive, got {seq_len}.")

    token_ids = [int(v) for v in tokenizer.encode(prompt)]
    if not token_ids:
        raise ValueError("Tokenizer produced an empty prompt token sequence.")

    if len(token_ids) > seq_len:
        token_ids = token_ids[:seq_len]
    elif len(token_ids) < seq_len:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = token_ids[-1]
        token_ids = token_ids + [int(eos_id)] * (seq_len - len(token_ids))

    return np.asarray([token_ids], dtype=np.int32)


def _resolve_model_path(report: dict[str, Any], run_dir: Path) -> Path:
    model_str = report.get("artifacts", {}).get("model")
    if model_str:
        model_path = Path(model_str)
        if model_path.exists():
            return model_path
    for candidate in ("model.mlpackage", "model.mlmodel"):
        path = run_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate model artifact (model.mlpackage or model.mlmodel).")


def _load_input_ids(run_dir: Path) -> np.ndarray | None:
    path = run_dir / "inputs.npz"
    if not path.exists():
        return None
    data = np.load(path)
    if "input_ids" not in data:
        return None
    return np.asarray(data["input_ids"], dtype=np.int32)


def _normalize_expected_output(value: Any) -> np.ndarray:
    def _to_numpy(x: Any) -> np.ndarray:
        try:
            return np.asarray(x)
        except Exception:
            if hasattr(x, "astype"):
                return np.asarray(x.astype(mx.float32))
            raise

    if isinstance(value, dict):
        if not value:
            raise ValueError("MLX model returned an empty dict output.")
        _, first = next(iter(value.items()))
        return _to_numpy(first)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("MLX model returned an empty sequence output.")
        return _to_numpy(value[0])
    return _to_numpy(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Weighted CoreML Parity Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Model id: `{report['model_id']}`",
        f"- Run dir: `{report['run_dir']}`",
        f"- Input shape: `{tuple(report['input_shape'])}`",
        f"- Expected shape: `{tuple(report['expected_shape'])}`",
        f"- Predicted shape: `{tuple(report['predicted_shape'])}`",
        f"- Predicted output key: `{report['predicted_output_key']}`",
        f"- Max abs err: `{report['max_abs_err']}`",
        f"- Mean abs err: `{report['mean_abs_err']}`",
        f"- P99 abs err: `{report['p99_abs_err']}`",
        f"- Max rel err: `{report['max_rel_err']}`",
        f"- atol: `{report['atol']}`",
        f"- rtol: `{report['rtol']}`",
        f"- Core ML model: `{report['model_artifact']}`",
        f"- Compiled model: `{report['compiled_artifact']}`",
        f"- Report JSON: `{report['artifacts']['json']}`",
    ]
    if report.get("error"):
        lines.append(f"- Error: `{report['error']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing weighted probe report: {report_path}")

    probe_report = json.loads(report_path.read_text(encoding="utf-8"))
    model_id = args.model_id or probe_report.get("model_id")
    if not model_id:
        raise ValueError("Could not resolve model id from args or report.json.")

    input_ids = _load_input_ids(run_dir)

    from mlx_lm import load

    model, tokenizer = load(model_id, lazy=args.lazy_load, revision=args.revision)
    if hasattr(model, "eval"):
        model.eval()

    if input_ids is None:
        prompt = args.prompt or probe_report.get("prompt")
        seq_len = args.seq_len or probe_report.get("seq_len")
        if prompt is None or seq_len is None:
            raise ValueError(
                "inputs.npz missing; provide --prompt and --seq-len (or ensure report has prompt/seq_len)."
            )
        input_ids = _tokenize_prompt(tokenizer, str(prompt), int(seq_len))

    mlx_out = model(mx.array(input_ids, dtype=mx.int32))
    expected = _normalize_expected_output(mlx_out)

    model_path = _resolve_model_path(probe_report, run_dir)
    compiled_path = run_dir / "model.mlmodelc"
    if not args.skip_compile:
        compiled_path = compile_mlmodel(model_path, compiled_path)
    elif not compiled_path.exists():
        raise FileNotFoundError(
            f"--skip-compile set but compiled artifact not found: {compiled_path}"
        )

    compiled_model = ct.models.CompiledMLModel(str(compiled_path))
    predicted_raw = compiled_model.predict({"input_ids": input_ids.astype(np.int32)})
    if not predicted_raw:
        raise RuntimeError("Compiled model produced no outputs.")

    predicted_key, predicted_value = next(iter(predicted_raw.items()))
    predicted = np.asarray(predicted_value)

    if predicted.shape != expected.shape:
        if predicted.size == expected.size:
            predicted = predicted.reshape(expected.shape)
        else:
            raise RuntimeError(
                f"Shape mismatch: predicted {predicted.shape}, expected {expected.shape}."
            )

    abs_err = np.abs(predicted - expected)
    max_abs_err = float(np.max(abs_err))
    mean_abs_err = float(np.mean(abs_err))
    p99_abs_err = float(np.percentile(abs_err, 99.0))
    denom = np.maximum(np.abs(expected), 1e-12)
    max_rel_err = float(np.max(abs_err / denom))

    error: str | None = None
    status = "ok"
    if not np.allclose(predicted, expected, atol=args.atol, rtol=args.rtol):
        status = "failed"
        error = (
            "Parity check failed: "
            f"max_abs_err={max_abs_err}, max_rel_err={max_rel_err}, "
            f"atol={args.atol}, rtol={args.rtol}"
        )

    run_context = build_run_context(
        run_kind="mlx_lm_weighted_parity",
        deployment_target=probe_report.get("run_context", {}).get("deployment_target", "unknown"),
        convert_to=probe_report.get("run_context", {}).get("convert_to", "unknown"),
        seed=probe_report.get("run_context", {}).get("seed", 0),
    )
    parity_json_path = run_dir / "parity_report.json"
    parity_md_path = run_dir / "parity_report.md"

    parity_report = {
        "schema_version": run_context["schema_version"],
        "run_kind": "mlx_lm_weighted_parity",
        "run_context": run_context,
        "status": status,
        "model_id": model_id,
        "run_dir": str(run_dir),
        "input_shape": list(input_ids.shape),
        "expected_shape": list(expected.shape),
        "predicted_shape": list(predicted.shape),
        "predicted_output_key": predicted_key,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "p99_abs_err": p99_abs_err,
        "max_rel_err": max_rel_err,
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "model_artifact": str(model_path),
        "compiled_artifact": str(compiled_path),
        "artifacts": {
            "json": str(parity_json_path),
            "markdown": str(parity_md_path),
        },
        "error": error,
    }

    write_json(parity_json_path, parity_report)
    _write_markdown(parity_md_path, parity_report)

    print(f"Model: {model_id}")
    print(f"Status: {status}")
    print(f"Predicted output key: {predicted_key}")
    print(f"Max abs err: {max_abs_err:.8g}")
    print(f"Mean abs err: {mean_abs_err:.8g}")
    print(f"P99 abs err: {p99_abs_err:.8g}")
    print(f"Max rel err: {max_rel_err:.8g}")
    print(f"Wrote parity JSON: {parity_json_path}")
    print(f"Wrote parity markdown: {parity_md_path}")

    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
