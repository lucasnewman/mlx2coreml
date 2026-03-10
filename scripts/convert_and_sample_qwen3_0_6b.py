import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.compilation import compile_mlmodel  # noqa: E402
from mlx2coreml.reporting import build_run_context, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert and sample mlx-community/Qwen3-0.6B-bf16 through Core ML, "
            "with optional step-wise token validation against MLX."
        )
    )
    parser.add_argument(
        "--model-id",
        default="mlx-community/Qwen3-0.6B-bf16",
        help="Hugging Face model id loadable via mlx_lm.load().",
    )
    parser.add_argument(
        "--prompt",
        default="Write one short sentence about Apple Neural Engine acceleration.",
        help="Prompt used for conversion capture and sampling.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=96,
        help="Maximum context length used during conversion and sampling windowing.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of tokens to generate from the Core ML model.",
    )
    parser.add_argument(
        "--validate-steps",
        type=int,
        default=4,
        help=(
            "How many initial decode steps to validate against MLX next-token argmax. "
            "Set 0 to disable validation."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 uses greedy argmax.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Optional top-k filtering when temperature > 0.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base artifacts directory.",
    )
    parser.add_argument(
        "--run-name",
        default="qwen3_0_6b_convert_sample",
        help="Run directory name under artifacts dir.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--lazy-load", action="store_true")
    parser.add_argument(
        "--deployment-target",
        default="iOS18",
    )
    parser.add_argument(
        "--target-profile",
        default="ane_ios18",
        choices=["default", "ane_ios18", "conservative"],
    )
    parser.add_argument(
        "--compute-precision",
        default="fp16",
        choices=["auto", "fp16", "fp32"],
    )
    parser.add_argument(
        "--compute-units",
        default="all",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
    )
    parser.add_argument(
        "--capture-mode",
        choices=["callback", "dot"],
        default="callback",
    )
    parser.add_argument(
        "--allow-unknown-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--capture-is-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass through to conversion script: temporarily set mlx-lm model training mode "
            "during export/callback capture to avoid some fast custom kernels."
        ),
    )
    parser.add_argument(
        "--analyze-placement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through to conversion script. Disabled by default for faster local iteration.",
    )
    parser.add_argument(
        "--force-reconvert",
        action="store_true",
        help="Always run conversion even if an existing model artifact is present.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip MLX validation steps and only sample from Core ML.",
    )
    parser.add_argument(
        "--state-specs-json",
        type=Path,
        default=None,
        help=(
            "Optional state spec payload passed through to conversion. Only valid when captured "
            "graphs include state ops."
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_path(run_dir: Path, report: dict[str, Any]) -> Path:
    model_str = report.get("artifacts", {}).get("model")
    if model_str:
        model_path = Path(model_str)
        if model_path.exists():
            return model_path
    for candidate in ("model.mlpackage", "model.mlmodel"):
        path = run_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate converted model artifact.")


def _resolve_compiled_path(
    run_dir: Path, report: dict[str, Any], model_path: Path
) -> tuple[Path, float, bool, str]:
    compiled_str = report.get("artifacts", {}).get("compiled")
    if compiled_str:
        compiled_path = Path(compiled_str)
        if compiled_path.exists():
            return compiled_path, 0.0, False, "report"
    compiled_path = run_dir / "model.mlmodelc"
    if compiled_path.exists():
        return compiled_path, 0.0, False, "run_dir"
    compile_start = time.perf_counter()
    compiled = compile_mlmodel(model_path, compiled_path)
    compile_sec = float(time.perf_counter() - compile_start)
    return compiled, compile_sec, True, "compile_mlmodel"


def _resolve_model_input_seq_len(model_path: Path, *, input_name: str = "input_ids") -> int | None:
    try:
        model = ct.models.MLModel(str(model_path), skip_model_load=True)
        spec = model.get_spec()
    except Exception:
        return None

    for input_spec in spec.description.input:
        if str(input_spec.name) != str(input_name):
            continue
        if input_spec.type.WhichOneof("Type") != "multiArrayType":
            return None
        shape = [int(v) for v in input_spec.type.multiArrayType.shape]
        if len(shape) < 1:
            return None
        return int(shape[-1])
    return None


_STATEFUL_GRAPH_OPS = {"read_state", "write_state", "state_update_masked"}


def _load_graph_ops(path: Path) -> set[str]:
    payload = _load_json(path)
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return set()
    ops: set[str] = set()
    for node in nodes:
        if isinstance(node, dict):
            op = str(node.get("op", "")).strip()
            if op:
                ops.add(op)
    return ops


def _inspect_stateful_ops(
    run_dir: Path,
    report: dict[str, Any],
    *,
    function_names: list[str],
) -> dict[str, Any]:
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    functions = artifacts.get("functions", {})
    if not isinstance(functions, dict):
        functions = {}

    function_graph_paths: dict[str, Path] = {}
    for function_name in function_names:
        fn_artifacts = functions.get(function_name)
        if isinstance(fn_artifacts, dict):
            graph_json = fn_artifacts.get("graph_json")
            if isinstance(graph_json, str) and graph_json.strip():
                function_graph_paths[str(function_name)] = Path(graph_json)

    if not function_graph_paths:
        graph_json = artifacts.get("graph_json")
        if isinstance(graph_json, str) and graph_json.strip():
            function_graph_paths["main"] = Path(graph_json)

    stateful_ops_by_function: dict[str, list[str]] = {}
    all_stateful_ops: set[str] = set()
    for function_name, graph_path in function_graph_paths.items():
        candidate = graph_path
        if not candidate.is_absolute():
            candidate = (run_dir / candidate).resolve()
        try:
            ops = _load_graph_ops(candidate)
        except Exception:
            ops = set()
        stateful_ops = sorted(op for op in ops if op in _STATEFUL_GRAPH_OPS)
        stateful_ops_by_function[str(function_name)] = stateful_ops
        all_stateful_ops.update(stateful_ops)

    return {
        "stateful_ops_by_function": stateful_ops_by_function,
        "stateful_ops_found": sorted(all_stateful_ops),
        "has_stateful_ops": bool(all_stateful_ops),
        "available_functions": sorted(str(name) for name in functions.keys()),
    }


def _normalize_model_output(value: Any) -> np.ndarray:
    def _to_numpy(x: Any) -> np.ndarray:
        try:
            return np.asarray(x)
        except Exception:
            if hasattr(x, "astype"):
                try:
                    import mlx.core as mx  # noqa: PLC0415

                    return np.asarray(x.astype(mx.float32))
                except Exception:
                    return np.asarray(x.astype(np.float32))
            raise

    if isinstance(value, dict):
        if not value:
            raise ValueError("Model produced empty dict outputs.")
        return _to_numpy(next(iter(value.values())))
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Model produced empty sequence outputs.")
        return _to_numpy(value[0])
    return _to_numpy(value)


def _pick_output_tensor(
    predicted_raw: Any,
) -> np.ndarray:
    if isinstance(predicted_raw, dict):
        if not predicted_raw:
            raise RuntimeError("Core ML model returned no outputs.")
        return _normalize_model_output(next(iter(predicted_raw.values())))
    return _normalize_model_output(predicted_raw)


def _next_token_from_model_output(
    predicted_raw: Any,
    *,
    valid_len: int,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    logits = _pick_output_tensor(predicted_raw)
    return _next_token_from_logits(
        logits,
        position=int(valid_len - 1),
        temperature=float(temperature),
        top_k=int(top_k),
        rng=rng,
    )


def _safe_tokens_per_sec(token_count: int, elapsed_sec: float) -> float | None:
    if int(token_count) <= 0:
        return None
    seconds = float(elapsed_sec)
    if not np.isfinite(seconds) or seconds <= 0.0:
        return None
    return float(int(token_count) / seconds)


def _build_flex_input_lens(prompt_len: int, max_new_tokens: int, seq_len: int) -> list[int]:
    if int(seq_len) <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    base = min(int(prompt_len), int(seq_len))
    lengths = {1, base}
    for step in range(1, int(max_new_tokens) + 1):
        lengths.add(min(int(seq_len), base + step))
    return sorted(lengths)


def _next_token_from_logits(
    logits: np.ndarray,
    *,
    position: int | None,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    arr = np.asarray(logits)
    if arr.ndim < 1:
        raise ValueError("Logits must be rank >= 1.")

    if arr.ndim == 1:
        rows = arr.reshape(1, -1)
    elif arr.ndim == 2:
        rows = arr
    else:
        # Common case: [batch, seq, vocab]. Fallback flattens any extra dims into row axis.
        leading = arr.shape[:-1]
        if int(leading[0]) == 1:
            rows = arr.reshape(-1, arr.shape[-1])
        else:
            rows = arr.reshape(-1, arr.shape[-1])

    row_count = int(rows.shape[0])
    if row_count <= 0:
        raise ValueError("Logits tensor has empty row dimension.")
    if row_count == 1:
        vocab = rows[0].reshape(-1)
    else:
        pos = row_count - 1 if position is None else int(position)
        if pos < 0:
            pos += row_count
        if pos < 0 or pos >= row_count:
            raise ValueError(
                f"Requested logits position {position} out of range for sequence length {row_count}."
            )
        vocab = rows[pos].reshape(-1)
    if vocab.size == 0:
        raise ValueError("Logits tensor has empty vocab dimension.")

    temp = float(temperature)
    if temp <= 0.0:
        return int(np.argmax(vocab))

    scores = vocab.astype(np.float64, copy=True) / temp
    k = int(top_k)
    if k > 0 and k < scores.size:
        keep_indices = np.argpartition(scores, -k)[-k:]
        mask = np.full(scores.shape, -np.inf, dtype=np.float64)
        mask[keep_indices] = scores[keep_indices]
        scores = mask

    scores = scores - float(np.max(scores))
    probs = np.exp(scores)
    probs_sum = float(np.sum(probs))
    if probs_sum <= 0.0 or not np.isfinite(probs_sum):
        return int(np.argmax(vocab))
    probs = probs / probs_sum
    return int(rng.choice(np.arange(probs.size), p=probs))


def _build_fixed_input_ids(
    sequence: list[int],
    *,
    seq_len: int,
    pad_token_id: int,
) -> tuple[np.ndarray, int]:
    window = [int(v) for v in sequence[-int(seq_len) :]]
    valid_len = len(window)
    if valid_len == 0:
        raise ValueError("Sequence window is empty.")
    batch = np.full((1, int(seq_len)), int(pad_token_id), dtype=np.int32)
    batch[0, :valid_len] = np.asarray(window, dtype=np.int32)
    return batch, int(valid_len)


def _run_conversion(args: argparse.Namespace, *, run_dir: Path, prompt_len: int) -> dict[str, Any]:
    flex_lens = _build_flex_input_lens(prompt_len, int(args.max_new_tokens), int(args.seq_len))
    cmd = [
        sys.executable,
        "-m",
        "mlx2coreml.convert_mlx_lm",
        "--model-id",
        str(args.model_id),
        "--prompt",
        str(args.prompt),
        "--seq-len",
        str(int(args.seq_len)),
        "--output",
        str(run_dir),
        "--deployment-target",
        str(args.deployment_target),
        "--target-profile",
        str(args.target_profile),
        "--compute-precision",
        str(args.compute_precision),
        "--compute-units",
        str(args.compute_units),
        "--capture-mode",
        str(args.capture_mode),
        "--flex-input-lens",
        ",".join(str(v) for v in flex_lens),
        "--analyze-placement" if bool(args.analyze_placement) else "--no-analyze-placement",
        "--allow-unknown-sources" if bool(args.allow_unknown_sources) else "--no-allow-unknown-sources",
        "--capture-is-training" if bool(args.capture_is_training) else "--no-capture-is-training",
    ]
    if args.state_specs_json is not None:
        cmd.extend(["--state-specs-json", str(args.state_specs_json.resolve())])
    if args.revision is not None:
        cmd.extend(["--revision", str(args.revision)])
    if bool(args.lazy_load):
        cmd.append("--lazy-load")

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Conversion report not found: {report_path}")
    return _load_json(report_path)


def main() -> None:
    args = parse_args()
    run_dir = (args.artifacts_dir / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = run_dir / "sample_report.json"
    report_md_path = run_dir / "sample_report.md"

    from mlx_lm import load
    import mlx.core as mx

    mlx_model_load_start = time.perf_counter()
    model, tokenizer = load(args.model_id, lazy=args.lazy_load, revision=args.revision)
    if hasattr(model, "eval"):
        model.eval()
    mlx_model_load_sec = float(time.perf_counter() - mlx_model_load_start)
    print(f"MLX model load sec: {mlx_model_load_sec:.4f}")

    prompt_tokens = [int(v) for v in tokenizer.encode(args.prompt)]
    if not prompt_tokens:
        raise ValueError("Tokenizer produced an empty prompt.")

    if len(prompt_tokens) > int(args.seq_len):
        prompt_tokens = prompt_tokens[-int(args.seq_len) :]

    conversion_report_path = run_dir / "report.json"
    should_reconvert = bool(args.force_reconvert) or not conversion_report_path.exists()
    if should_reconvert:
        conversion_report = _run_conversion(args, run_dir=run_dir, prompt_len=len(prompt_tokens))
    else:
        conversion_report = _load_json(conversion_report_path)

    stateful_inspection = _inspect_stateful_ops(
        run_dir,
        conversion_report,
        function_names=["main"],
    )
    stateful_ops_found = list(stateful_inspection["stateful_ops_found"])
    stateful_ops_by_function = dict(stateful_inspection["stateful_ops_by_function"])
    available_functions = list(stateful_inspection["available_functions"])
    has_stateful_ops = bool(stateful_inspection["has_stateful_ops"])

    coreml_prepare_start = time.perf_counter()
    model_path = _resolve_model_path(run_dir, conversion_report)
    model_input_seq_len = _resolve_model_input_seq_len(model_path, input_name="input_ids")
    runtime_seq_len = int(args.seq_len)
    coreml_resolve_start = time.perf_counter()
    (
        compiled_path,
        coreml_compile_artifact_sec,
        coreml_compiled_artifact_created,
        coreml_compiled_artifact_source,
    ) = _resolve_compiled_path(run_dir, conversion_report, model_path)
    coreml_resolve_total_sec = float(time.perf_counter() - coreml_resolve_start)
    coreml_artifact_resolve_sec = float(
        max(0.0, coreml_resolve_total_sec - coreml_compile_artifact_sec)
    )

    print(f"Using compiled model artifact: {compiled_path}")

    if model_input_seq_len is not None and int(model_input_seq_len) > 0 and int(model_input_seq_len) != int(
        runtime_seq_len
    ):
        runtime_seq_len = int(model_input_seq_len)
        print(
            "Warning: adjusted runtime sequence length to match model input_ids shape: "
            f"{runtime_seq_len}."
        )

    coreml_model_init_start = time.perf_counter()
    predictor = ct.models.CompiledMLModel(str(compiled_path))
    coreml_compiled_model_init_sec = float(time.perf_counter() - coreml_model_init_start)
    decode_load_strategy = "main_compiled"

    if has_stateful_ops:
        print(
            "Warning: converted main graph contains stateful ops "
            f"({', '.join(sorted(_STATEFUL_GRAPH_OPS))}); this sample runner uses single-shot "
            "predict() calls without explicit Core ML state objects."
        )

    coreml_total_prepare_sec = float(time.perf_counter() - coreml_prepare_start)
    coreml_model_load_sec = coreml_total_prepare_sec
    print(
        f"Core ML artifact resolve sec: {coreml_artifact_resolve_sec:.4f} "
        f"(source={coreml_compiled_artifact_source}, created={coreml_compiled_artifact_created})"
    )
    print(f"Core ML artifact compile sec: {coreml_compile_artifact_sec:.4f}")
    print(f"Core ML compiled model init sec: {coreml_compiled_model_init_sec:.4f}")
    print(f"Core ML model load sec: {coreml_model_load_sec:.4f}")

    rng = np.random.default_rng(0)
    generated_tokens: list[int] = []
    validation: list[dict[str, Any]] = []
    sequence = list(prompt_tokens)
    validate_steps = 0 if bool(args.skip_validation) else max(0, int(args.validate_steps))
    eos_token_id = int(getattr(tokenizer, "eos_token_id", prompt_tokens[-1]))
    generation_start = time.perf_counter()
    decode_elapsed_sec = 0.0
    infer_call_count = 0
    time_to_first_token_sec: float | None = None

    for step in range(int(args.max_new_tokens)):
        decode_step_start = time.perf_counter()
        input_ids, valid_len = _build_fixed_input_ids(
            sequence,
            seq_len=int(runtime_seq_len),
            pad_token_id=eos_token_id,
        )
        predicted_raw = predictor.predict({"input_ids": input_ids})
        next_token_coreml = _next_token_from_model_output(
            predicted_raw,
            valid_len=int(valid_len),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            rng=rng,
        )
        decode_step_elapsed = float(time.perf_counter() - decode_step_start)
        decode_elapsed_sec += decode_step_elapsed
        infer_call_count += 1
        if time_to_first_token_sec is None:
            time_to_first_token_sec = float(time.perf_counter() - generation_start)
            print(f"Time to first token sec: {time_to_first_token_sec:.4f}")

        if step < validate_steps:
            mlx_out = model(mx.array(input_ids, dtype=mx.int32))
            logits_mlx = _normalize_model_output(mlx_out)
            next_token_mlx = _next_token_from_logits(
                logits_mlx,
                position=int(valid_len - 1),
                temperature=0.0,
                top_k=0,
                rng=rng,
            )
            validation.append(
                {
                    "step": int(step),
                    "coreml_token": int(next_token_coreml),
                    "mlx_token": int(next_token_mlx),
                    "match": bool(int(next_token_coreml) == int(next_token_mlx)),
                }
            )

        sequence.append(int(next_token_coreml))
        generated_tokens.append(int(next_token_coreml))

    generation_elapsed_sec = float(time.perf_counter() - generation_start)
    total_tokens_per_sec = _safe_tokens_per_sec(len(generated_tokens), generation_elapsed_sec)
    decode_tokens_per_sec = _safe_tokens_per_sec(len(generated_tokens), decode_elapsed_sec)
    infer_decode_sec = decode_elapsed_sec
    infer_tokens_per_sec = decode_tokens_per_sec

    generated_text = tokenizer.decode(sequence)
    prompt_text = tokenizer.decode(prompt_tokens)

    mismatches = [item for item in validation if not bool(item["match"])]
    status = "ok" if not mismatches else "failed"
    error = None if status == "ok" else f"{len(mismatches)} validation step(s) mismatched MLX."

    run_context = build_run_context(
        run_kind="convert_and_sample_qwen3_0_6b",
        deployment_target=str(args.deployment_target),
        convert_to=str(conversion_report.get("run_context", {}).get("convert_to", "mlprogram")),
        seed=0,
    )
    run_context["model_id"] = args.model_id
    run_context["run_dir"] = str(run_dir)
    run_context["seq_len"] = int(args.seq_len)
    run_context["runtime_input_seq_len"] = int(runtime_seq_len)
    run_context["model_input_seq_len"] = int(model_input_seq_len) if model_input_seq_len is not None else None
    run_context["max_new_tokens"] = int(args.max_new_tokens)
    run_context["validate_steps"] = int(validate_steps)
    run_context["temperature"] = float(args.temperature)
    run_context["top_k"] = int(args.top_k)
    run_context["decode_mode"] = "full"
    run_context["decode_load_strategy"] = decode_load_strategy
    run_context["decode_supports_explicit_state"] = False
    run_context["conversion_state_specs_count"] = len(conversion_report.get("state_specs", []))
    run_context["state_specs_json"] = str(args.state_specs_json.resolve()) if args.state_specs_json else None
    run_context["stateful_ops_found"] = list(stateful_ops_found)
    run_context["has_stateful_ops"] = bool(has_stateful_ops)

    performance = {
        "mlx_model_load_sec": mlx_model_load_sec,
        "coreml_artifact_resolve_sec": coreml_artifact_resolve_sec,
        "coreml_compile_artifact_sec": coreml_compile_artifact_sec,
        "coreml_compiled_model_init_sec": coreml_compiled_model_init_sec,
        "coreml_total_prepare_sec": coreml_total_prepare_sec,
        "coreml_model_load_sec": coreml_model_load_sec,
        "coreml_compiled_artifact_source": coreml_compiled_artifact_source,
        "coreml_compiled_artifact_created": coreml_compiled_artifact_created,
        "decode_mode": "full",
        "decode_load_strategy": decode_load_strategy,
        "model_input_seq_len": int(model_input_seq_len) if model_input_seq_len is not None else None,
        "runtime_input_seq_len": int(runtime_seq_len),
        "infer_call_count": infer_call_count,
        "infer_decode_sec": infer_decode_sec,
        "infer_tokens_per_sec": infer_tokens_per_sec,
        "stateful_ops_found": list(stateful_ops_found),
        "has_stateful_ops": bool(has_stateful_ops),
        "time_to_first_token_sec": time_to_first_token_sec,
        "generation_elapsed_sec": generation_elapsed_sec,
        "decode_elapsed_sec": decode_elapsed_sec,
        "generated_token_count": len(generated_tokens),
        "total_tokens_per_sec": total_tokens_per_sec,
        "decode_tokens_per_sec": decode_tokens_per_sec,
    }

    decode_details = {
        "mode": "full",
        "load_strategy": decode_load_strategy,
        "model_input_seq_len": int(model_input_seq_len) if model_input_seq_len is not None else None,
        "runtime_input_seq_len": int(runtime_seq_len),
        "stateful_ops_found": list(stateful_ops_found),
        "stateful_ops_by_function": stateful_ops_by_function,
        "available_functions": available_functions,
        "has_stateful_ops": bool(has_stateful_ops),
    }

    sample_report = {
        "schema_version": run_context["schema_version"],
        "run_kind": "convert_and_sample_qwen3_0_6b",
        "run_context": run_context,
        "status": status,
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "prompt_text": prompt_text,
        "generated_text": generated_text,
        "validation_steps": validation,
        "validation_mismatch_count": len(mismatches),
        "model_artifact": str(model_path),
        "compiled_artifact": str(compiled_path),
        "conversion_report": str(conversion_report_path.resolve()),
        "decode": decode_details,
        "performance": performance,
        "artifacts": {
            "json": str(report_json_path.resolve()),
            "markdown": str(report_md_path.resolve()),
        },
        "error": error,
    }
    write_json(report_json_path, sample_report)

    md_lines = [
        "# Qwen3 Convert + Sample Report",
        "",
        f"- Status: `{sample_report['status']}`",
        f"- Model: `{args.model_id}`",
        f"- Run dir: `{run_dir}`",
        f"- Prompt: `{args.prompt}`",
        f"- Generated tokens: `{generated_tokens}`",
        f"- Validation mismatches: `{len(mismatches)}`",
        "- Decode mode: `full`",
        f"- Decode load strategy: `{decode_load_strategy}`",
        (
            f"- Model input seq len: `{model_input_seq_len}`"
            if model_input_seq_len is not None
            else "- Model input seq len: `unknown`"
        ),
        f"- Runtime input seq len: `{runtime_seq_len}`",
        f"- Has stateful ops: `{has_stateful_ops}`",
        f"- Stateful ops found: `{', '.join(stateful_ops_found)}`" if stateful_ops_found else "- Stateful ops found: `none`",
        f"- MLX model load (sec): `{mlx_model_load_sec:.4f}`",
        f"- Core ML artifact resolve (sec): `{coreml_artifact_resolve_sec:.4f}`",
        f"- Core ML artifact compile (sec): `{coreml_compile_artifact_sec:.4f}`",
        f"- Core ML compiled model init (sec): `{coreml_compiled_model_init_sec:.4f}`",
        f"- Core ML total prepare (sec): `{coreml_total_prepare_sec:.4f}`",
        f"- Core ML model load (sec): `{coreml_model_load_sec:.4f}`",
        f"- Infer decode (sec): `{infer_decode_sec:.4f}`",
        f"- Infer tokens/sec: `{infer_tokens_per_sec:.4f}`" if infer_tokens_per_sec is not None else "- Infer tokens/sec: `n/a`",
        f"- Time to first token (sec): `{time_to_first_token_sec:.4f}`" if time_to_first_token_sec is not None else "- Time to first token (sec): `n/a`",
        f"- Total tokens/sec: `{total_tokens_per_sec:.4f}`" if total_tokens_per_sec is not None else "- Total tokens/sec: `n/a`",
        f"- Model artifact: `{model_path}`",
        f"- Compiled artifact: `{compiled_path}`",
        f"- JSON report: `{report_json_path}`",
    ]
    if error is not None:
        md_lines.append(f"- Error: `{error}`")
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Run dir: {run_dir}")
    print(f"Status: {status}")
    print("Decode mode: full")
    print(f"Decode load strategy: {decode_load_strategy}")
    if model_input_seq_len is None:
        print("Model input seq len: unknown")
    else:
        print(f"Model input seq len: {model_input_seq_len}")
    print(f"Runtime input seq len: {runtime_seq_len}")
    print(f"Has stateful ops: {has_stateful_ops}")
    print(f"Stateful ops found: {', '.join(stateful_ops_found) if stateful_ops_found else 'none'}")
    print(f"MLX model load sec: {mlx_model_load_sec:.4f}")
    print(
        f"Core ML artifact resolve sec: {coreml_artifact_resolve_sec:.4f} "
        f"(source={coreml_compiled_artifact_source}, created={coreml_compiled_artifact_created})"
    )
    print(f"Core ML artifact compile sec: {coreml_compile_artifact_sec:.4f}")
    print(f"Core ML compiled model init sec: {coreml_compiled_model_init_sec:.4f}")
    print(f"Core ML total prepare sec: {coreml_total_prepare_sec:.4f}")
    print(f"Core ML model load sec: {coreml_model_load_sec:.4f}")
    print(f"Infer decode sec: {infer_decode_sec:.4f}")
    if time_to_first_token_sec is None:
        print("Time to first token sec: n/a")
    else:
        print(f"Time to first token sec: {time_to_first_token_sec:.4f}")
    if infer_tokens_per_sec is None:
        print("Infer tokens/sec: n/a")
    else:
        print(f"Infer tokens/sec: {infer_tokens_per_sec:.4f}")
    if total_tokens_per_sec is None:
        print("Total tokens/sec: n/a")
    else:
        print(f"Total tokens/sec: {total_tokens_per_sec:.4f}")
    print(f"Generated text: {generated_text}")
    print(f"Wrote JSON report: {report_json_path}")
    print(f"Wrote markdown report: {report_md_path}")

    if error is not None:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()
