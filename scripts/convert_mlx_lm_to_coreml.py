import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.from_mlx import capture_graph_from_mlx_function
from mlx2coreml.lower_to_mil import (
    build_mil_program,
    compile_model_artifact,
    convert_program_to_model,
)
from mlx2coreml.op_registry import ensure_supported, unsupported_op_details
from mlx2coreml.passes import infer_graph_specs, normalize_graph, summarize_inference
from mlx2coreml.reporting import build_run_context, write_json


def parse_deployment_target(target_name: str):
    if not hasattr(ct.target, target_name):
        valid = [name for name in dir(ct.target) if name.startswith("iOS")]
        raise ValueError(
            f"Unknown deployment target: {target_name}. Valid examples: {', '.join(valid)}"
        )
    return getattr(ct.target, target_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an mlx-lm text model to Core ML via MLX capture -> IR normalize -> "
            "MIL lower -> Core ML convert -> optional compile."
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id loadable with mlx_lm.load().",
    )
    parser.add_argument(
        "--prompt",
        default="hello",
        help="Prompt text used to build the input_ids capture batch.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Token sequence length used for input_ids (truncate/pad).",
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
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Base artifacts directory.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to sanitized model id.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=["callback", "dot"],
        default="callback",
        help="Capture backend. 'callback' preserves primitive attrs; 'dot' is legacy fallback.",
    )
    parser.add_argument(
        "--deployment-target",
        default="iOS18",
        help="Core ML minimum deployment target (for example: iOS18).",
    )
    parser.add_argument(
        "--convert-to",
        default="mlprogram",
        choices=["mlprogram", "neuralnetwork"],
        help="Core ML conversion backend.",
    )
    parser.add_argument(
        "--compute-precision",
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="Core ML conversion compute precision.",
    )
    parser.add_argument(
        "--compute-units",
        default="all",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        help="Core ML conversion compute units.",
    )
    parser.add_argument(
        "--skip-lower",
        action="store_true",
        help="Skip MIL lowering (implies --skip-convert and --skip-compile).",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip Core ML conversion (implies --skip-compile).",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip model compilation to .mlmodelc.",
    )
    parser.add_argument(
        "--allow-unknown-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow parser to keep source nodes without explicit input specs. "
            "Enabled by default for robust model capture."
        ),
    )
    return parser.parse_args()


def _sanitize_run_name(model_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "mlx_lm_convert"


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


def _select_primary_output(value: Any) -> Any:
    if isinstance(value, dict):
        if not value:
            raise ValueError("Model returned an empty dict output.")
        return next(iter(value.values()))
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Model returned an empty sequence output.")
        return value[0]
    return value


def _top_ops(graph_ops: list[str]) -> list[list[Any]]:
    ranked = sorted(Counter(graph_ops).items(), key=lambda item: (-item[1], item[0]))
    return [[name, int(count)] for name, count in ranked]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    top_ops_str = ", ".join(f"`{name}` x{count}" for name, count in report["top_ops"])
    compiled = report["artifacts"]["compiled"]
    lines = [
        "# Weighted Model Probe Report",
        "",
        f"- Model: `{report['model_id']}`",
        f"- Prompt: `{report['prompt']}`",
        f"- Seq length: `{report['seq_len']}`",
        f"- Status: `{report['status']}`",
        f"- Stage status: `{json.dumps(report['stage_status'], sort_keys=True)}`",
        f"- Ops clean: `{report['ops_clean']}`",
        f"- Weights captured as constants: `{report['weights_captured_as_constants']}`",
        f"- Extra non-user inputs: `{report['extra_input_count']}`",
        f"- Unsupported ops: `{report['unsupported_op_count']}`",
        f"- Capture mode: `{report['capture_mode']}`",
        f"- Deployment target: `{report['run_context']['deployment_target']}`",
        f"- Conversion backend: `{report['run_context']['convert_to']}`",
        f"- Compute precision: `{report['run_context']['compute_precision']}`",
        f"- Compute units: `{report['run_context']['compute_units']}`",
        "- Versions:",
        f"  - python: `{report['run_context']['versions']['python']}`",
        f"  - coremltools: `{report['run_context']['versions']['coremltools']}`",
        f"  - numpy: `{report['run_context']['versions']['numpy']}`",
        f"  - mlx: `{report['run_context']['versions']['mlx']}`",
        f"- Graph JSON: `{report['artifacts']['graph_json']}`",
        f"- Capture DOT: `{report['artifacts']['capture_dot']}`",
        f"- Inputs NPZ: `{report['artifacts']['inputs']}`",
        f"- Expected outputs NPZ: `{report['artifacts']['expected']}`",
    ]
    if report["artifacts"]["mil_program"]:
        lines.append(f"- MIL Program: `{report['artifacts']['mil_program']}`")
    if report["artifacts"]["model"]:
        lines.append(f"- Model artifact: `{report['artifacts']['model']}`")
    if compiled:
        lines.append(f"- Compiled artifact: `{compiled}`")
    if report["extra_input_names_sample"]:
        lines.append(
            f"- Extra input names sample: `{', '.join(report['extra_input_names_sample'])}`"
        )
    if report.get("inference") is not None:
        lines.append(
            "- Inference coverage: "
            f"{report['inference']['with_shape']}/{report['inference']['total_tensors']} with shape, "
            f"{report['inference']['with_dtype']}/{report['inference']['total_tensors']} with dtype"
        )
    if top_ops_str:
        lines.append(f"- Top ops: {top_ops_str}")
    if report["unsupported_ops"]:
        lines.append("- Unsupported op details:")
        for detail in report["unsupported_ops"]:
            primitive = detail.get("primitive")
            primitive_text = f", primitive={primitive}" if primitive else ""
            lines.append(
                f"  - `{detail['op']}` x{detail.get('count', 1)} "
                f"(status={detail.get('status')}, source={detail.get('source')}{primitive_text})"
            )
    if report.get("error") is not None:
        lines.append(f"- Error: `{report['error']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    skip_lower = bool(args.skip_lower)
    skip_convert = bool(args.skip_convert or skip_lower)
    skip_compile = bool(args.skip_compile or skip_convert)

    run_name = args.run_name or _sanitize_run_name(args.model_id)
    run_dir = args.artifacts_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dot_path = run_dir / "capture_graph.dot"
    graph_json_path = run_dir / "graph.json"
    inputs_path = run_dir / "inputs.npz"
    expected_path = run_dir / "expected_outputs.npz"
    mil_path = run_dir / "program.mil.txt"
    model_path = run_dir / ("model.mlpackage" if args.convert_to == "mlprogram" else "model.mlmodel")
    compiled_path = run_dir / "model.mlmodelc"
    run_context_path = run_dir / "run_context.json"
    report_json_path = run_dir / "report.json"
    report_md_path = run_dir / "report.md"

    run_context = build_run_context(
        run_kind="mlx_lm_weighted_probe",
        deployment_target=args.deployment_target,
        convert_to=args.convert_to,
        seed=0,
    )
    run_context["model_id"] = args.model_id
    run_context["capture_mode"] = args.capture_mode
    run_context["compute_precision"] = args.compute_precision
    run_context["compute_units"] = args.compute_units
    run_context["prompt"] = args.prompt
    run_context["seq_len"] = int(args.seq_len)
    run_context["revision"] = args.revision
    run_context["lazy_load"] = bool(args.lazy_load)
    run_context["allow_unknown_sources"] = bool(args.allow_unknown_sources)
    write_json(run_context_path, run_context)

    stage_status = {
        "load_model": "pending",
        "tokenize": "pending",
        "capture": "pending",
        "normalize": "pending",
        "type_infer": "pending",
        "support_check": "pending",
        "lower": "skipped" if skip_lower else "pending",
        "convert": "skipped" if skip_convert else "pending",
        "compile": "skipped" if skip_compile else "pending",
    }

    graph_ops: list[str] = []
    op_counts: list[list[Any]] = []
    unsupported_details: list[dict[str, Any]] = []
    extra_input_names: list[str] = []
    inference_summary: dict[str, int] | None = None
    saved_model_path: Path | None = None
    saved_compiled_path: Path | None = None
    error_message: str | None = None

    try:
        from mlx_lm import load

        model, tokenizer = load(args.model_id, lazy=args.lazy_load, revision=args.revision)
        if hasattr(model, "eval"):
            model.eval()
        stage_status["load_model"] = "ok"

        input_ids = _tokenize_prompt(tokenizer, args.prompt, args.seq_len)
        inputs = {"input_ids": input_ids}
        stage_status["tokenize"] = "ok"

        graph, normalized_inputs, expected = capture_graph_from_mlx_function(
            dot_output_path=dot_path,
            inputs=inputs,
            function=lambda input_ids: _select_primary_output(model(input_ids)),
            allow_unknown_sources=args.allow_unknown_sources,
            capture_mode=args.capture_mode,
        )
        stage_status["capture"] = "ok"

        graph_json_path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n", encoding="utf-8")
        np.savez(inputs_path, **normalized_inputs)
        np.savez(expected_path, **expected)

        normalized_graph = normalize_graph(graph)
        graph_ops = [node.op for node in normalized_graph.nodes]
        op_counts = _top_ops(graph_ops)
        stage_status["normalize"] = "ok"

        inferred = infer_graph_specs(normalized_graph)
        inference_summary = summarize_inference(inferred)
        stage_status["type_infer"] = "ok"

        unsupported_details = unsupported_op_details(normalized_graph)
        ensure_supported(normalized_graph)
        stage_status["support_check"] = "ok"

        input_names = set(normalized_inputs.keys())
        extra_input_names = [spec.name for spec in normalized_graph.inputs if spec.name not in input_names]

        program = None
        if not skip_lower:
            target = parse_deployment_target(args.deployment_target)
            program = build_mil_program(normalized_graph, deployment_target=target, normalize=False)
            mil_path.write_text(str(program) + "\n", encoding="utf-8")
            stage_status["lower"] = "ok"

        converted = None
        if not skip_convert:
            if program is None:
                raise RuntimeError("Internal error: program missing before convert stage.")
            target = parse_deployment_target(args.deployment_target)
            converted = convert_program_to_model(
                program,
                deployment_target=target,
                convert_to=args.convert_to,
                compute_precision=args.compute_precision,
                compute_units=args.compute_units,
            )
            converted.save(str(model_path))
            saved_model_path = model_path
            stage_status["convert"] = "ok"

        if not skip_compile:
            if converted is None and saved_model_path is None:
                raise RuntimeError("Internal error: converted model missing before compile stage.")
            saved_compiled_path = compile_model_artifact(model_path, compiled_path)
            stage_status["compile"] = "ok"

    except Exception as exc:
        error_message = str(exc)
        pending_stage = next((k for k, v in stage_status.items() if v == "pending"), None)
        if pending_stage is not None:
            stage_status[pending_stage] = "failed"

    unsupported_count = len(unsupported_details)
    weights_captured_as_constants = len(extra_input_names) == 0
    report_json = {
        "schema_version": run_context["schema_version"],
        "run_kind": "mlx_lm_weighted_probe",
        "run_context": run_context,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "seq_len": int(args.seq_len),
        "capture_mode": args.capture_mode,
        "status": "ok" if error_message is None else "failed",
        "stage_status": stage_status,
        "ops_clean": bool(unsupported_count == 0 and weights_captured_as_constants),
        "weights_captured_as_constants": bool(weights_captured_as_constants),
        "extra_input_count": len(extra_input_names),
        "extra_input_names_sample": extra_input_names[:25],
        "unsupported_op_count": unsupported_count,
        "unsupported_ops": unsupported_details,
        "top_ops": op_counts,
        "inference": inference_summary,
        "artifacts": {
            "capture_dot": str(dot_path),
            "graph_json": str(graph_json_path),
            "inputs": str(inputs_path),
            "expected": str(expected_path),
            "mil_program": str(mil_path) if (mil_path.exists() and not skip_lower) else None,
            "model": str(saved_model_path) if saved_model_path is not None else None,
            "compiled": str(saved_compiled_path) if saved_compiled_path is not None else None,
        },
        "error": error_message,
    }
    write_json(report_json_path, report_json)
    _write_markdown(report_md_path, report_json)

    if error_message is not None:
        raise RuntimeError(error_message)

    print(f"Model: {args.model_id}")
    print(f"Run dir: {run_dir}")
    print(f"Wrote DOT graph: {dot_path}")
    print(f"Wrote graph JSON: {graph_json_path}")
    print(f"Wrote inputs: {inputs_path}")
    print(f"Wrote expected outputs: {expected_path}")
    if not skip_lower:
        print(f"Wrote MIL program: {mil_path}")
    if saved_model_path is not None:
        print(f"Wrote model: {saved_model_path}")
    if saved_compiled_path is not None:
        print(f"Wrote compiled model: {saved_compiled_path}")
    print(f"Wrote report markdown: {report_md_path}")
    print(f"Wrote report json: {report_json_path}")


if __name__ == "__main__":
    main()
