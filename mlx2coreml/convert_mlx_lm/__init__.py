from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..compilation import compile_mlmodel
from ..compute_plan import analyze_compiled_model_placement
from ..conversion import (
    ConversionConfig,
    build_conversion_inputs as _build_conversion_inputs,
    capture_mlx_graph,
    collect_unsupported_details,
    convert_lowered_program,
    ensure_graph_supported,
    find_extra_input_names,
    load_state_specs as _load_state_specs,
    lower_graph_to_mil,
    normalize_graph_for_conversion as _normalize_graph_for_function,
    parse_flex_input_names,
    parse_flex_lengths,
    summarize_graph_inference,
    temporary_capture_training_mode as _temporary_capture_training_mode,
)
from ..ir import Graph
from ..reporting import (
    build_run_context,
    init_stage_timings,
    summarize_stage_timings,
    timed_stage,
    write_json,
)


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
        "--output",
        type=Path,
        required=True,
        help="Output directory written exactly as provided.",
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
        "--target-profile",
        default="default",
        choices=["default", "ane_ios18", "conservative"],
        help="Lowering profile used to control rewrite aggressiveness.",
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
        "--state-specs-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file defining Core ML state specs, for example: "
            "[{\"name\":\"kv_cache\",\"shape\":[1,32,128,64],\"dtype\":\"fp16\"}]"
        ),
    )
    parser.add_argument(
        "--analyze-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Analyze compiled model placement using Core ML compute-plan APIs and include "
            "fallback counts in reports."
        ),
    )
    parser.add_argument(
        "--flex-input-lens",
        default=None,
        help=(
            "Optional comma-separated token lengths for flexible input shapes "
            "(for example: 1,16,64). Use 'auto' to enable preset [1, seq_len]."
        ),
    )
    parser.add_argument(
        "--flex-input-names",
        default="input_ids,attention_mask,position_ids,token_type_ids",
        help=(
            "Comma-separated input names eligible for flexible sequence length handling "
            "when --flex-input-lens is enabled."
        ),
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
    parser.add_argument(
        "--capture-is-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Temporarily set the MLX model to training mode during graph export capture. "
            "Useful for models whose eval path uses fast custom kernels that are harder to lower."
        ),
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


def _parse_flex_lengths(raw: str | None, *, seq_len: int) -> list[int] | None:
    parsed = parse_flex_lengths(raw, preset_values=[1, int(seq_len)])
    if parsed is None:
        return None
    normalized: list[int] = []
    for value in [1, *parsed, int(seq_len)]:
        if int(value) <= 0:
            raise ValueError(f"--flex-input-lens values must be positive, got {value}.")
        ivalue = int(value)
        if ivalue not in normalized:
            normalized.append(ivalue)
    return normalized


def _parse_flex_input_names(raw: str) -> set[str]:
    return parse_flex_input_names(raw)


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
        f"- Stage durations (sec): `{json.dumps(report['stage_durations_sec'], sort_keys=True)}`",
        f"- Total stage duration (sec): `{report['total_duration_sec']}`",
        f"- Ops clean: `{report['ops_clean']}`",
        f"- Weights captured as constants: `{report['weights_captured_as_constants']}`",
        f"- Extra non-user inputs: `{report['extra_input_count']}`",
        f"- Unsupported ops: `{report['unsupported_op_count']}`",
        f"- Fallback ops: `{report['fallback_op_count']}`",
        f"- Capture mode: `{report['capture_mode']}`",
        f"- Target profile: `{report['run_context']['target_profile']}`",
        f"- State specs: `{json.dumps(report['state_specs'])}`",
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
    if report.get("flex_input_shapes"):
        lines.append(f"- Flexible input shapes: `{json.dumps(report['flex_input_shapes'], sort_keys=True)}`")
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
    if report["top_fallback_ops"]:
        fallback_ops_str = ", ".join(
            f"`{name}` x{count}" for name, count in report["top_fallback_ops"]
        )
        lines.append(f"- Top fallback ops: {fallback_ops_str}")
    if report.get("placement_error"):
        lines.append(f"- Placement analysis error: `{report['placement_error']}`")
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
    flex_input_lens = _parse_flex_lengths(args.flex_input_lens, seq_len=int(args.seq_len))
    flex_input_names = _parse_flex_input_names(args.flex_input_names)
    state_specs = _load_state_specs(args.state_specs_json)
    conversion_config = ConversionConfig(
        capture_mode=args.capture_mode,
        allow_unknown_sources=bool(args.allow_unknown_sources),
        capture_is_training=bool(args.capture_is_training),
        deployment_target=args.deployment_target,
        target_profile=args.target_profile,
        convert_to=args.convert_to,
        compute_precision=args.compute_precision,
        compute_units=args.compute_units,
        state_specs=state_specs,
        flex_input_lens=flex_input_lens,
        flex_input_names=set(flex_input_names),
    )

    run_dir = args.output
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
    run_context["output"] = str(args.output)
    run_context["capture_mode"] = args.capture_mode
    run_context["target_profile"] = args.target_profile
    run_context["compute_precision"] = args.compute_precision
    run_context["compute_units"] = args.compute_units
    run_context["prompt"] = args.prompt
    run_context["seq_len"] = int(args.seq_len)
    run_context["revision"] = args.revision
    run_context["lazy_load"] = bool(args.lazy_load)
    run_context["allow_unknown_sources"] = bool(args.allow_unknown_sources)
    run_context["capture_is_training"] = bool(args.capture_is_training)
    run_context["analyze_placement"] = bool(args.analyze_placement)
    run_context["flex_input_lens"] = list(flex_input_lens) if flex_input_lens is not None else None
    run_context["flex_input_names"] = sorted(flex_input_names)
    run_context["state_specs_json"] = str(args.state_specs_json) if args.state_specs_json else None
    run_context["state_specs"] = [spec.to_dict() for spec in state_specs] if state_specs else []
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
        "placement_analysis": (
            "pending" if (not skip_compile and bool(args.analyze_placement)) else "skipped"
        ),
    }
    stage_timings = init_stage_timings(stage_status.keys())

    main_function_name = "main"
    op_counts: list[list[Any]] = []
    unsupported_details: list[dict[str, Any]] = []
    extra_input_names: list[str] = []
    flex_input_shapes: dict[str, list[list[int]]] = {}
    inference_summary: dict[str, int] | None = None
    normalized_graph: Graph | None = None
    normalized_inputs: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    function_artifacts: dict[str, dict[str, str]] = {}
    saved_model_path: Path | None = None
    saved_compiled_path: Path | None = None
    placement_analysis: dict[str, Any] | None = None
    placement_error: str | None = None
    error_message: str | None = None
    program = None
    converted = None

    try:
        with timed_stage(stage_timings, "load_model"):
            from mlx_lm import load

            model, tokenizer = load(args.model_id, lazy=args.lazy_load, revision=args.revision)
            if hasattr(model, "eval"):
                model.eval()
        stage_status["load_model"] = "ok"

        with timed_stage(stage_timings, "tokenize"):
            input_ids = _tokenize_prompt(tokenizer, args.prompt, args.seq_len)
            inputs = {"input_ids": input_ids}
        stage_status["tokenize"] = "ok"

        with timed_stage(stage_timings, "capture"):
            captured = capture_mlx_graph(
                model,
                inputs,
                dot_output_path=dot_path,
                capture_mode=conversion_config.capture_mode,
                allow_unknown_sources=conversion_config.allow_unknown_sources,
                capture_is_training=conversion_config.capture_is_training,
                capture_function=lambda input_ids: _select_primary_output(model(input_ids)),
            )
            graph = captured.graph
            normalized_inputs = captured.normalized_inputs
            expected = captured.expected_outputs
        stage_status["capture"] = "ok"

        function_artifacts["main"] = {
            "capture_dot": str(dot_path),
            "graph_json": str(graph_json_path),
            "inputs": str(inputs_path),
            "expected": str(expected_path),
        }
        graph_json_path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n", encoding="utf-8")
        np.savez(inputs_path, **normalized_inputs)

        with timed_stage(stage_timings, "normalize"):
            normalized_graph, expected, op_counts = _normalize_graph_for_function(
                graph,
                expected,
            )
        stage_status["normalize"] = "ok"
        np.savez(expected_path, **expected)

        with timed_stage(stage_timings, "type_infer"):
            assert normalized_graph is not None
            inference_summary = summarize_graph_inference(normalized_graph)
        stage_status["type_infer"] = "ok"

        with timed_stage(stage_timings, "support_check"):
            assert normalized_graph is not None
            unsupported_details = collect_unsupported_details(normalized_graph)
            ensure_graph_supported(normalized_graph)
        stage_status["support_check"] = "ok"

        extra_input_names = find_extra_input_names(normalized_graph, normalized_inputs)

        if not skip_lower:
            with timed_stage(stage_timings, "lower"):
                assert normalized_graph is not None
                program = lower_graph_to_mil(
                    normalized_graph,
                    config=conversion_config,
                )
                mil_path.write_text(str(program) + "\n", encoding="utf-8")
            stage_status["lower"] = "ok"

        if not skip_convert:
            if program is None:
                raise RuntimeError("Internal error: program missing before convert stage.")
            assert normalized_graph is not None
            with timed_stage(stage_timings, "convert"):
                converted, _, flex_input_shapes = convert_lowered_program(
                    program,
                    normalized_graph.inputs,
                    config=conversion_config,
                )
                converted.save(str(model_path))
            function_artifacts["main"]["model"] = str(model_path)
            saved_model_path = model_path
            stage_status["convert"] = "ok"

        if not skip_compile:
            if converted is None and saved_model_path is None:
                raise RuntimeError("Internal error: converted model missing before compile stage.")
            with timed_stage(stage_timings, "compile"):
                saved_compiled_path = compile_mlmodel(model_path, compiled_path)
            stage_status["compile"] = "ok"
            if stage_status["placement_analysis"] == "pending":
                try:
                    with timed_stage(stage_timings, "placement_analysis"):
                        placement_analysis = analyze_compiled_model_placement(
                            saved_compiled_path,
                            compute_units=args.compute_units,
                        )
                    stage_status["placement_analysis"] = "ok"
                except Exception as exc:  # pragma: no cover - environment-dependent APIs
                    placement_error = str(exc)
                    stage_status["placement_analysis"] = "failed"

    except Exception as exc:
        error_message = str(exc)
        pending_stage = next((k for k, v in stage_status.items() if v == "pending"), None)
        if pending_stage is not None:
            stage_status[pending_stage] = "failed"

    stage_durations_sec, total_duration_sec = summarize_stage_timings(stage_timings)
    unsupported_count = len(unsupported_details)
    weights_captured_as_constants = len(extra_input_names) == 0
    inference_by_function = {main_function_name: inference_summary} if inference_summary is not None else {}
    unsupported_by_function = {main_function_name: unsupported_details}
    flex_input_shapes_by_function = (
        {main_function_name: flex_input_shapes} if flex_input_shapes else {}
    )
    report_json = {
        "schema_version": run_context["schema_version"],
        "run_kind": "mlx_lm_weighted_probe",
        "run_context": run_context,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "seq_len": int(args.seq_len),
        "capture_mode": args.capture_mode,
        "main_function": main_function_name,
        "inference_by_function": inference_by_function,
        "unsupported_by_function": unsupported_by_function,
        "state_specs": [spec.to_dict() for spec in state_specs] if state_specs else [],
        "status": "ok" if error_message is None else "failed",
        "stage_status": stage_status,
        "stage_durations_sec": stage_durations_sec,
        "total_duration_sec": total_duration_sec,
        "ops_clean": bool(unsupported_count == 0 and weights_captured_as_constants),
        "weights_captured_as_constants": bool(weights_captured_as_constants),
        "extra_input_count": len(extra_input_names),
        "extra_input_names_sample": extra_input_names[:25],
        "flex_input_lens": list(flex_input_lens) if flex_input_lens is not None else None,
        "flex_input_names": sorted(flex_input_names),
        "flex_input_shapes": flex_input_shapes,
        "flex_input_shapes_by_function": flex_input_shapes_by_function,
        "unsupported_op_count": unsupported_count,
        "unsupported_ops": unsupported_details,
        "fallback_op_count": (
            int(placement_analysis["fallback_operation_count"]) if placement_analysis is not None else None
        ),
        "top_fallback_ops": (
            placement_analysis.get("top_fallback_ops", []) if placement_analysis is not None else []
        ),
        "placement_analysis": placement_analysis,
        "placement_error": placement_error,
        "top_ops": op_counts,
        "inference": inference_summary,
        "artifacts": {
            "capture_dot": str(dot_path),
            "graph_json": str(graph_json_path),
            "inputs": str(inputs_path),
            "expected": str(expected_path),
            "functions": function_artifacts,
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
    print(f"Output dir: {run_dir}")
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
__all__ = [
    "_build_conversion_inputs",
    "_load_state_specs",
    "_normalize_graph_for_function",
    "_parse_flex_input_names",
    "_parse_flex_lengths",
    "_temporary_capture_training_mode",
    "main",
    "parse_args",
]
