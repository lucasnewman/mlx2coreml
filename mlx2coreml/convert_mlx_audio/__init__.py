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
)
from ..conversion import capture_mlx_graph
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
            "Convert an mlx-audio VAD model to Core ML via MLX capture -> IR normalize -> "
            "MIL lower -> Core ML convert -> optional compile. "
            "The current example path targets Smart Turn-style models that expose "
            "prepare_input_features()."
        )
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id loadable with mlx_audio.vad.load().",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=None,
        help=(
            "Optional audio file used to build sample input_features for capture. "
            "If omitted, the script generates silence."
        ),
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=None,
        help=(
            "Duration of generated silence when --audio-path is omitted. "
            "Defaults to the model processor max_audio_seconds."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help=(
            "Sample rate for generated silence. Defaults to the model processor sampling rate. "
            "Ignored when --audio-path is used."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision passed to mlx_audio.vad.load().",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Pass lazy=True to mlx_audio.vad.load().",
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
            "[{\"name\":\"cache\",\"shape\":[1,32,128],\"dtype\":\"fp16\"}]"
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
        "--flex-input-frame-lens",
        default=None,
        help=(
            "Optional comma-separated frame lengths for flexible input_features shapes "
            "(for example: 100,400,800). Use 'auto' to enable preset [1, feature_frames]."
        ),
    )
    parser.add_argument(
        "--flex-input-names",
        default="input_features",
        help=(
            "Comma-separated input names eligible for flexible last-dimension handling "
            "when --flex-input-frame-lens is enabled."
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
        help="Temporarily set the MLX model to training mode during graph export capture.",
    )
    return parser.parse_args()


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


def _parse_flex_frame_lengths(
    raw: str | None,
    *,
    feature_frame_count: int,
) -> list[int] | None:
    parsed = parse_flex_lengths(raw, preset_values=[1, int(feature_frame_count)])
    if parsed is None:
        return None
    normalized: list[int] = []
    for value in [1, *parsed, int(feature_frame_count)]:
        if int(value) <= 0:
            raise ValueError(f"--flex-input-frame-lens values must be positive, got {value}.")
        ivalue = int(value)
        if ivalue not in normalized:
            normalized.append(ivalue)
    return normalized


def _parse_flex_input_names(raw: str) -> set[str]:
    return parse_flex_input_names(raw)


def _resolve_processor_attr(model: Any, attr_name: str, default: Any) -> Any:
    config = getattr(model, "config", None)
    processor = getattr(config, "processor_config", None)
    return getattr(processor, attr_name, default)


def _prepare_input_features(
    model: Any,
    *,
    audio_path: Path | None,
    audio_seconds: float | None,
    sample_rate: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not hasattr(model, "prepare_input_features"):
        raise TypeError(
            "mlx-audio conversion currently expects a model with prepare_input_features()."
        )

    resolved_sample_rate = int(
        sample_rate
        if sample_rate is not None
        else _resolve_processor_attr(model, "sampling_rate", 16000)
    )
    if resolved_sample_rate <= 0:
        raise ValueError(f"sample rate must be positive, got {resolved_sample_rate}.")

    if audio_path is not None:
        features = model.prepare_input_features(str(audio_path), sample_rate=resolved_sample_rate)
        metadata = {
            "audio_source": str(Path(audio_path).resolve()),
            "input_feature_source": "audio_path",
            "resolved_sample_rate": int(_resolve_processor_attr(model, "sampling_rate", resolved_sample_rate)),
            "requested_audio_seconds": None,
            "generated_samples": None,
        }
    else:
        resolved_audio_seconds = float(
            audio_seconds
            if audio_seconds is not None
            else _resolve_processor_attr(model, "max_audio_seconds", 8)
        )
        if resolved_audio_seconds <= 0.0:
            raise ValueError(f"audio duration must be positive, got {resolved_audio_seconds}.")
        sample_count = max(1, int(round(resolved_sample_rate * resolved_audio_seconds)))
        silence = np.zeros((sample_count,), dtype=np.float32)
        features = model.prepare_input_features(silence, sample_rate=resolved_sample_rate)
        metadata = {
            "audio_source": "generated_silence",
            "input_feature_source": "generated_silence",
            "resolved_sample_rate": resolved_sample_rate,
            "requested_audio_seconds": resolved_audio_seconds,
            "generated_samples": sample_count,
        }

    features_np = np.asarray(features)
    if features_np.ndim not in (2, 3):
        raise ValueError(
            f"prepare_input_features() must return rank-2 or rank-3 features, got {features_np.shape}."
        )
    metadata["input_feature_shape"] = [int(v) for v in features_np.shape]
    metadata["input_feature_frames"] = int(features_np.shape[-1])
    return features_np, metadata


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    top_ops_str = ", ".join(f"`{name}` x{count}" for name, count in report["top_ops"])
    compiled = report["artifacts"]["compiled"]
    lines = [
        "# Audio Model Probe Report",
        "",
        f"- Model: `{report['model_id']}`",
        f"- Audio source: `{report['audio_source']}`",
        f"- Input feature source: `{report['input_feature_source']}`",
        f"- Input feature shape: `{json.dumps(report['input_feature_shape'])}`",
        f"- Input feature frames: `{report['input_feature_frames']}`",
        f"- Resolved sample rate: `{report['resolved_sample_rate']}`",
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
        lines.append(
            f"- Flexible input shapes: `{json.dumps(report['flex_input_shapes'], sort_keys=True)}`"
        )
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
    flex_input_names = _parse_flex_input_names(args.flex_input_names)
    state_specs = _load_state_specs(args.state_specs_json)

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
        run_kind="mlx_audio_vad_weighted_probe",
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
    run_context["audio_path"] = str(args.audio_path.resolve()) if args.audio_path else None
    run_context["audio_seconds"] = float(args.audio_seconds) if args.audio_seconds is not None else None
    run_context["sample_rate"] = int(args.sample_rate) if args.sample_rate is not None else None
    run_context["revision"] = args.revision
    run_context["lazy_load"] = bool(args.lazy_load)
    run_context["allow_unknown_sources"] = bool(args.allow_unknown_sources)
    run_context["capture_is_training"] = bool(args.capture_is_training)
    run_context["analyze_placement"] = bool(args.analyze_placement)
    run_context["flex_input_frame_lens"] = args.flex_input_frame_lens
    run_context["flex_input_names"] = sorted(flex_input_names)
    run_context["state_specs_json"] = str(args.state_specs_json) if args.state_specs_json else None
    run_context["state_specs"] = [spec.to_dict() for spec in state_specs] if state_specs else []
    run_context["input_feature_shape"] = None
    run_context["input_feature_frames"] = None
    write_json(run_context_path, run_context)

    stage_status = {
        "load_model": "pending",
        "prepare_features": "pending",
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
    input_feature_metadata: dict[str, Any] = {
        "audio_source": str(args.audio_path.resolve()) if args.audio_path else "generated_silence",
        "input_feature_source": "unknown",
        "input_feature_shape": None,
        "input_feature_frames": None,
        "resolved_sample_rate": (
            int(args.sample_rate) if args.sample_rate is not None else None
        ),
        "requested_audio_seconds": (
            float(args.audio_seconds) if args.audio_seconds is not None else None
        ),
        "generated_samples": None,
    }
    program = None
    converted = None
    conversion_config: ConversionConfig | None = None

    try:
        with timed_stage(stage_timings, "load_model"):
            from mlx_audio.vad import load

            model = load(args.model_id, lazy=args.lazy_load, revision=args.revision)
            if hasattr(model, "eval"):
                model.eval()
        stage_status["load_model"] = "ok"

        with timed_stage(stage_timings, "prepare_features"):
            input_features, input_feature_metadata = _prepare_input_features(
                model,
                audio_path=args.audio_path,
                audio_seconds=args.audio_seconds,
                sample_rate=args.sample_rate,
            )
            flex_input_lens = _parse_flex_frame_lengths(
                args.flex_input_frame_lens,
                feature_frame_count=int(input_feature_metadata["input_feature_frames"]),
            )
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
            inputs = {"input_features": input_features}
            run_context["input_feature_shape"] = input_feature_metadata["input_feature_shape"]
            run_context["input_feature_frames"] = input_feature_metadata["input_feature_frames"]
            run_context["resolved_sample_rate"] = input_feature_metadata["resolved_sample_rate"]
            write_json(run_context_path, run_context)
        stage_status["prepare_features"] = "ok"

        with timed_stage(stage_timings, "capture"):
            assert conversion_config is not None
            captured = capture_mlx_graph(
                model,
                inputs,
                dot_output_path=dot_path,
                capture_mode=conversion_config.capture_mode,
                allow_unknown_sources=conversion_config.allow_unknown_sources,
                capture_is_training=conversion_config.capture_is_training,
                capture_function=lambda input_features: _select_primary_output(model(input_features)),
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
            normalized_graph, expected, op_counts = _normalize_graph_for_function(graph, expected)
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
                assert conversion_config is not None
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
            assert conversion_config is not None
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
        "run_kind": "mlx_audio_vad_weighted_probe",
        "run_context": run_context,
        "model_id": args.model_id,
        "audio_source": input_feature_metadata["audio_source"],
        "input_feature_source": input_feature_metadata["input_feature_source"],
        "input_feature_shape": input_feature_metadata["input_feature_shape"],
        "input_feature_frames": input_feature_metadata["input_feature_frames"],
        "resolved_sample_rate": input_feature_metadata["resolved_sample_rate"],
        "requested_audio_seconds": input_feature_metadata["requested_audio_seconds"],
        "generated_samples": input_feature_metadata["generated_samples"],
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
        "flex_input_frame_lens": (
            list(conversion_config.flex_input_lens)
            if conversion_config is not None and conversion_config.flex_input_lens is not None
            else None
        ),
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
    print(f"Audio source: {input_feature_metadata['audio_source']}")
    print(f"Input feature shape: {input_feature_metadata['input_feature_shape']}")
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
    "_load_state_specs",
    "_normalize_graph_for_function",
    "_parse_flex_frame_lengths",
    "_parse_flex_input_names",
    "_prepare_input_features",
    "main",
    "parse_args",
]
