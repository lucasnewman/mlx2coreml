import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.compute_plan import analyze_compiled_model_placement  # noqa: E402
from mlx2coreml.compilation import compile_mlmodel  # noqa: E402
from mlx2coreml.lower_to_mil import (  # noqa: E402
    build_mil_program,
    convert_program_to_model,
)
from tests.model_zoo import (  # noqa: E402
    available_model_names,
    available_live_model_names,
    capture_model_spec,
    get_model_spec,
    supports_live_capture,
)
from mlx2coreml.op_registry import UnsupportedOpsError, ensure_supported, unsupported_op_details  # noqa: E402
from mlx2coreml.passes import infer_graph_specs, normalize_graph, summarize_inference  # noqa: E402
from mlx2coreml.reporting import (  # noqa: E402
    build_run_context,
    init_stage_timings,
    summarize_stage_timings,
    timed_stage,
    write_json,
)


def parse_deployment_target(target_name: str):
    if not hasattr(ct.target, target_name):
        valid = [name for name in dir(ct.target) if name.startswith("iOS")]
        raise ValueError(
            f"Unknown deployment target: {target_name}. Valid examples: {', '.join(valid)}"
        )
    return getattr(ct.target, target_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model-zoo conversion/evaluation for MLX->MIL translator."
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or 'all'.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/zoo"),
        help="Base artifacts directory (or run directory when --run-name is omitted).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name; when set, artifacts are written under artifacts-dir/run-name.",
    )
    parser.add_argument(
        "--capture-mode",
        default="static",
        choices=["static", "live", "auto"],
        help=(
            "Graph source mode: static (zoo fixture graphs), "
            "live (capture graphs via callback/static capture hooks; fail if unavailable), "
            "auto (try live capture, then fallback to static)."
        ),
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
        "--skip-compile",
        action="store_true",
        help="Skip .mlmodelc compilation.",
    )
    parser.add_argument(
        "--eval-compiled",
        action="store_true",
        help="Evaluate compiled model (.mlmodelc) against expected outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed base used for deterministic model-input generation.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-3,
        help="Fallback absolute tolerance for compiled parity checks.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="Fallback relative tolerance for compiled parity checks.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one model fails.",
    )
    parser.add_argument(
        "--analyze-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Analyze ANE/GPU/CPU placement from compiled models using compute plans.",
    )
    return parser.parse_args()


def _select_models(arg: str, capture_mode: str) -> list[str]:
    static_models = available_model_names()
    live_only_models = available_live_model_names()
    allowed_models = (
        sorted(set([*static_models, *live_only_models]))
        if capture_mode in {"live", "auto"}
        else static_models
    )

    if arg.strip().lower() == "all":
        return allowed_models
    models = [name.strip() for name in arg.split(",") if name.strip()]
    unknown = [name for name in models if name not in allowed_models]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(allowed_models)}"
        )
    return models


def _evaluate_compiled_outputs(
    compiled_path: Path,
    inputs: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    output_order: list[str],
    atol: float,
    rtol: float,
) -> dict[str, dict[str, float | str]]:
    compiled = ct.models.CompiledMLModel(str(compiled_path))
    predicted_raw = compiled.predict(inputs)
    predicted_items = list(predicted_raw.items())
    if len(predicted_items) != len(expected):
        raise RuntimeError(
            f"Output count mismatch: expected {len(expected)}, got {len(predicted_items)}"
        )

    stats: dict[str, dict[str, float | str]] = {}
    for idx, expected_name in enumerate(output_order):
        expected_array = np.asarray(expected[expected_name])
        if expected_name in predicted_raw:
            pred_key = expected_name
            predicted_array = np.asarray(predicted_raw[pred_key])
        else:
            pred_key, pred_value = predicted_items[idx]
            predicted_array = np.asarray(pred_value)

        if predicted_array.shape != expected_array.shape:
            # Core ML may represent scalar outputs as length-1 tensors.
            if expected_array.size == predicted_array.size == 1:
                expected_array = expected_array.reshape(1)
                predicted_array = predicted_array.reshape(1)
            else:
                raise RuntimeError(
                    f"Shape mismatch for {expected_name}: expected {expected_array.shape}, got {predicted_array.shape}"
                )

        if np.issubdtype(expected_array.dtype, np.integer):
            ok = np.array_equal(predicted_array.astype(expected_array.dtype), expected_array)
            max_abs_err = float(np.max(np.abs(predicted_array.astype(np.float64) - expected_array.astype(np.float64))))
            max_rel_err = 0.0
        else:
            abs_err = np.abs(predicted_array - expected_array)
            max_abs_err = float(np.max(abs_err))
            denom = np.maximum(np.abs(expected_array), 1e-12)
            max_rel_err = float(np.max(abs_err / denom))
            ok = np.allclose(predicted_array, expected_array, atol=atol, rtol=rtol)

        if not ok:
            raise RuntimeError(
                f"Parity failed for {expected_name}: pred_key={pred_key}, "
                f"max_abs_err={max_abs_err}, max_rel_err={max_rel_err}, atol={atol}, rtol={rtol}"
            )

        stats[expected_name] = {
            "predicted_key": pred_key,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
        }
    return stats


def _build_model_report(
    *,
    run_context: dict[str, Any],
    model_name: str,
    description: str,
    seed: int,
    graph_ops: list[str],
    stage_status: dict[str, str],
    stage_durations_sec: dict[str, float | None],
    total_duration_sec: float,
    model_path: Path,
    capture_dot_path: Path | None,
    compiled_path: Path | None,
    fallback_op_count: int | None,
    top_fallback_ops: list[list[Any]],
    placement_analysis: dict[str, Any] | None,
    placement_error: str | None,
    parity_stats: dict[str, dict[str, float | str]] | None,
    inference_summary: dict[str, int] | None,
    unsupported_details: list[dict[str, Any]] | None = None,
    capture_note: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": run_context["schema_version"],
        "run_kind": "zoo_model",
        "run_context": run_context,
        "model": model_name,
        "description": description,
        "seed": int(seed),
        "ops": list(graph_ops),
        "stage_status": dict(stage_status),
        "stage_durations_sec": dict(stage_durations_sec),
        "total_duration_sec": float(total_duration_sec),
        "status": "ok" if error is None else "failed",
        "fallback_op_count": fallback_op_count,
        "top_fallback_ops": top_fallback_ops,
        "placement_analysis": placement_analysis,
        "placement_error": placement_error,
        "artifacts": {
            "capture_dot": str(capture_dot_path) if capture_dot_path else None,
            "model": str(model_path),
            "compiled": str(compiled_path) if compiled_path else None,
        },
        "compiled": compiled_path is not None,
        "evaluated": parity_stats is not None,
        "inference": inference_summary,
        "unsupported_ops": unsupported_details or [],
        "capture_note": capture_note,
        "parity": parity_stats,
        "error": error,
    }


def _write_model_report_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Model Zoo Run: {report['model']}",
        "",
        f"- Description: {report['description']}",
        f"- Seed: `{report['seed']}`",
        f"- Capture mode: `{report['run_context']['capture_mode']}`",
        f"- Ops: `{', '.join(report['ops'])}`",
        f"- Stage status: `{json.dumps(report['stage_status'], sort_keys=True)}`",
        f"- Stage durations (sec): `{json.dumps(report['stage_durations_sec'], sort_keys=True)}`",
        f"- Total stage duration (sec): `{report['total_duration_sec']}`",
        f"- Status: `{report['status']}`",
        f"- Deployment target: `{report['run_context']['deployment_target']}`",
        f"- Target profile: `{report['run_context']['target_profile']}`",
        f"- Conversion backend: `{report['run_context']['convert_to']}`",
        f"- Fallback ops: `{report['fallback_op_count']}`",
        "- Versions:",
        f"  - python: `{report['run_context']['versions']['python']}`",
        f"  - coremltools: `{report['run_context']['versions']['coremltools']}`",
        f"  - numpy: `{report['run_context']['versions']['numpy']}`",
        f"  - mlx: `{report['run_context']['versions']['mlx']}`",
        f"- Model artifact: `{report['artifacts']['model']}`",
        (
            f"- Capture DOT: `{report['artifacts']['capture_dot']}`"
            if report["artifacts"]["capture_dot"]
            else "- Capture DOT: skipped"
        ),
        (
            f"- Compiled artifact: `{report['artifacts']['compiled']}`"
            if report["artifacts"]["compiled"]
            else "- Compiled artifact: skipped"
        ),
    ]
    if report.get("capture_note"):
        lines.append(f"- Capture note: `{report['capture_note']}`")
    if report["top_fallback_ops"]:
        lines.append(
            "- Top fallback ops: "
            + ", ".join(f"`{name}` x{count}" for name, count in report["top_fallback_ops"])
        )
    if report.get("placement_error"):
        lines.append(f"- Placement analysis error: `{report['placement_error']}`")
    if report.get("inference") is not None:
        lines.append(
            "- Inference coverage: "
            f"{report['inference']['with_shape']}/{report['inference']['total_tensors']} with shape, "
            f"{report['inference']['with_dtype']}/{report['inference']['total_tensors']} with dtype"
        )
    if report["unsupported_ops"]:
        lines.append("- Unsupported ops:")
        for detail in report["unsupported_ops"]:
            count = detail.get("count", 1)
            primitive = detail.get("primitive")
            primitive_text = f", primitive={primitive}" if primitive else ""
            lines.append(
                f"  - `{detail['op']}` (count={count}, status={detail['status']}, "
                f"source={detail['source']}{primitive_text})"
            )
    if report["parity"] is not None:
        lines.append("- Parity:")
        for out_name, stats in report["parity"].items():
            lines.append(
                f"  - `{out_name}` -> key `{stats['predicted_key']}`, "
                f"max_abs_err={stats['max_abs_err']}, max_rel_err={stats['max_rel_err']}"
            )
    if report["error"] is not None:
        lines.append(f"- Error: `{report['error']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    target = parse_deployment_target(args.deployment_target)
    selected_models = _select_models(args.models, args.capture_mode)
    random.seed(args.seed)
    np.random.seed(args.seed)

    artifacts_root = (
        args.artifacts_dir / args.run_name if args.run_name else args.artifacts_dir
    )
    artifacts_root.mkdir(parents=True, exist_ok=True)

    run_context = build_run_context(
        run_kind="zoo",
        deployment_target=args.deployment_target,
        convert_to=args.convert_to,
        seed=args.seed,
    )
    run_context["capture_mode"] = args.capture_mode
    run_context["target_profile"] = args.target_profile
    run_context["model_count"] = len(selected_models)
    run_context["models"] = list(selected_models)
    run_context["analyze_placement"] = bool(args.analyze_placement)
    write_json(artifacts_root / "run_context.json", run_context)

    summary: list[dict[str, object]] = []
    failures: list[str] = []
    for model_index, model_name in enumerate(selected_models):
        model_seed = args.seed + model_index
        model_dir = artifacts_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        stage_status = {
            "capture": "pending",
            "normalize": "pending",
            "type_infer": "pending",
            "support_check": "pending",
            "lower": "pending",
            "convert": "pending",
            "compile": "pending" if not args.skip_compile else "skipped",
            "placement_analysis": (
                "pending" if (not args.skip_compile and bool(args.analyze_placement)) else "skipped"
            ),
            "eval": "pending" if (not args.skip_compile and args.eval_compiled) else "skipped",
        }
        stage_timings = init_stage_timings(stage_status.keys())
        error_message: str | None = None
        unsupported: list[dict[str, Any]] | None = None
        capture_note: str | None = None
        spec = None

        model_path = model_dir / ("model.mlpackage" if args.convert_to == "mlprogram" else "model.mlmodel")
        compiled_path: Path | None = None
        capture_dot_path: Path | None = None
        parity_stats: dict[str, dict[str, float | str]] | None = None
        placement_analysis: dict[str, Any] | None = None
        placement_error: str | None = None
        inference_summary: dict[str, int] | None = None

        try:
            with timed_stage(stage_timings, "capture"):
                if args.capture_mode == "static":
                    spec = get_model_spec(model_name, seed=model_seed)
                    stage_status["capture"] = "mock"
                elif args.capture_mode == "live":
                    if not supports_live_capture(model_name):
                        stage_status["capture"] = "failed"
                        raise RuntimeError(
                            f"Live capture is not implemented for '{model_name}'. "
                            "Use --capture-mode static/auto or implement capture hook."
                        )
                    spec = capture_model_spec(
                        model_name,
                        seed=model_seed,
                        artifacts_dir=model_dir,
                        write_debug_dot=True,
                    )
                    maybe_dot = model_dir / "capture_graph.dot"
                    capture_dot_path = maybe_dot if maybe_dot.exists() else None
                    stage_status["capture"] = "ok"
                else:  # auto
                    if supports_live_capture(model_name):
                        try:
                            spec = capture_model_spec(
                                model_name,
                                seed=model_seed,
                                artifacts_dir=model_dir,
                                write_debug_dot=True,
                            )
                            maybe_dot = model_dir / "capture_graph.dot"
                            capture_dot_path = maybe_dot if maybe_dot.exists() else None
                            stage_status["capture"] = "ok"
                        except Exception as exc:
                            spec = get_model_spec(model_name, seed=model_seed)
                            stage_status["capture"] = "mock"
                            capture_note = (
                                "Live capture failed, using static fixture graph. "
                                f"Reason: {exc}"
                            )
                    else:
                        spec = get_model_spec(model_name, seed=model_seed)
                        stage_status["capture"] = "mock"
                        capture_note = (
                            "Live capture unavailable for this model; using static fixture graph."
                        )

            if spec is None:
                raise RuntimeError("Internal error: model spec was not resolved after capture stage.")

            (model_dir / "graph.json").write_text(
                json.dumps(spec.graph.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
            np.savez(model_dir / "inputs.npz", **spec.inputs)
            np.savez(model_dir / "expected_outputs.npz", **spec.expected)

            with timed_stage(stage_timings, "normalize"):
                normalized_graph = normalize_graph(spec.graph)
            stage_status["normalize"] = "ok"

            with timed_stage(stage_timings, "type_infer"):
                inferred = infer_graph_specs(normalized_graph)
                inference_summary = summarize_inference(inferred)
            stage_status["type_infer"] = "ok"

            with timed_stage(stage_timings, "support_check"):
                unsupported = unsupported_op_details(normalized_graph)
                ensure_supported(normalized_graph)
            stage_status["support_check"] = "ok"

            with timed_stage(stage_timings, "lower"):
                program = build_mil_program(
                    normalized_graph,
                    deployment_target=target,
                    normalize=False,
                    target_profile=args.target_profile,
                )
            stage_status["lower"] = "ok"
            (model_dir / "program.mil.txt").write_text(str(program) + "\n", encoding="utf-8")

            with timed_stage(stage_timings, "convert"):
                converted = convert_program_to_model(
                    program,
                    deployment_target=target,
                    convert_to=args.convert_to,
                )
                converted.save(str(model_path))
            stage_status["convert"] = "ok"

            if not args.skip_compile:
                with timed_stage(stage_timings, "compile"):
                    compiled_path = compile_mlmodel(model_path, model_dir / "model.mlmodelc")
                stage_status["compile"] = "ok"
                if stage_status["placement_analysis"] == "pending":
                    try:
                        with timed_stage(stage_timings, "placement_analysis"):
                            placement_analysis = analyze_compiled_model_placement(
                                compiled_path,
                                compute_units="all",
                            )
                        stage_status["placement_analysis"] = "ok"
                    except Exception as exc:  # pragma: no cover - environment dependent
                        placement_error = str(exc)
                        stage_status["placement_analysis"] = "failed"
                if args.eval_compiled:
                    with timed_stage(stage_timings, "eval"):
                        parity_stats = _evaluate_compiled_outputs(
                            compiled_path=compiled_path,
                            inputs=spec.inputs,
                            expected=spec.expected,
                            output_order=spec.graph.outputs,
                            atol=max(args.atol, spec.atol),
                            rtol=max(args.rtol, spec.rtol),
                        )
                    stage_status["eval"] = "ok"
        except UnsupportedOpsError as exc:
            stage_status["support_check"] = "failed"
            unsupported = exc.details
            error_message = str(exc)
        except Exception as exc:
            if not any(value == "failed" for value in stage_status.values()):
                failed_stage = next((k for k, v in stage_status.items() if v == "pending"), None)
                if failed_stage is not None:
                    stage_status[failed_stage] = "failed"
            error_message = str(exc)

        stage_durations_sec, total_duration_sec = summarize_stage_timings(stage_timings)
        report = _build_model_report(
            run_context=run_context,
            model_name=spec.name if spec is not None else model_name,
            description=spec.description if spec is not None else "Model conversion failed before spec resolved.",
            seed=model_seed,
            graph_ops=[node.op for node in spec.graph.nodes] if spec is not None else [],
            stage_status=stage_status,
            stage_durations_sec=stage_durations_sec,
            total_duration_sec=total_duration_sec,
            model_path=model_path,
            capture_dot_path=capture_dot_path,
            compiled_path=compiled_path,
            fallback_op_count=(
                int(placement_analysis["fallback_operation_count"])
                if placement_analysis is not None
                else None
            ),
            top_fallback_ops=(
                placement_analysis.get("top_fallback_ops", []) if placement_analysis is not None else []
            ),
            placement_analysis=placement_analysis,
            placement_error=placement_error,
            parity_stats=parity_stats,
            inference_summary=inference_summary,
            unsupported_details=unsupported,
            capture_note=capture_note,
            error=error_message,
        )
        write_json(model_dir / "report.json", report)
        _write_model_report_markdown(model_dir / "report.md", report)

        success = error_message is None
        summary.append(report)
        if success:
            print(
                f"[ok] {report['model']}: model={model_path}"
                + (f", compiled={compiled_path}" if compiled_path else ", compiled=skipped")
            )
        else:
            failures.append(report["model"])
            print(f"[failed] {report['model']}: {error_message}")
            if args.fail_fast:
                break

    summary_path = artifacts_root / "summary.json"
    write_json(summary_path, summary)
    print(f"Wrote summary: {summary_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} model(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
