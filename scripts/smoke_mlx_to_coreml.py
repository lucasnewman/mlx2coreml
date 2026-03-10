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

from mlx2coreml.from_mlx import (
    build_smoke_numpy_inputs,
    capture_smoke_graph,
    evaluate_smoke_numpy,
    make_mock_smoke_graph,
)
from mlx2coreml.compute_plan import analyze_compiled_model_placement
from mlx2coreml.lower_to_mil import (
    build_mil_program,
    compile_model_artifact,
    convert_program_to_model,
)
from mlx2coreml.op_registry import ensure_supported
from mlx2coreml.passes import infer_graph_specs, normalize_graph, summarize_inference
from mlx2coreml.reporting import (
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
        description="Smoke pipeline: MLX graph capture -> MIL lowering -> Core ML model + compile."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/smoke"),
        help="Base artifacts directory (or run directory when --run-name is omitted).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name; when set, artifacts are written under artifacts-dir/run-name.",
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
        help="Skip .mlmodelc compilation step.",
    )
    parser.add_argument(
        "--analyze-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Analyze ANE/GPU/CPU placement from the compiled model compute plan.",
    )
    parser.add_argument(
        "--eval-compiled",
        action="store_true",
        help="Run prediction directly from compiled .mlmodelc and compare against expected output.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=2e-3,
        help="Absolute tolerance used for compiled-model output comparison checks.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="Relative tolerance used for compiled-model output comparison checks.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock IR instead of live MLX capture (useful when MLX runtime is unavailable).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for smoke input generation.")
    return parser.parse_args()


def _write_report_markdown(report_path: Path, report: dict[str, Any]) -> None:
    compiled_artifact = report["artifacts"]["compiled"]
    lines = [
        "# Smoke Run Report",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Seed: `{report['seed']}`",
        f"- Status: `{report['status']}`",
        f"- Stage status: `{json.dumps(report['stage_status'], sort_keys=True)}`",
        f"- Stage durations (sec): `{json.dumps(report['stage_durations_sec'], sort_keys=True)}`",
        f"- Total stage duration (sec): `{report['total_duration_sec']}`",
        f"- Deployment target: `{report['run_context']['deployment_target']}`",
        f"- Target profile: `{report['run_context']['target_profile']}`",
        f"- Conversion backend: `{report['run_context']['convert_to']}`",
        f"- Graph ops: `{', '.join(report['ops'])}`",
        f"- Fallback ops: `{report['fallback_op_count']}`",
        "- Versions:",
        f"  - python: `{report['run_context']['versions']['python']}`",
        f"  - coremltools: `{report['run_context']['versions']['coremltools']}`",
        f"  - numpy: `{report['run_context']['versions']['numpy']}`",
        f"  - mlx: `{report['run_context']['versions']['mlx']}`",
        f"- MIL dump: `{report['artifacts']['mil_program']}`",
        f"- Model artifact: `{report['artifacts']['model']}`",
        f"- Compiled artifact: `{compiled_artifact}`" if compiled_artifact else "- Compiled artifact: skipped",
    ]
    if report.get("inference") is not None:
        lines.append(
            "- Inference coverage: "
            f"{report['inference']['with_shape']}/{report['inference']['total_tensors']} with shape, "
            f"{report['inference']['with_dtype']}/{report['inference']['total_tensors']} with dtype"
        )
    if report["parity"] is not None:
        lines.extend(
            [
                f"- Compiled eval output key: `{report['parity']['output_key']}`",
                f"- Compiled eval max abs err: `{report['parity']['max_abs_err']}`",
                f"- Compiled eval max rel err: `{report['parity']['max_rel_err']}`",
            ]
        )
    if report["top_fallback_ops"]:
        lines.append(
            "- Top fallback ops: "
            + ", ".join(f"`{name}` x{count}" for name, count in report["top_fallback_ops"])
        )
    if report.get("placement_error") is not None:
        lines.append(f"- Placement analysis error: `{report['placement_error']}`")
    if report["error"] is not None:
        lines.append(f"- Error: `{report['error']}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_compiled_model(
    compiled_path: Path,
    inputs: dict[str, np.ndarray],
    expected: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, str]:
    model = ct.models.CompiledMLModel(str(compiled_path))
    outputs = model.predict(inputs)
    if len(outputs) != 1:
        raise RuntimeError(
            f"Expected 1 output from smoke model, got {len(outputs)}: {list(outputs.keys())}"
        )

    output_key, output_value = next(iter(outputs.items()))
    predicted = np.asarray(output_value)

    if predicted.shape != expected.shape:
        raise RuntimeError(
            f"Compiled output shape mismatch: expected {expected.shape}, got {predicted.shape}"
        )

    abs_err = np.abs(predicted - expected)
    max_abs_err = float(np.max(abs_err))
    denom = np.maximum(np.abs(expected), 1e-12)
    max_rel_err = float(np.max(abs_err / denom))
    if not np.allclose(predicted, expected, atol=atol, rtol=rtol):
        raise RuntimeError(
            "Compiled-model parity check failed: "
            f"max_abs_err={max_abs_err}, max_rel_err={max_rel_err}, atol={atol}, rtol={rtol}"
        )

    return {
        "output_key": output_key,
        "max_abs_err": f"{max_abs_err:.8g}",
        "max_rel_err": f"{max_rel_err:.8g}",
    }


def main() -> None:
    args = parse_args()
    target = parse_deployment_target(args.deployment_target)
    random.seed(args.seed)
    np.random.seed(args.seed)

    artifacts_dir = args.artifacts_dir / args.run_name if args.run_name else args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_context = build_run_context(
        run_kind="smoke",
        deployment_target=args.deployment_target,
        convert_to=args.convert_to,
        seed=args.seed,
    )
    run_context["target_profile"] = args.target_profile
    run_context["analyze_placement"] = bool(args.analyze_placement)
    write_json(artifacts_dir / "run_context.json", run_context)

    dot_path = artifacts_dir / "smoke_graph.dot"
    graph_json_path = artifacts_dir / "smoke_graph.json"
    inputs_npz_path = artifacts_dir / "smoke_inputs.npz"
    expected_npy_path = artifacts_dir / "smoke_expected.npy"
    mil_program_path = artifacts_dir / "smoke_program.mil.txt"
    model_path = artifacts_dir / ("smoke.mlpackage" if args.convert_to == "mlprogram" else "smoke.mlmodel")
    compiled_path = artifacts_dir / "smoke.mlmodelc"
    report_path = artifacts_dir / "smoke_report.md"
    report_json_path = artifacts_dir / "smoke_report.json"

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

    mode: str
    graph_ops: list[str] = []
    compiled_out: Path | None = None
    compiled_eval_result: dict[str, str] | None = None
    placement_analysis: dict[str, Any] | None = None
    placement_error: str | None = None
    inference_summary: dict[str, int] | None = None
    error_message: str | None = None
    try:
        if args.mock:
            with timed_stage(stage_timings, "capture"):
                graph = make_mock_smoke_graph()
                inputs = build_smoke_numpy_inputs(seed=args.seed)
                expected = evaluate_smoke_numpy(inputs)
            mode = "mock"
            stage_status["capture"] = "skipped"
            stage_timings["capture"] = None
        else:
            with timed_stage(stage_timings, "capture"):
                graph, inputs, expected = capture_smoke_graph(dot_output_path=dot_path, seed=args.seed)
            mode = "mlx-live"
            stage_status["capture"] = "ok"

        # Simple parity check between expected capture output and numpy reference.
        numpy_reference = evaluate_smoke_numpy(inputs)
        if not np.allclose(expected, numpy_reference, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Smoke validation failed: MLX output does not match numpy reference.")

        with timed_stage(stage_timings, "normalize"):
            normalized_graph = normalize_graph(graph)
        stage_status["normalize"] = "ok"

        with timed_stage(stage_timings, "type_infer"):
            inferred = infer_graph_specs(normalized_graph)
            inference_summary = summarize_inference(inferred)
        stage_status["type_infer"] = "ok"

        with timed_stage(stage_timings, "support_check"):
            ensure_supported(normalized_graph)
        stage_status["support_check"] = "ok"

        graph_ops = [node.op for node in normalized_graph.nodes]
        graph_json_path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n", encoding="utf-8")
        np.savez(inputs_npz_path, **inputs)
        np.save(expected_npy_path, expected)

        with timed_stage(stage_timings, "lower"):
            program = build_mil_program(
                normalized_graph,
                deployment_target=target,
                normalize=False,
                target_profile=args.target_profile,
            )
        stage_status["lower"] = "ok"
        mil_program_path.write_text(str(program) + "\n", encoding="utf-8")

        with timed_stage(stage_timings, "convert"):
            model = convert_program_to_model(program, deployment_target=target, convert_to=args.convert_to)
            model.save(str(model_path))
        stage_status["convert"] = "ok"

        if not args.skip_compile:
            with timed_stage(stage_timings, "compile"):
                compiled_out = compile_model_artifact(model_path, compiled_path)
            stage_status["compile"] = "ok"
            if stage_status["placement_analysis"] == "pending":
                try:
                    with timed_stage(stage_timings, "placement_analysis"):
                        placement_analysis = analyze_compiled_model_placement(
                            compiled_out,
                            compute_units="all",
                        )
                    stage_status["placement_analysis"] = "ok"
                except Exception as exc:  # pragma: no cover - environment-dependent APIs
                    placement_error = str(exc)
                    stage_status["placement_analysis"] = "failed"
            if args.eval_compiled:
                with timed_stage(stage_timings, "eval"):
                    compiled_eval_result = evaluate_compiled_model(
                        compiled_path=compiled_out,
                        inputs=inputs,
                        expected=expected,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                stage_status["eval"] = "ok"
    except Exception as exc:
        error_message = str(exc)
        failed_stage = next((k for k, v in stage_status.items() if v == "pending"), "unknown")
        stage_status[failed_stage] = "failed"
        mode = "mock" if args.mock else "mlx-live"

    stage_durations_sec, total_duration_sec = summarize_stage_timings(stage_timings)
    report_json = {
        "schema_version": run_context["schema_version"],
        "run_kind": "smoke",
        "run_context": run_context,
        "mode": mode,
        "seed": args.seed,
        "status": "ok" if error_message is None else "failed",
        "stage_status": stage_status,
        "stage_durations_sec": stage_durations_sec,
        "total_duration_sec": total_duration_sec,
        "ops": graph_ops,
        "fallback_op_count": (
            int(placement_analysis["fallback_operation_count"]) if placement_analysis is not None else None
        ),
        "top_fallback_ops": (
            placement_analysis.get("top_fallback_ops", []) if placement_analysis is not None else []
        ),
        "placement_analysis": placement_analysis,
        "placement_error": placement_error,
        "artifacts": {
            "graph_json": str(graph_json_path),
            "dot_graph": str(dot_path) if not args.mock else None,
            "inputs": str(inputs_npz_path),
            "expected": str(expected_npy_path),
            "mil_program": str(mil_program_path),
            "model": str(model_path),
            "compiled": str(compiled_out) if compiled_out else None,
        },
        "compiled": compiled_out is not None,
        "evaluated": compiled_eval_result is not None,
        "inference": inference_summary,
        "parity": compiled_eval_result,
        "error": error_message,
    }
    write_json(report_json_path, report_json)
    _write_report_markdown(report_path, report_json)

    if error_message is not None:
        raise RuntimeError(error_message)

    print(f"Mode: {mode}")
    print(f"Wrote graph JSON: {graph_json_path}")
    if not args.mock:
        print(f"Wrote DOT graph: {dot_path}")
    print(f"Wrote inputs: {inputs_npz_path}")
    print(f"Wrote expected output: {expected_npy_path}")
    print(f"Wrote MIL dump: {mil_program_path}")
    print(f"Wrote model: {model_path}")
    if compiled_out:
        print(f"Wrote compiled model: {compiled_out}")
        if compiled_eval_result is not None:
            print(
                "Compiled eval passed: "
                f"output={compiled_eval_result['output_key']}, "
                f"max_abs_err={compiled_eval_result['max_abs_err']}, "
                f"max_rel_err={compiled_eval_result['max_rel_err']}"
            )
    else:
        print("Skipped compiled model step.")
    print(f"Wrote report: {report_path}")
    print(f"Wrote report JSON: {report_json_path}")


if __name__ == "__main__":
    main()
