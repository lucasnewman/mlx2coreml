import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.reporting import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Core ML inference latency across compute-unit settings "
            "(ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE)."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to .mlmodelc, .mlpackage, or .mlmodel.",
    )
    parser.add_argument(
        "--inputs-npz",
        type=Path,
        required=True,
        help="NPZ file containing model input arrays.",
    )
    parser.add_argument(
        "--compute-units",
        default="all,cpu_only,cpu_and_gpu,cpu_and_ne",
        help="Comma-separated compute-unit modes to benchmark.",
    )
    parser.add_argument(
        "--function-name",
        default=None,
        help="Optional function name for multifunction models.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Warmup iterations per compute-unit mode.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Timed iterations per compute-unit mode.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for output consistency checks across compute units.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for output consistency checks across compute units.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <model-dir>/compute_unit_benchmark.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to <model-dir>/compute_unit_benchmark.md",
    )
    return parser.parse_args()


def _parse_compute_unit(name: str):
    key = str(name).strip().lower()
    if key in {"all", "default"}:
        return "all", ct.ComputeUnit.ALL
    if key in {"cpu_only", "cpu"}:
        return "cpu_only", ct.ComputeUnit.CPU_ONLY
    if key in {"cpu_and_gpu", "cpu_gpu"}:
        return "cpu_and_gpu", ct.ComputeUnit.CPU_AND_GPU
    if key in {"cpu_and_ne", "cpu_ne", "cpu_and_neural_engine"}:
        return "cpu_and_ne", ct.ComputeUnit.CPU_AND_NE
    raise ValueError(
        f"Unsupported compute unit '{name}'. Use one of all,cpu_only,cpu_and_gpu,cpu_and_ne."
    )


def _load_predictor(model_path: Path, *, compute_unit, function_name: str | None):
    if model_path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(
            str(model_path),
            compute_units=compute_unit,
            function_name=function_name,
        )
    return ct.models.MLModel(
        str(model_path),
        compute_units=compute_unit,
        function_name=function_name,
    )


def _latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    sorted_values = sorted(latencies_ms)
    if not sorted_values:
        return {"min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    p50_index = int(round(0.50 * (len(sorted_values) - 1)))
    p95_index = int(round(0.95 * (len(sorted_values) - 1)))
    return {
        "min_ms": float(sorted_values[0]),
        "max_ms": float(sorted_values[-1]),
        "mean_ms": float(statistics.mean(sorted_values)),
        "p50_ms": float(sorted_values[p50_index]),
        "p95_ms": float(sorted_values[p95_index]),
    }


def _compare_outputs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    details: dict[str, Any] = {"ok": True, "outputs": {}}
    baseline_items = list(baseline.items())
    candidate_items = list(candidate.items())
    if len(baseline_items) != len(candidate_items):
        details["ok"] = False
        details["error"] = (
            f"Output count mismatch: baseline={len(baseline_items)} candidate={len(candidate_items)}"
        )
        return details

    for idx, (base_key, base_value) in enumerate(baseline_items):
        if base_key in candidate:
            cand_key = base_key
            cand_value = candidate[cand_key]
        else:
            cand_key, cand_value = candidate_items[idx]

        base_arr = np.asarray(base_value)
        cand_arr = np.asarray(cand_value)
        if base_arr.shape != cand_arr.shape:
            details["ok"] = False
            details["outputs"][base_key] = {
                "ok": False,
                "error": f"shape mismatch baseline={base_arr.shape}, candidate={cand_arr.shape}",
            }
            continue

        abs_err = np.abs(base_arr - cand_arr)
        max_abs_err = float(np.max(abs_err))
        denom = np.maximum(np.abs(base_arr), 1e-12)
        max_rel_err = float(np.max(abs_err / denom))
        ok = bool(np.allclose(cand_arr, base_arr, atol=atol, rtol=rtol))
        details["outputs"][base_key] = {
            "ok": ok,
            "candidate_key": cand_key,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
        }
        if not ok:
            details["ok"] = False
    return details


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Compute Unit Benchmark",
        "",
        f"- Model path: `{report['model_path']}`",
        f"- Inputs: `{report['inputs_npz']}`",
        f"- Warmup iters: `{report['warmup_iters']}`",
        f"- Timed iters: `{report['iters']}`",
        f"- atol: `{report['atol']}`",
        f"- rtol: `{report['rtol']}`",
    ]
    for item in report["results"]:
        stats = item["latency"]
        lines.extend(
            [
                f"- Mode `{item['compute_unit']}`:",
                (
                    f"  min={stats['min_ms']:.4f}ms mean={stats['mean_ms']:.4f}ms "
                    f"p50={stats['p50_ms']:.4f}ms p95={stats['p95_ms']:.4f}ms "
                    f"max={stats['max_ms']:.4f}ms"
                ),
            ]
        )
        if "parity" in item:
            lines.append(f"  parity_ok={item['parity']['ok']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    inputs_path = args.inputs_npz.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not inputs_path.exists():
        raise FileNotFoundError(f"Inputs NPZ path does not exist: {inputs_path}")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else model_path.parent / "compute_unit_benchmark.json"
    )
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else model_path.parent / "compute_unit_benchmark.md"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with np.load(inputs_path) as npz:
        inputs = {name: np.asarray(npz[name]) for name in npz.files}
    if not inputs:
        raise ValueError(f"No inputs found in {inputs_path}")

    requested_modes = [item.strip() for item in args.compute_units.split(",") if item.strip()]
    parsed_modes = [_parse_compute_unit(item) for item in requested_modes]

    baseline_outputs: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for mode_name, compute_unit in parsed_modes:
        predictor = _load_predictor(
            model_path,
            compute_unit=compute_unit,
            function_name=args.function_name,
        )
        for _ in range(args.warmup_iters):
            predictor.predict(inputs)

        latencies_ms: list[float] = []
        final_outputs: dict[str, Any] | None = None
        for _ in range(args.iters):
            start = time.perf_counter()
            final_outputs = predictor.predict(inputs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

        entry = {
            "compute_unit": mode_name,
            "latency": _latency_stats(latencies_ms),
        }
        if final_outputs is not None:
            if baseline_outputs is None:
                baseline_outputs = final_outputs
            else:
                entry["parity"] = _compare_outputs(
                    baseline_outputs,
                    final_outputs,
                    atol=args.atol,
                    rtol=args.rtol,
                )
        results.append(entry)

    report = {
        "run_kind": "compute_unit_benchmark",
        "model_path": str(model_path),
        "inputs_npz": str(inputs_path),
        "function_name": args.function_name,
        "warmup_iters": int(args.warmup_iters),
        "iters": int(args.iters),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "results": results,
    }
    write_json(output_json, report)
    _write_markdown(output_md, report)
    print(f"Wrote benchmark JSON: {output_json}")
    print(f"Wrote benchmark markdown: {output_md}")
    for item in results:
        stats = item["latency"]
        print(
            f"[{item['compute_unit']}] mean={stats['mean_ms']:.4f}ms, "
            f"p50={stats['p50_ms']:.4f}ms, p95={stats['p95_ms']:.4f}ms"
        )


if __name__ == "__main__":
    main()
