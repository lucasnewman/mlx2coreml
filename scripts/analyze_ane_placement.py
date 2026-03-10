import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.compute_plan import (  # noqa: E402
    ComputePlanUnavailableError,
    analyze_compiled_model_placement,
)
from mlx2coreml.lower_to_mil import compile_model_artifact  # noqa: E402
from mlx2coreml.reporting import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ANE/GPU/CPU placement and fallback operations from a compiled Core ML model."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to .mlmodelc (preferred), .mlpackage, or .mlmodel.",
    )
    parser.add_argument(
        "--compiled-path",
        type=Path,
        default=None,
        help=(
            "Optional output path for compiled model when --model-path is not .mlmodelc. "
            "If omitted, uses a temporary directory."
        ),
    )
    parser.add_argument(
        "--compute-units",
        default="all",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
        help="Compute-unit target used while loading the compute plan.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top ops to include in top-op summaries.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Maximum number of fallback op samples to include in report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <model-dir>/placement_report.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path. Defaults to <model-dir>/placement_report.md",
    )
    return parser.parse_args()


def _resolve_output_path(
    explicit: Path | None,
    *,
    base_dir: Path,
    filename: str,
) -> Path:
    if explicit is not None:
        return explicit
    return base_dir / filename


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    analysis = report["analysis"]
    lines = [
        "# ANE Placement Report",
        "",
        f"- Input model path: `{report['input_model_path']}`",
        f"- Compiled model path: `{report['compiled_model_path']}`",
        f"- Compute units: `{analysis['compute_units']}`",
        f"- Total operations: `{analysis['total_operations']}`",
        f"- Fallback operations: `{analysis['fallback_operation_count']}`",
        f"- Fallback ratio: `{analysis['fallback_ratio']}`",
        f"- Preferred device counts: `{json.dumps(analysis['preferred_device_counts'], sort_keys=True)}`",
        f"- Fallback cost ratio: `{analysis['fallback_cost_ratio']}`",
        f"- Top ops: `{json.dumps(analysis['top_ops'])}`",
    ]
    if analysis["top_fallback_ops"]:
        lines.append(
            "- Top fallback ops: "
            + ", ".join(f"`{name}` x{count}" for name, count in analysis["top_fallback_ops"])
        )
    if analysis["fallback_samples"]:
        lines.append("- Fallback samples:")
        for sample in analysis["fallback_samples"][:20]:
            lines.append(
                f"  - fn=`{sample['function']}` op=`{sample['operator']}` "
                f"preferred=`{sample['preferred_device']}` "
                f"supported=`{','.join(sample['supported_devices'])}` "
                f"outputs=`{','.join(sample['outputs'])}` "
                f"cost=`{sample['estimated_cost']}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _analyze_from_compiled(
    compiled_path: Path,
    *,
    compute_units: str,
    top_k: int,
    sample_limit: int,
) -> dict[str, Any]:
    return analyze_compiled_model_placement(
        compiled_path,
        compute_units=compute_units,
        top_k=top_k,
        sample_limit=sample_limit,
    )


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    base_dir = model_path if model_path.is_dir() else model_path.parent
    output_json = _resolve_output_path(args.output_json, base_dir=base_dir, filename="placement_report.json")
    output_md = _resolve_output_path(args.output_md, base_dir=base_dir, filename="placement_report.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any]
    if model_path.suffix == ".mlmodelc":
        analysis = _analyze_from_compiled(
            model_path,
            compute_units=args.compute_units,
            top_k=args.top_k,
            sample_limit=args.sample_limit,
        )
        report = {
            "run_kind": "placement_analysis",
            "input_model_path": str(model_path),
            "compiled_model_path": str(model_path),
            "analysis": analysis,
        }
    else:
        if args.compiled_path is not None:
            compiled_out = compile_model_artifact(model_path, args.compiled_path.resolve())
            analysis = _analyze_from_compiled(
                compiled_out,
                compute_units=args.compute_units,
                top_k=args.top_k,
                sample_limit=args.sample_limit,
            )
            report = {
                "run_kind": "placement_analysis",
                "input_model_path": str(model_path),
                "compiled_model_path": str(compiled_out),
                "analysis": analysis,
            }
        else:
            with TemporaryDirectory(prefix="mlx2coreml_placement_") as temp_dir:
                compiled_temp = Path(temp_dir) / "model.mlmodelc"
                compiled_out = compile_model_artifact(model_path, compiled_temp)
                analysis = _analyze_from_compiled(
                    compiled_out,
                    compute_units=args.compute_units,
                    top_k=args.top_k,
                    sample_limit=args.sample_limit,
                )
                report = {
                    "run_kind": "placement_analysis",
                    "input_model_path": str(model_path),
                    "compiled_model_path": str(compiled_out),
                    "analysis": analysis,
                }

    write_json(output_json, report)
    _write_markdown(output_md, report)
    print(f"Wrote placement JSON: {output_json}")
    print(f"Wrote placement markdown: {output_md}")
    print(
        "Fallback ops: "
        f"{report['analysis']['fallback_operation_count']}/{report['analysis']['total_operations']}"
    )


if __name__ == "__main__":
    try:
        main()
    except ComputePlanUnavailableError as exc:
        raise RuntimeError(f"Compute-plan analysis unavailable: {exc}") from exc
