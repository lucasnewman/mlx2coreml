from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import coremltools as ct


class ComputePlanUnavailableError(RuntimeError):
    pass


def _parse_compute_units(compute_units: str):
    key = str(compute_units).strip().lower()
    if key in {"all", "default"}:
        return ct.ComputeUnit.ALL
    if key in {"cpu_only", "cpu"}:
        return ct.ComputeUnit.CPU_ONLY
    if key in {"cpu_and_gpu", "cpu_gpu"}:
        return ct.ComputeUnit.CPU_AND_GPU
    if key in {"cpu_and_ne", "cpu_ne", "cpu_and_neural_engine"}:
        return ct.ComputeUnit.CPU_AND_NE
    raise ValueError(
        f"Unsupported compute_units={compute_units!r}. "
        "Use one of: all, cpu_only, cpu_and_gpu, cpu_and_ne."
    )


def _device_label(device: object | None) -> str:
    if device is None:
        return "UNKNOWN"
    class_name = device.__class__.__name__.lower()
    if "neuralengine" in class_name:
        return "ANE"
    if "gpu" in class_name:
        return "GPU"
    if "cpu" in class_name:
        return "CPU"
    return str(device.__class__.__name__)


def _iter_nested_blocks(blocks: object) -> Iterable[object]:
    if blocks is None:
        return ()
    if isinstance(blocks, dict):
        return blocks.values()
    if isinstance(blocks, (list, tuple)):
        return blocks
    return ()


def _iter_program_operations(block: object) -> Iterable[object]:
    operations = getattr(block, "operations", None) or []
    for op in operations:
        yield op
        for nested in _iter_nested_blocks(getattr(op, "blocks", None)):
            yield from _iter_program_operations(nested)


def _op_output_names(op: object) -> list[str]:
    names: list[str] = []
    for output in getattr(op, "outputs", None) or []:
        name = getattr(output, "name", None)
        if name is None:
            continue
        names.append(str(name))
    return names


def _sorted_counter_items(counter: Counter[str], top_k: int | None = None) -> list[list[Any]]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ranked = ranked[: max(int(top_k), 0)]
    return [[name, int(count)] for name, count in ranked]


def analyze_compiled_model_placement(
    compiled_model_path: Path,
    *,
    compute_units: str = "all",
    top_k: int = 25,
    sample_limit: int = 100,
) -> dict[str, Any]:
    if not hasattr(ct.models, "compute_plan"):
        raise ComputePlanUnavailableError("coremltools compute_plan API is unavailable.")

    plan_api = ct.models.compute_plan
    if not hasattr(plan_api, "MLComputePlan"):
        raise ComputePlanUnavailableError("coremltools MLComputePlan API is unavailable.")

    compiled_model_path = Path(compiled_model_path).resolve()
    if compiled_model_path.suffix != ".mlmodelc":
        raise ValueError(
            f"Expected a compiled model directory ending with .mlmodelc, got {compiled_model_path}."
        )
    if not compiled_model_path.exists():
        raise FileNotFoundError(f"Compiled model not found: {compiled_model_path}")

    plan = plan_api.MLComputePlan.load_from_path(
        str(compiled_model_path),
        compute_units=_parse_compute_units(compute_units),
    )
    structure = plan.model_structure
    program = getattr(structure, "program", None)
    if program is None:
        raise ComputePlanUnavailableError(
            "Only ML Program compute-plan analysis is implemented in this repository."
        )

    preferred_device_counts: Counter[str] = Counter()
    fallback_operator_counts: Counter[str] = Counter()
    all_operator_counts: Counter[str] = Counter()
    fallback_samples: list[dict[str, Any]] = []
    function_summaries: dict[str, dict[str, Any]] = {}

    total_ops = 0
    fallback_ops = 0
    fallback_cost = 0.0
    known_cost = 0.0

    for fn_name, fn in (program.functions or {}).items():
        fn_preferred_counts: Counter[str] = Counter()
        fn_fallback_counts: Counter[str] = Counter()
        fn_total = 0
        fn_fallback = 0
        fn_fallback_cost = 0.0
        fn_known_cost = 0.0

        block = getattr(fn, "block", None)
        if block is None:
            function_summaries[str(fn_name)] = {
                "total_operations": 0,
                "fallback_operation_count": 0,
                "fallback_ratio": 0.0,
                "preferred_device_counts": {},
                "top_fallback_ops": [],
            }
            continue

        for op in _iter_program_operations(block):
            fn_total += 1
            total_ops += 1

            operator_name = str(getattr(op, "operator_name", "unknown"))
            all_operator_counts[operator_name] += 1

            device_usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred_device = _device_label(
                getattr(device_usage, "preferred_compute_device", None)
            )
            supported_devices = sorted(
                {_device_label(device) for device in (getattr(device_usage, "supported_compute_devices", None) or [])}
            )

            preferred_device_counts[preferred_device] += 1
            fn_preferred_counts[preferred_device] += 1

            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            cost_weight = float(getattr(cost, "weight", 0.0)) if cost is not None else None
            if cost_weight is not None:
                known_cost += cost_weight
                fn_known_cost += cost_weight

            if preferred_device != "ANE":
                fallback_ops += 1
                fn_fallback += 1
                fallback_operator_counts[operator_name] += 1
                fn_fallback_counts[operator_name] += 1
                if cost_weight is not None:
                    fallback_cost += cost_weight
                    fn_fallback_cost += cost_weight

                if len(fallback_samples) < max(int(sample_limit), 0):
                    fallback_samples.append(
                        {
                            "function": str(fn_name),
                            "operator": operator_name,
                            "preferred_device": preferred_device,
                            "supported_devices": supported_devices,
                            "outputs": _op_output_names(op),
                            "estimated_cost": cost_weight,
                        }
                    )

        fn_fallback_ratio = float(fn_fallback / fn_total) if fn_total > 0 else 0.0
        function_summaries[str(fn_name)] = {
            "total_operations": int(fn_total),
            "fallback_operation_count": int(fn_fallback),
            "fallback_ratio": round(fn_fallback_ratio, 6),
            "preferred_device_counts": dict(fn_preferred_counts),
            "top_fallback_ops": _sorted_counter_items(fn_fallback_counts, top_k=top_k),
            "fallback_cost_weight": round(fn_fallback_cost, 6),
            "known_cost_weight": round(fn_known_cost, 6),
        }

    fallback_ratio = float(fallback_ops / total_ops) if total_ops > 0 else 0.0
    fallback_cost_ratio = float(fallback_cost / known_cost) if known_cost > 0 else 0.0
    return {
        "compiled_model_path": str(compiled_model_path),
        "model_type": "mlprogram",
        "compute_units": str(compute_units).strip().lower(),
        "total_operations": int(total_ops),
        "fallback_operation_count": int(fallback_ops),
        "fallback_ratio": round(fallback_ratio, 6),
        "preferred_device_counts": dict(preferred_device_counts),
        "top_fallback_ops": _sorted_counter_items(fallback_operator_counts, top_k=top_k),
        "top_ops": _sorted_counter_items(all_operator_counts, top_k=top_k),
        "fallback_samples": fallback_samples,
        "fallback_cost_weight": round(fallback_cost, 6),
        "known_cost_weight": round(known_cost, 6),
        "fallback_cost_ratio": round(fallback_cost_ratio, 6),
        "functions": function_summaries,
    }
