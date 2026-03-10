from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import coremltools as ct
import numpy as np

from .from_mlx import capture_graph_from_mlx_function
from .ir import Graph, StateSpec, TensorSpec
from .lower_to_mil import build_mil_program, convert_program_to_model
from .op_registry import ensure_supported, unsupported_op_details
from .passes import infer_graph_specs, normalize_graph, summarize_inference


@dataclass(slots=True)
class ConversionConfig:
    capture_mode: str = "callback"
    allow_unknown_sources: bool = True
    capture_is_training: bool = False
    deployment_target: str | Any = "iOS18"
    target_profile: str | None = "default"
    convert_to: str = "mlprogram"
    compute_precision: str = "auto"
    compute_units: str = "all"
    state_specs: list[StateSpec] | None = None
    flex_input_lens: list[int] | None = None
    flex_input_names: set[str] = field(default_factory=set)


@dataclass(slots=True)
class CapturedMLXGraph:
    graph: Graph
    normalized_inputs: dict[str, np.ndarray]
    expected_outputs: dict[str, np.ndarray]


@dataclass(slots=True)
class PreparedMLXGraph:
    captured: CapturedMLXGraph
    normalized_graph: Graph
    expected_outputs: dict[str, np.ndarray]
    top_ops: list[list[Any]]
    inference_summary: dict[str, int]
    unsupported_details: list[dict[str, Any]]
    extra_input_names: list[str]

    @property
    def graph(self) -> Graph:
        return self.captured.graph

    @property
    def normalized_inputs(self) -> dict[str, np.ndarray]:
        return self.captured.normalized_inputs

    @property
    def weights_captured_as_constants(self) -> bool:
        return len(self.extra_input_names) == 0


@dataclass(slots=True)
class ConvertedCoreMLModel:
    prepared: PreparedMLXGraph
    program: Any
    model: Any
    conversion_inputs: list[Any] | None
    flex_input_shapes: dict[str, list[list[int]]]


def parse_deployment_target(target_name: str) -> Any:
    if not hasattr(ct.target, target_name):
        valid = [name for name in dir(ct.target) if name.startswith("iOS")]
        raise ValueError(
            f"Unknown deployment target: {target_name}. Valid examples: {', '.join(valid)}"
        )
    return getattr(ct.target, target_name)


def parse_flex_lengths(
    raw: str | None,
    *,
    preset_values: Sequence[int] | None = None,
    required_values: Sequence[int] = (),
) -> list[int] | None:
    if raw is None:
        return None

    text = str(raw).strip()
    if not text or text.lower() in {"auto", "preset"}:
        if preset_values is None:
            raise ValueError("--flex-input-lens requires explicit values for this conversion path.")
        values = [int(value) for value in preset_values]
    else:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]

    if not values:
        raise ValueError("--flex-input-lens did not contain any values.")

    normalized: list[int] = []
    for value in [*values, *[int(v) for v in required_values]]:
        if int(value) <= 0:
            raise ValueError(f"--flex-input-lens values must be positive, got {value}.")
        ivalue = int(value)
        if ivalue not in normalized:
            normalized.append(ivalue)
    return normalized


def parse_flex_input_names(raw: str) -> set[str]:
    return {name.strip() for name in str(raw).split(",") if name.strip()}


def load_state_specs(path: Path | None) -> list[StateSpec] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("states") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(
            f"--state-specs-json must contain a list or a dict with 'states' list, got {type(entries).__name__}."
        )

    specs: list[StateSpec] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"State spec at index {idx} must be an object, got {type(entry).__name__}.")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"State spec at index {idx} is missing non-empty 'name'.")
        shape_raw = entry.get("shape")
        if not isinstance(shape_raw, (list, tuple)):
            raise ValueError(f"State spec '{name}' must provide 'shape' as a list.")
        shape = tuple(int(v) for v in shape_raw)
        if any(int(v) <= 0 for v in shape):
            raise ValueError(f"State spec '{name}' has non-positive shape dimension(s): {shape}.")
        dtype = str(entry.get("dtype", "fp16")).strip().lower()
        specs.append(StateSpec(name=name, shape=shape, dtype=dtype))
    return specs


@contextmanager
def temporary_capture_training_mode(obj: Any, enabled: bool):
    if not bool(enabled):
        yield
        return

    prior_training: bool | None = None
    if hasattr(obj, "training"):
        try:
            prior_training = bool(getattr(obj, "training"))
        except Exception:
            prior_training = None

    prior_is_training: bool | None = None
    had_is_training = hasattr(obj, "is_training")
    if had_is_training:
        try:
            prior_is_training = bool(getattr(obj, "is_training"))
        except Exception:
            prior_is_training = None

    train_fn = getattr(obj, "train", None)
    eval_fn = getattr(obj, "eval", None)

    if callable(train_fn):
        train_fn()
    elif hasattr(obj, "training"):
        try:
            setattr(obj, "training", True)
        except Exception:
            pass

    if had_is_training:
        try:
            setattr(obj, "is_training", True)
        except Exception:
            pass

    try:
        yield
    finally:
        if prior_training is not None:
            if prior_training and callable(train_fn):
                train_fn()
            elif (not prior_training) and callable(eval_fn):
                eval_fn()
            elif hasattr(obj, "training"):
                try:
                    setattr(obj, "training", prior_training)
                except Exception:
                    pass

        if had_is_training and prior_is_training is not None:
            try:
                setattr(obj, "is_training", prior_is_training)
            except Exception:
                pass


def capture_mlx_graph(
    target: Any,
    inputs: Mapping[str, Any],
    *,
    dot_output_path: Path | None = None,
    capture_mode: str = "callback",
    allow_unknown_sources: bool = True,
    capture_is_training: bool = False,
    capture_function: Callable[..., Any] | None = None,
) -> CapturedMLXGraph:
    normalized_inputs = {name: np.asarray(value) for name, value in inputs.items()}
    resolved_capture_function, capture_target = _resolve_capture_components(target, capture_function)
    with temporary_capture_training_mode(capture_target, enabled=capture_is_training):
        graph, captured_inputs, expected_outputs = capture_graph_from_mlx_function(
            dot_output_path=dot_output_path,
            inputs=normalized_inputs,
            function=resolved_capture_function,
            allow_unknown_sources=allow_unknown_sources,
            capture_mode=capture_mode,
        )
    return CapturedMLXGraph(
        graph=graph,
        normalized_inputs=captured_inputs,
        expected_outputs=expected_outputs,
    )


def normalize_graph_for_conversion(
    graph: Graph,
    expected_outputs: dict[str, np.ndarray],
) -> tuple[Graph, dict[str, np.ndarray], list[list[Any]]]:
    normalized_graph = normalize_graph(graph)
    graph_ops = [node.op for node in normalized_graph.nodes]
    return normalized_graph, expected_outputs, top_ops(graph_ops)


def summarize_graph_inference(graph: Graph) -> dict[str, int]:
    inferred = infer_graph_specs(graph)
    return summarize_inference(inferred)


def collect_unsupported_details(graph: Graph) -> list[dict[str, Any]]:
    return unsupported_op_details(graph)


def ensure_graph_supported(graph: Graph) -> None:
    ensure_supported(graph)


def find_extra_input_names(
    graph: Graph,
    normalized_inputs: Mapping[str, Any],
) -> list[str]:
    input_names = set(normalized_inputs.keys())
    return [spec.name for spec in graph.inputs if spec.name not in input_names]


def lower_graph_to_mil(
    graph: Graph,
    *,
    config: ConversionConfig | None = None,
) -> Any:
    resolved = ConversionConfig() if config is None else config
    return build_mil_program(
        graph,
        deployment_target=resolve_deployment_target(resolved.deployment_target),
        normalize=False,
        target_profile=resolved.target_profile,
        shared_state_specs=resolved.state_specs,
    )


def build_conversion_inputs(
    input_specs: list[TensorSpec],
    *,
    flex_input_lens: list[int] | None,
    flex_input_names: set[str],
) -> tuple[list[Any] | None, dict[str, list[list[int]]]]:
    if flex_input_lens is None:
        return None, {}

    apply_to_all = len(flex_input_names) == 0 or "all" in flex_input_names
    converted_inputs: list[Any] = []
    applied_shapes: dict[str, list[list[int]]] = {}

    for spec in input_specs:
        base_shape = tuple(int(v) for v in spec.shape)
        dtype = tensor_spec_numpy_dtype(spec)
        allow_flex = apply_to_all or spec.name in flex_input_names

        if allow_flex and len(base_shape) >= 1:
            enumerated: list[tuple[int, ...]] = [base_shape]
            for seq_len in flex_input_lens:
                shape = list(base_shape)
                shape[-1] = int(seq_len)
                candidate = tuple(shape)
                if candidate not in enumerated:
                    enumerated.append(candidate)
            if len(enumerated) > 1:
                converted_inputs.append(
                    ct.TensorType(
                        name=spec.name,
                        shape=ct.EnumeratedShapes(shapes=enumerated),
                        dtype=dtype,
                    )
                )
                applied_shapes[spec.name] = [list(shape) for shape in enumerated]
                continue

        converted_inputs.append(ct.TensorType(name=spec.name, shape=base_shape, dtype=dtype))

    if not applied_shapes:
        raise ValueError(
            "--flex-input-lens was provided but no eligible inputs were found for flexible shapes."
        )

    return converted_inputs, applied_shapes


def convert_lowered_program(
    program: Any,
    input_specs: list[TensorSpec],
    *,
    config: ConversionConfig | None = None,
) -> tuple[Any, list[Any] | None, dict[str, list[list[int]]]]:
    resolved = ConversionConfig() if config is None else config
    conversion_inputs, flex_input_shapes = build_conversion_inputs(
        input_specs,
        flex_input_lens=resolved.flex_input_lens,
        flex_input_names=resolved.flex_input_names,
    )
    model = convert_program_to_model(
        program,
        deployment_target=resolve_deployment_target(resolved.deployment_target),
        convert_to=resolved.convert_to,
        compute_precision=resolved.compute_precision,
        compute_units=resolved.compute_units,
        inputs=conversion_inputs,
        state_specs=resolved.state_specs,
    )
    return model, conversion_inputs, flex_input_shapes


def prepare_mlx_conversion(
    target: Any,
    inputs: Mapping[str, Any],
    *,
    config: ConversionConfig | None = None,
    dot_output_path: Path | None = None,
    capture_function: Callable[..., Any] | None = None,
) -> PreparedMLXGraph:
    resolved = ConversionConfig() if config is None else config
    resolved_capture_function, capture_target = _resolve_capture_components(target, capture_function)
    captured = capture_mlx_graph(
        capture_target,
        inputs,
        dot_output_path=dot_output_path,
        capture_mode=resolved.capture_mode,
        allow_unknown_sources=resolved.allow_unknown_sources,
        capture_is_training=resolved.capture_is_training,
        capture_function=resolved_capture_function,
    )
    normalized_graph, expected_outputs, op_counts = normalize_graph_for_conversion(
        captured.graph,
        captured.expected_outputs,
    )
    inference_summary = summarize_graph_inference(normalized_graph)
    unsupported_details = collect_unsupported_details(normalized_graph)
    ensure_graph_supported(normalized_graph)
    extra_input_names = find_extra_input_names(normalized_graph, captured.normalized_inputs)
    return PreparedMLXGraph(
        captured=captured,
        normalized_graph=normalized_graph,
        expected_outputs=expected_outputs,
        top_ops=op_counts,
        inference_summary=inference_summary,
        unsupported_details=unsupported_details,
        extra_input_names=extra_input_names,
    )


def convert_mlx_to_coreml(
    target: Any,
    inputs: Mapping[str, Any],
    *,
    config: ConversionConfig | None = None,
    dot_output_path: Path | None = None,
    capture_function: Callable[..., Any] | None = None,
) -> ConvertedCoreMLModel:
    resolved_capture_function, capture_target = _resolve_capture_components(target, capture_function)
    prepared = prepare_mlx_conversion(
        capture_target,
        inputs,
        config=config,
        dot_output_path=dot_output_path,
        capture_function=resolved_capture_function,
    )
    program = lower_graph_to_mil(prepared.normalized_graph, config=config)
    model, conversion_inputs, flex_input_shapes = convert_lowered_program(
        program,
        prepared.normalized_graph.inputs,
        config=config,
    )
    return ConvertedCoreMLModel(
        prepared=prepared,
        program=program,
        model=model,
        conversion_inputs=conversion_inputs,
        flex_input_shapes=flex_input_shapes,
    )


def resolve_deployment_target(deployment_target: str | Any) -> Any:
    if isinstance(deployment_target, str):
        return parse_deployment_target(deployment_target)
    return deployment_target


def _resolve_capture_components(
    target: Any,
    capture_function: Callable[..., Any] | None,
) -> tuple[Callable[..., Any], Any]:
    if capture_function is None:
        if not callable(target):
            raise TypeError("target must be callable when capture_function is not provided.")
        capture_function = target
    return capture_function, target


def tensor_spec_numpy_dtype(spec: TensorSpec) -> Any:
    mapping = {
        "fp16": np.float16,
        "fp32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }
    if spec.dtype not in mapping:
        raise ValueError(
            f"Unsupported TensorSpec dtype '{spec.dtype}' for conversion inputs."
        )
    return mapping[spec.dtype]


def top_ops(graph_ops: list[str]) -> list[list[Any]]:
    ranked = sorted(Counter(graph_ops).items(), key=lambda item: (-item[1], item[0]))
    return [[name, int(count)] for name, count in ranked]


__all__ = [
    "CapturedMLXGraph",
    "ConversionConfig",
    "ConvertedCoreMLModel",
    "PreparedMLXGraph",
    "build_conversion_inputs",
    "capture_mlx_graph",
    "collect_unsupported_details",
    "convert_lowered_program",
    "convert_mlx_to_coreml",
    "ensure_graph_supported",
    "find_extra_input_names",
    "load_state_specs",
    "lower_graph_to_mil",
    "normalize_graph_for_conversion",
    "parse_deployment_target",
    "parse_flex_input_names",
    "parse_flex_lengths",
    "prepare_mlx_conversion",
    "resolve_deployment_target",
    "summarize_graph_inference",
    "temporary_capture_training_mode",
]
