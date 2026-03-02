from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from .ir import Graph, Node
from .op_registry import mil_op_for_mlx, normalize_mlx_op_name

_IDENTITY_OPS = {"identity", "stop_gradient", "copy", "contiguous"}
_CONSTANT_OPS = {"const", "constant", "literal"}
_SAFE_NAME_RE = re.compile(r"[^0-9a-zA-Z_]")
_MAX_INLINE_ARRAY_VALUES = 4096
_INPUT_DTYPE_ALIASES: dict[str, str] = {
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    "fp64": "fp64",
    "float64": "fp64",
    "double": "fp64",
    "int32": "int32",
    "int": "int32",
    "int64": "int64",
    "long": "int64",
    "bool": "bool",
}


@dataclass(frozen=True)
class InferredTensorSpec:
    shape: tuple[int, ...] | None
    dtype: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
        }


def _normalize_input_dtype(dtype: str) -> str:
    key = dtype.strip().lower()
    return _INPUT_DTYPE_ALIASES.get(key, key)


def canonicalize_input_specs(graph: Graph) -> Graph:
    inputs = [
        spec.__class__(
            name=str(spec.name).strip(),
            shape=tuple(int(v) for v in spec.shape),
            dtype=_normalize_input_dtype(spec.dtype),
        )
        for spec in graph.inputs
    ]
    normalized = Graph(inputs=inputs, nodes=list(graph.nodes), outputs=list(graph.outputs))
    normalized.validate()
    return normalized


def _sanitize_tensor_name(raw: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", raw.strip())
    if cleaned == "":
        cleaned = "t"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def canonicalize_tensor_names(graph: Graph) -> Graph:
    used: set[str] = set()
    name_map: dict[str, str] = {}

    def reserve(old: str) -> str:
        if old in name_map:
            return name_map[old]
        base = _sanitize_tensor_name(old)
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        name_map[old] = candidate
        return candidate

    inputs = [
        spec.__class__(
            name=reserve(spec.name),
            shape=spec.shape,
            dtype=spec.dtype,
        )
        for spec in graph.inputs
    ]
    nodes = []
    for node in graph.nodes:
        mapped_inputs = tuple(name_map.get(name, name) for name in node.inputs)
        mapped_output = reserve(node.output)
        nodes.append(
            Node(
                op=node.op,
                inputs=mapped_inputs,
                output=mapped_output,
                attrs=dict(node.attrs),
                source=node.source,
            )
        )

    outputs = [name_map.get(name, name) for name in graph.outputs]
    normalized = Graph(inputs=inputs, nodes=nodes, outputs=outputs)
    normalized.validate()
    return normalized


def _normalize_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        # Avoid exploding memory/time for large weight tensors during normalization.
        if value.size <= _MAX_INLINE_ARRAY_VALUES:
            return value.tolist()
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize_attr_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_attr_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_attr_value(v) for k, v in value.items()}
    return value


def canonicalize_constant_attrs(graph: Graph) -> Graph:
    nodes: list[Node] = []
    for node in graph.nodes:
        op = node.op
        attrs = {str(k): _normalize_attr_value(v) for k, v in node.attrs.items()}
        if op in _CONSTANT_OPS:
            op = "constant"
            if "value" not in attrs:
                for key in ("val", "data", "tensor"):
                    if key in attrs:
                        attrs["value"] = attrs.pop(key)
                        break
            dtype = attrs.get("dtype")
            if isinstance(dtype, str):
                attrs["dtype"] = _normalize_input_dtype(dtype)
        nodes.append(
            Node(
                op=op,
                inputs=node.inputs,
                output=node.output,
                attrs=attrs,
                source=node.source,
            )
        )

    normalized = Graph(inputs=list(graph.inputs), nodes=nodes, outputs=list(graph.outputs))
    normalized.validate()
    return normalized


def canonicalize_op_names(graph: Graph) -> Graph:
    nodes = [
        Node(
            op=normalize_mlx_op_name(node.op),
            inputs=node.inputs,
            output=node.output,
            attrs=dict(node.attrs),
            source=node.source,
        )
        for node in graph.nodes
    ]
    normalized = Graph(inputs=list(graph.inputs), nodes=nodes, outputs=list(graph.outputs))
    normalized.validate()
    return normalized


def eliminate_identity_noops(graph: Graph) -> Graph:
    replacements: dict[str, str] = {}
    kept_nodes: list[Node] = []
    graph_outputs = set(graph.outputs)

    def resolve(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    for node in graph.nodes:
        mapped_inputs = tuple(resolve(name) for name in node.inputs)
        canonical_node = Node(
            op=node.op,
            inputs=mapped_inputs,
            output=node.output,
            attrs=dict(node.attrs),
            source=node.source,
        )
        if (
            canonical_node.op in _IDENTITY_OPS
            and len(canonical_node.inputs) == 1
            and canonical_node.output not in graph_outputs
        ):
            replacements[canonical_node.output] = canonical_node.inputs[0]
            continue
        kept_nodes.append(canonical_node)

    outputs = [resolve(name) for name in graph.outputs]
    normalized = Graph(inputs=list(graph.inputs), nodes=kept_nodes, outputs=outputs)
    normalized.validate()
    return normalized


def _promote_dtype(lhs: str | None, rhs: str | None) -> str | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    rank = {"bool": 0, "int32": 1, "int64": 2, "fp16": 3, "fp32": 4, "fp64": 5}
    lhs_n = _normalize_input_dtype(lhs)
    rhs_n = _normalize_input_dtype(rhs)
    if lhs_n not in rank or rhs_n not in rank:
        return lhs_n if lhs_n == rhs_n else lhs_n
    return lhs_n if rank[lhs_n] >= rank[rhs_n] else rhs_n


def _infer_const_spec(node: Node) -> InferredTensorSpec:
    if "value" not in node.attrs:
        return InferredTensorSpec(shape=None, dtype=_normalize_input_dtype(str(node.attrs.get("dtype", "fp32"))))
    arr = np.asarray(node.attrs["value"])
    dtype = str(arr.dtype).lower()
    if "float16" in dtype:
        out_dtype = "fp16"
    elif "float64" in dtype:
        out_dtype = "fp64"
    elif "float" in dtype:
        out_dtype = "fp32"
    elif "int64" in dtype:
        out_dtype = "int64"
    elif "int" in dtype:
        out_dtype = "int32"
    elif "bool" in dtype:
        out_dtype = "bool"
    else:
        out_dtype = _normalize_input_dtype(str(node.attrs.get("dtype", "fp32")))
    return InferredTensorSpec(shape=tuple(int(v) for v in arr.shape), dtype=out_dtype)


def _infer_broadcast_shape(
    lhs: tuple[int, ...] | None,
    rhs: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    if lhs is None or rhs is None:
        return None
    out_rank = max(len(lhs), len(rhs))
    out: list[int] = []
    for axis in range(out_rank):
        li = axis - (out_rank - len(lhs))
        ri = axis - (out_rank - len(rhs))
        ld = lhs[li] if li >= 0 else 1
        rd = rhs[ri] if ri >= 0 else 1
        if ld == rd or ld == 1:
            out.append(rd)
        elif rd == 1:
            out.append(ld)
        else:
            return None
    return tuple(out)


def _normalize_axes_for_rank(axes_raw: Any, rank: int) -> list[int]:
    if axes_raw is None:
        return list(range(rank))
    if isinstance(axes_raw, int):
        axes = [int(axes_raw)]
    elif isinstance(axes_raw, (list, tuple)):
        axes = [int(v) for v in axes_raw]
    else:
        return list(range(rank))

    norm: list[int] = []
    for axis in axes:
        a = axis + rank if axis < 0 else axis
        if 0 <= a < rank and a not in norm:
            norm.append(a)
    return norm


def _reduced_shape(shape: tuple[int, ...] | None, axes_raw: Any, keep_dims: bool) -> tuple[int, ...] | None:
    if shape is None:
        return None
    rank = len(shape)
    axes = _normalize_axes_for_rank(axes_raw, rank)
    if keep_dims:
        return tuple(1 if i in axes else d for i, d in enumerate(shape))
    return tuple(d for i, d in enumerate(shape) if i not in axes)


def _infer_node_spec(node: Node, input_specs: list[InferredTensorSpec]) -> InferredTensorSpec:
    mil_op = mil_op_for_mlx(node.op)
    if mil_op is None:
        return InferredTensorSpec(shape=None, dtype=None)

    if mil_op == "const":
        return _infer_const_spec(node)

    if mil_op == "identity":
        return input_specs[0] if input_specs else InferredTensorSpec(shape=None, dtype=None)

    if mil_op in {
        "add",
        "sub",
        "mul",
        "real_div",
        "pow",
        "mod",
        "maximum",
        "minimum",
        "floor_div",
        "logaddexp",
    } and len(input_specs) == 2:
        return InferredTensorSpec(
            shape=_infer_broadcast_shape(input_specs[0].shape, input_specs[1].shape),
            dtype=_promote_dtype(input_specs[0].dtype, input_specs[1].dtype),
        )

    if mil_op in {"equal", "not_equal", "less", "less_equal", "greater", "greater_equal"} and len(input_specs) == 2:
        return InferredTensorSpec(
            shape=_infer_broadcast_shape(input_specs[0].shape, input_specs[1].shape),
            dtype="bool",
        )

    if mil_op in {"bitwisebinary"} and len(input_specs) == 2:
        return InferredTensorSpec(
            shape=_infer_broadcast_shape(input_specs[0].shape, input_specs[1].shape),
            dtype="bool",
        )

    if mil_op in {"matmul"} and len(input_specs) == 2:
        x_shape = input_specs[0].shape
        y_shape = input_specs[1].shape
        out_shape: tuple[int, ...] | None = None
        if x_shape is not None and y_shape is not None and len(x_shape) == 2 and len(y_shape) == 2:
            out_shape = (x_shape[0], y_shape[1])
        return InferredTensorSpec(shape=out_shape, dtype=_promote_dtype(input_specs[0].dtype, input_specs[1].dtype))

    if mil_op == "scaled_dot_product_attention" and len(input_specs) >= 3:
        q_shape = input_specs[0].shape
        v_shape = input_specs[2].shape
        out_shape: tuple[int, ...] | None = None
        if (
            q_shape is not None
            and v_shape is not None
            and len(q_shape) >= 3
            and len(v_shape) >= 3
        ):
            out_shape = tuple(list(q_shape[:-1]) + [v_shape[-1]])
        return InferredTensorSpec(shape=out_shape, dtype=input_specs[0].dtype)

    if mil_op in {"reduce_sum", "reduce_mean", "reduce_min", "reduce_max", "reduce_prod", "reduce_log_sum_exp"}:
        src = input_specs[0] if input_specs else InferredTensorSpec(shape=None, dtype=None)
        keep_dims = bool(node.attrs.get("keep_dims", False))
        return InferredTensorSpec(
            shape=_reduced_shape(src.shape, node.attrs.get("axes"), keep_dims),
            dtype=src.dtype,
        )

    if mil_op in {"reduce_argmax", "reduce_argmin"}:
        src = input_specs[0] if input_specs else InferredTensorSpec(shape=None, dtype=None)
        keep_dims = bool(node.attrs.get("keep_dims", False))
        axis = node.attrs.get("axis")
        if axis is None and "axes" in node.attrs:
            axes = node.attrs.get("axes")
            axis = axes[0] if isinstance(axes, (list, tuple)) and axes else None
        return InferredTensorSpec(
            shape=_reduced_shape(src.shape, axis, keep_dims),
            dtype="int32",
        )

    if mil_op in {"reshape", "flatten", "unflatten"}:
        shape_attr = node.attrs.get("shape")
        if isinstance(shape_attr, (list, tuple)):
            out_shape = tuple(int(v) for v in shape_attr)
            if -1 in out_shape and input_specs:
                src_shape = input_specs[0].shape
                if src_shape is not None:
                    known = 1
                    unknown_count = 0
                    for dim in out_shape:
                        if dim == -1:
                            unknown_count += 1
                        else:
                            known *= dim
                    if unknown_count == 1 and known != 0:
                        total = math.prod(src_shape)
                        fill = total // known if total % known == 0 else -1
                        out_shape = tuple(fill if dim == -1 else dim for dim in out_shape)
            return InferredTensorSpec(shape=out_shape, dtype=input_specs[0].dtype if input_specs else None)
        return InferredTensorSpec(shape=None, dtype=input_specs[0].dtype if input_specs else None)

    if mil_op == "transpose" and input_specs:
        src = input_specs[0]
        perm_attr = node.attrs.get("perm")
        if src.shape is None or not isinstance(perm_attr, (list, tuple)):
            return InferredTensorSpec(shape=None, dtype=src.dtype)
        perm = [int(v) for v in perm_attr]
        if len(src.shape) != len(perm):
            return InferredTensorSpec(shape=None, dtype=src.dtype)
        try:
            out_shape = tuple(src.shape[idx] for idx in perm)
        except IndexError:
            out_shape = None
        return InferredTensorSpec(shape=out_shape, dtype=src.dtype)

    if mil_op == "concat":
        if not input_specs:
            return InferredTensorSpec(shape=None, dtype=None)
        axis = int(node.attrs.get("axis", 0))
        dtype = input_specs[0].dtype
        shapes = [spec.shape for spec in input_specs]
        if any(shape is None for shape in shapes):
            return InferredTensorSpec(shape=None, dtype=dtype)
        shape0 = list(shapes[0])  # type: ignore[index]
        rank = len(shape0)
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank:
            return InferredTensorSpec(shape=None, dtype=dtype)
        total = 0
        for shape in shapes:  # type: ignore[assignment]
            assert shape is not None
            if len(shape) != rank:
                return InferredTensorSpec(shape=None, dtype=dtype)
            for i, dim in enumerate(shape):
                if i != axis and dim != shape0[i]:
                    return InferredTensorSpec(shape=None, dtype=dtype)
            total += shape[axis]
        shape0[axis] = total
        return InferredTensorSpec(shape=tuple(shape0), dtype=dtype)

    if mil_op == "cast":
        dtype = node.attrs.get("dtype")
        if isinstance(dtype, str):
            return InferredTensorSpec(
                shape=input_specs[0].shape if input_specs else None,
                dtype=_normalize_input_dtype(dtype),
            )
        return InferredTensorSpec(shape=input_specs[0].shape if input_specs else None, dtype=None)

    if mil_op in {"zeros", "ones", "full"}:
        shape_attr = node.attrs.get("shape")
        shape = tuple(int(v) for v in shape_attr) if isinstance(shape_attr, (list, tuple)) else None
        return InferredTensorSpec(shape=shape, dtype="fp32")

    if mil_op in {"zeros_like", "ones_like", "full_like"} and input_specs:
        return InferredTensorSpec(shape=input_specs[0].shape, dtype=input_specs[0].dtype)

    if mil_op == "number_of_elements":
        return InferredTensorSpec(shape=tuple(), dtype="int32")

    if mil_op == "arange":
        start = float(node.attrs.get("start", 0))
        end = node.attrs.get("end")
        step = float(node.attrs.get("step", 1))
        if end is None or step == 0:
            return InferredTensorSpec(shape=None, dtype="int32")
        n = max(0, int(math.ceil((float(end) - start) / step)))
        return InferredTensorSpec(shape=(n,), dtype="int32")

    if mil_op == "linspace":
        num = int(node.attrs.get("num", 50))
        dtype = _normalize_input_dtype(str(node.attrs.get("dtype", "fp32")))
        return InferredTensorSpec(shape=(max(0, num),), dtype=dtype)

    if mil_op == "select" and len(input_specs) == 3:
        return InferredTensorSpec(
            shape=_infer_broadcast_shape(input_specs[1].shape, input_specs[2].shape),
            dtype=_promote_dtype(input_specs[1].dtype, input_specs[2].dtype),
        )

    if mil_op in {"all", "any"} and input_specs:
        keep_dims = bool(node.attrs.get("keep_dims", False))
        return InferredTensorSpec(
            shape=_reduced_shape(input_specs[0].shape, node.attrs.get("axes"), keep_dims),
            dtype="bool",
        )

    if mil_op in {"array_equal", "allclose"}:
        return InferredTensorSpec(shape=tuple(), dtype="bool")

    if mil_op in {"isclose", "isnan", "isinf", "isfinite", "isneginf", "isposinf"} and input_specs:
        return InferredTensorSpec(shape=input_specs[0].shape, dtype="bool")

    if mil_op in {"diag", "diagonal", "trace", "tri", "tril", "triu", "eye"}:
        # These ops have multiple shape forms; keep dtype where possible and shape unknown for v1-light.
        dtype = input_specs[0].dtype if input_specs else _normalize_input_dtype(str(node.attrs.get("dtype", "fp32")))
        return InferredTensorSpec(shape=None, dtype=dtype)

    if mil_op in {"broadcast_to"} and input_specs:
        shape_attr = node.attrs.get("shape")
        shape = tuple(int(v) for v in shape_attr) if isinstance(shape_attr, (list, tuple)) else None
        return InferredTensorSpec(shape=shape, dtype=input_specs[0].dtype)

    if mil_op in {"broadcast_arrays"} and input_specs:
        shapes = [spec.shape for spec in input_specs]
        out_shape = None
        for shape in shapes:
            out_shape = _infer_broadcast_shape(out_shape, shape) if out_shape is not None else shape
        return InferredTensorSpec(shape=out_shape, dtype=input_specs[0].dtype)

    if mil_op in {"conv", "conv_transpose"} and input_specs:
        src = input_specs[0]
        dtype = src.dtype
        return InferredTensorSpec(shape=None, dtype=dtype)

    if mil_op in {"sigmoid", "softmax"} and input_specs:
        return InferredTensorSpec(shape=input_specs[0].shape, dtype=input_specs[0].dtype)

    if mil_op == "rmsnorm" and input_specs:
        return InferredTensorSpec(shape=input_specs[0].shape, dtype=input_specs[0].dtype)

    if mil_op == "rope" and input_specs:
        return InferredTensorSpec(shape=input_specs[0].shape, dtype=input_specs[0].dtype)

    # Default conservative fallback.
    return InferredTensorSpec(shape=None, dtype=input_specs[0].dtype if input_specs else None)


def infer_graph_specs(graph: Graph) -> dict[str, InferredTensorSpec]:
    graph.validate()
    inferred: dict[str, InferredTensorSpec] = {
        spec.name: InferredTensorSpec(
            shape=tuple(int(v) for v in spec.shape),
            dtype=_normalize_input_dtype(spec.dtype),
        )
        for spec in graph.inputs
    }
    for node in graph.nodes:
        input_specs = [inferred.get(name, InferredTensorSpec(shape=None, dtype=None)) for name in node.inputs]
        inferred[node.output] = _infer_node_spec(node, input_specs)
    return inferred


def summarize_inference(inferred: dict[str, InferredTensorSpec]) -> dict[str, int]:
    total = len(inferred)
    with_shape = sum(1 for spec in inferred.values() if spec.shape is not None)
    with_dtype = sum(1 for spec in inferred.values() if spec.dtype is not None)
    return {"total_tensors": total, "with_shape": with_shape, "with_dtype": with_dtype}


def normalize_graph(graph: Graph) -> Graph:
    graph.validate()
    canonical = canonicalize_op_names(graph)
    canonical = canonicalize_input_specs(canonical)
    canonical = canonicalize_tensor_names(canonical)
    canonical = canonicalize_constant_attrs(canonical)
    canonical = eliminate_identity_noops(canonical)
    return canonical
