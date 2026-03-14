from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types

from .ir import Graph, Node, StateSpec, TensorSpec
from .op_registry import ensure_supported, mil_op_for_mlx
from .passes import normalize_graph

_DTYPE_TO_MIL = {
    "fp16": types.fp16,
    "fp32": types.fp32,
    "int32": types.int32,
    "int64": types.int64,
    "bool": types.bool,
}

_DTYPE_TO_NUMPY = {
    "fp16": np.float16,
    "fp32": np.float32,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}

_CAST_DTYPE_ALIASES = {
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
class LoweringProfile:
    name: str
    linearize_matmul_rank_gt2: bool


@dataclass(frozen=True)
class MultifunctionBinding:
    model_path: Path
    source_function: str
    target_function: str


def resolve_lowering_profile(target_profile: str | None = None) -> LoweringProfile:
    key = str(target_profile or "default").strip().lower()
    if key in {"default", "ane_ios18"}:
        return LoweringProfile(name=key, linearize_matmul_rank_gt2=True)
    if key in {"conservative", "baseline"}:
        return LoweringProfile(name=key, linearize_matmul_rank_gt2=False)
    raise ValueError(
        f"Unsupported target_profile={target_profile!r}. "
        "Use one of: default, ane_ios18, conservative."
    )


def _metadata_to_string_map(metadata: dict[str, object]) -> dict[str, str]:
    encoded: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            encoded[str(key)] = value
        else:
            encoded[str(key)] = json.dumps(value, sort_keys=True)
    return encoded


def save_multifunction_package(
    *,
    bindings: list[MultifunctionBinding],
    destination_path: Path,
    default_function_name: str,
    metadata: dict[str, object] | None = None,
) -> Path:
    if not bindings:
        raise ValueError("save_multifunction_package requires at least one function binding.")
    if not str(default_function_name).strip():
        raise ValueError("default_function_name must be non-empty.")

    destination_path = Path(destination_path).resolve()
    descriptor = ct.utils.MultiFunctionDescriptor()
    for binding in bindings:
        descriptor.add_function(
            str(Path(binding.model_path).resolve()),
            str(binding.source_function),
            str(binding.target_function),
        )
    descriptor.default_function_name = str(default_function_name)
    ct.utils.save_multifunction(descriptor, str(destination_path))

    if metadata:
        mlmodel = ct.models.MLModel(str(destination_path), skip_model_load=True)
        for key, value in _metadata_to_string_map(metadata).items():
            mlmodel.user_defined_metadata[key] = value
        tmp_destination = destination_path.with_name(
            f"{destination_path.stem}.tmp_meta{destination_path.suffix}"
        )
        if tmp_destination.exists():
            shutil.rmtree(tmp_destination)
        mlmodel.save(str(tmp_destination))
        if tmp_destination.exists():
            if destination_path.exists():
                shutil.rmtree(destination_path)
            tmp_destination.replace(destination_path)

    return destination_path


def _to_tensor_spec(spec: TensorSpec) -> mb.TensorSpec:
    if spec.dtype not in _DTYPE_TO_MIL:
        raise ValueError(f"Unsupported input dtype: {spec.dtype}")
    # Core ML placeholders do not support rank-0 inputs. Lift scalar inputs to shape [1].
    shape = spec.shape if len(spec.shape) > 0 else (1,)
    return mb.TensorSpec(shape=shape, dtype=_DTYPE_TO_MIL[spec.dtype])


def _to_state_type(spec: StateSpec) -> ct.StateType:
    if spec.dtype not in _DTYPE_TO_NUMPY:
        raise ValueError(f"Unsupported state dtype: {spec.dtype}")
    # Keep consistent with input placeholder handling.
    shape = spec.shape if len(spec.shape) > 0 else (1,)
    wrapped = ct.TensorType(shape=shape, dtype=_DTYPE_TO_NUMPY[spec.dtype])
    return ct.StateType(wrapped_type=wrapped, name=spec.name)


def build_coreml_state_types(state_specs: list[StateSpec]) -> list[ct.StateType]:
    return [_to_state_type(spec) for spec in state_specs]


def _to_state_tensor_spec(spec: StateSpec) -> mb.StateTensorSpec:
    if spec.dtype not in _DTYPE_TO_MIL:
        raise ValueError(f"Unsupported state dtype: {spec.dtype}")
    shape = spec.shape if len(spec.shape) > 0 else (1,)
    return mb.StateTensorSpec(shape=shape, dtype=_DTYPE_TO_MIL[spec.dtype])


def _int_list(value: Iterable[int] | None, name: str, node: Node) -> list[int]:
    if value is None:
        raise ValueError(f"{node.op} node '{node.output}' requires '{name}' attr.")
    return [int(v) for v in value]


def _normalize_cast_dtype(dtype: str) -> str:
    key = dtype.strip().lower()
    if key not in _CAST_DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported cast dtype '{dtype}'. Supported keys: {', '.join(sorted(_CAST_DTYPE_ALIASES))}"
        )
    return _CAST_DTYPE_ALIASES[key]


def _static_shape_list(value: object, name: str, node: Node) -> list[int]:
    static_shape = _shape_list_if_static(value)
    shape = getattr(value, "shape", None)
    if static_shape is None:
        if shape is None:
            raise ValueError(f"{node.op} node '{node.output}' requires a tensor input for '{name}'.")
        raise ValueError(
            f"{node.op} node '{node.output}' requires static shape for '{name}', got {shape}."
        )
    return static_shape


def _shape_list_if_static(value: object) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    static_shape: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            static_shape.append(dim)
            continue
        try:
            static_shape.append(int(dim))
        except Exception:
            return None
    return static_shape


def _prod_ints(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _infer_broadcast_shape(shapes: list[list[int]], node: Node) -> list[int]:
    if not shapes:
        raise ValueError(f"{node.op} node '{node.output}' requires at least one input shape.")
    out_rank = max(len(shape) for shape in shapes)
    out: list[int] = []
    for axis in range(out_rank):
        dims = []
        for shape in shapes:
            offset = axis - (out_rank - len(shape))
            dims.append(shape[offset] if offset >= 0 else 1)
        non_ones = {dim for dim in dims if dim != 1}
        if len(non_ones) > 1:
            raise ValueError(
                f"{node.op} node '{node.output}' has incompatible broadcast dims at axis {axis}: {dims}"
            )
        out.append(next(iter(non_ones)) if non_ones else 1)
    return out


def _broadcast_shapes(lhs: list[int], rhs: list[int]) -> list[int] | None:
    out_rank = max(len(lhs), len(rhs))
    out: list[int] = []
    for axis in range(out_rank):
        lhs_offset = axis - (out_rank - len(lhs))
        rhs_offset = axis - (out_rank - len(rhs))
        lhs_dim = int(lhs[lhs_offset]) if lhs_offset >= 0 else 1
        rhs_dim = int(rhs[rhs_offset]) if rhs_offset >= 0 else 1
        if lhs_dim == rhs_dim or lhs_dim == 1:
            out.append(rhs_dim)
            continue
        if rhs_dim == 1:
            out.append(lhs_dim)
            continue
        return None
    return out


def _matmul_output_shape(
    lhs_shape: list[int],
    rhs_shape: list[int],
    *,
    transpose_lhs: bool,
    transpose_rhs: bool,
) -> list[int] | None:
    if len(lhs_shape) < 2 or len(rhs_shape) < 2:
        return None
    lhs_rows = int(lhs_shape[-1]) if transpose_lhs else int(lhs_shape[-2])
    lhs_inner = int(lhs_shape[-2]) if transpose_lhs else int(lhs_shape[-1])
    rhs_inner = int(rhs_shape[-1]) if transpose_rhs else int(rhs_shape[-2])
    rhs_cols = int(rhs_shape[-2]) if transpose_rhs else int(rhs_shape[-1])
    if lhs_inner != rhs_inner:
        return None
    batch = _broadcast_shapes(lhs_shape[:-2], rhs_shape[:-2])
    if batch is None:
        return None
    return batch + [lhs_rows, rhs_cols]


def _broadcast_tensor_to_shape(x: object, source_shape: list[int], target_shape: list[int], node: Node) -> object:
    x_expanded = x
    working_shape = list(source_shape)
    if len(working_shape) > len(target_shape):
        raise ValueError(
            f"{node.op} node '{node.output}' cannot broadcast rank-{len(working_shape)} "
            f"to rank-{len(target_shape)}."
        )
    if len(working_shape) < len(target_shape):
        prepend = len(target_shape) - len(working_shape)
        x_expanded = mb.expand_dims(
            x=x_expanded, axes=list(range(prepend)), name=f"{node.output}_expand"
        )
        working_shape = [1] * prepend + working_shape

    reps: list[int] = []
    for source_dim, target_dim in zip(working_shape, target_shape):
        if source_dim == target_dim:
            reps.append(1)
        elif source_dim == 1 and target_dim >= 1:
            reps.append(target_dim)
        else:
            raise ValueError(
                f"{node.op} node '{node.output}' has incompatible shape: "
                f"source {working_shape} -> target {target_shape}."
            )
    if all(rep == 1 for rep in reps):
        return mb.identity(x=x_expanded, name=node.output)
    return mb.tile(x=x_expanded, reps=reps, name=node.output)


def _normalize_axes(rank: int, axes: list[int], arg_name: str, node: Node) -> list[int]:
    normalized: list[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise ValueError(
                f"{node.op} node '{node.output}' has out-of-range axis {axis} for {arg_name} with rank {rank}."
            )
        if a in normalized:
            raise ValueError(
                f"{node.op} node '{node.output}' has duplicate axis {a} in {arg_name}."
            )
        normalized.append(a)
    return normalized


def _parse_tensordot_axes(node: Node, x_rank: int, y_rank: int) -> tuple[list[int], list[int]]:
    axes = node.attrs.get("axes", 2)
    if isinstance(axes, int):
        count = int(axes)
        if count < 0:
            raise ValueError(f"{node.op} node '{node.output}' requires non-negative integer axes.")
        if count > min(x_rank, y_rank):
            raise ValueError(
                f"{node.op} node '{node.output}' axes={count} exceeds input ranks {x_rank}, {y_rank}."
            )
        x_axes = list(range(x_rank - count, x_rank))
        y_axes = list(range(0, count))
        return x_axes, y_axes

    if not isinstance(axes, (list, tuple)) or len(axes) != 2:
        raise ValueError(
            f"{node.op} node '{node.output}' expects axes=int or pair of axis lists, got {axes!r}."
        )
    raw_x_axes, raw_y_axes = axes
    if isinstance(raw_x_axes, int):
        raw_x_axes = [raw_x_axes]
    if isinstance(raw_y_axes, int):
        raw_y_axes = [raw_y_axes]
    if not isinstance(raw_x_axes, (list, tuple)) or not isinstance(raw_y_axes, (list, tuple)):
        raise ValueError(
            f"{node.op} node '{node.output}' expects axes pair entries to be int/list, got {axes!r}."
        )

    x_axes = _normalize_axes(x_rank, [int(v) for v in raw_x_axes], "axes[0]", node)
    y_axes = _normalize_axes(y_rank, [int(v) for v in raw_y_axes], "axes[1]", node)
    if len(x_axes) != len(y_axes):
        raise ValueError(
            f"{node.op} node '{node.output}' requires same number of contraction axes, got {x_axes} and {y_axes}."
        )
    return x_axes, y_axes


def _dtype_name(value: object) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return ""
    if hasattr(dtype, "__name__"):
        return str(dtype.__name__).lower()
    return str(dtype).lower()


def _maybe_lower_matmul_as_linear(
    node: Node,
    x: object,
    y: object,
    *,
    allow_rank_gt2: bool,
) -> object | None:
    """
    Prefer MIL `linear` for matmul with compile-time constant RHS weights.
    This matches common projection patterns (flatten -> matmul -> unflatten)
    and helps ANE placement for projection-heavy blocks.
    """
    x_shape = getattr(x, "shape", None)
    y_shape = getattr(y, "shape", None)
    if x_shape is None or y_shape is None:
        return None
    if len(x_shape) < 2 or len(y_shape) != 2:
        return None

    x_dtype = _dtype_name(x)
    if not any(token in x_dtype for token in ("fp16", "float16", "fp32", "float32")):
        return None

    y_val = getattr(y, "val", None)
    if y_val is None:
        return None
    weight_rhs = np.asarray(y_val)
    if weight_rhs.ndim != 2:
        return None
    if not np.issubdtype(weight_rhs.dtype, np.floating):
        return None

    x_shape_static: list[int] = []
    try:
        for dim in x_shape:
            x_shape_static.append(int(dim))
    except Exception:  # pragma: no cover - symbolic dims fallback to plain matmul
        return None
    if len(x_shape_static) < 2:
        return None
    if len(x_shape_static) > 2 and not allow_rank_gt2:
        return None

    k_dim = int(x_shape_static[-1])
    if k_dim <= 0 or int(weight_rhs.shape[0]) != k_dim:
        return None

    # matmul(x[..., k], y[k, n]) == linear(x_2d, weight[n, k], bias=None) with reshape.
    weight = np.ascontiguousarray(weight_rhs.T)
    if len(x_shape_static) == 2:
        return mb.linear(x=x, weight=weight, bias=None, name=node.output)

    rows = _prod_ints(x_shape_static[:-1])
    cols = int(weight_rhs.shape[1])
    x_2d = mb.reshape(x=x, shape=[rows, k_dim], name=f"{node.output}_x2d")
    out_2d = mb.linear(x=x_2d, weight=weight, bias=None, name=f"{node.output}_linear")
    out_shape = [*x_shape_static[:-1], cols]
    return mb.reshape(x=out_2d, shape=out_shape, name=node.output)


def _nan_to_num_defaults(dtype_name: str) -> tuple[float, float, float] | None:
    if "fp16" in dtype_name or "float16" in dtype_name:
        info = np.finfo(np.float16)
    elif "fp32" in dtype_name or "float32" in dtype_name:
        info = np.finfo(np.float32)
    elif "fp64" in dtype_name or "float64" in dtype_name:
        info = np.finfo(np.float64)
    else:
        return None
    return 0.0, float(info.max), float(info.min)


def _build_isclose_mask(node: Node, x: object, y: object, prefix: str) -> object:
    rtol = float(node.attrs.get("rtol", 1e-5))
    atol = float(node.attrs.get("atol", 1e-8))
    equal_nan = bool(node.attrs.get("equal_nan", False))

    diff = mb.sub(x=x, y=y, name=f"{prefix}_diff")
    abs_diff = mb.abs(x=diff, name=f"{prefix}_abs_diff")
    abs_y = mb.abs(x=y, name=f"{prefix}_abs_y")
    rtol_term = mb.mul(x=rtol, y=abs_y, name=f"{prefix}_rtol_term")
    tol = mb.add(x=atol, y=rtol_term, name=f"{prefix}_tol")
    close = mb.less_equal(x=abs_diff, y=tol, name=f"{prefix}_close_base")
    equal = mb.equal(x=x, y=y, name=f"{prefix}_equal")
    close = mb.logical_or(x=close, y=equal, name=f"{prefix}_close_or_equal")

    if equal_nan:
        x_nan = mb.not_equal(x=x, y=x, name=f"{prefix}_x_nan")
        y_nan = mb.not_equal(x=y, y=y, name=f"{prefix}_y_nan")
        both_nan = mb.logical_and(x=x_nan, y=y_nan, name=f"{prefix}_both_nan")
        close = mb.logical_or(x=close, y=both_nan, name=f"{prefix}_close_nan")
    return close


def _diagonal_flat_indices(dim1: int, dim2: int, offset: int) -> tuple[np.ndarray, int]:
    row_start = max(-offset, 0)
    col_start = max(offset, 0)
    diag_len = min(dim1 - row_start, dim2 - col_start)
    if diag_len <= 0:
        raise ValueError(
            f"Diagonal has non-positive length for shape ({dim1}, {dim2}) and offset={offset}."
        )
    rows = np.arange(row_start, row_start + diag_len, dtype=np.int32)
    cols = np.arange(col_start, col_start + diag_len, dtype=np.int32)
    return (rows * np.int32(dim2) + cols).astype(np.int32), diag_len


def _extract_diagonal(
    x: object,
    x_shape: list[int],
    axis1: int,
    axis2: int,
    offset: int,
    name_prefix: str,
) -> object:
    if axis1 == axis2:
        raise ValueError(f"{name_prefix}: axis1 and axis2 must be different.")

    other_axes = [axis for axis in range(len(x_shape)) if axis not in (axis1, axis2)]
    perm = other_axes + [axis1, axis2]
    permuted = x if perm == list(range(len(x_shape))) else mb.transpose(x=x, perm=perm, name=f"{name_prefix}_perm")

    prefix_shape = [x_shape[axis] for axis in other_axes]
    dim1 = x_shape[axis1]
    dim2 = x_shape[axis2]
    flat_indices, diag_len = _diagonal_flat_indices(dim1, dim2, offset)

    prefix_size = _prod_ints(prefix_shape) if prefix_shape else 1
    flat = mb.reshape(x=permuted, shape=[prefix_size, dim1 * dim2], name=f"{name_prefix}_flat")
    gathered = mb.gather(x=flat, indices=flat_indices, axis=1, name=f"{name_prefix}_gather")
    out_shape = prefix_shape + [diag_len]
    return mb.reshape(x=gathered, shape=out_shape, name=f"{name_prefix}_diag")


def _diag_from_vector(x: object, x_shape: list[int], offset: int, node: Node) -> object:
    n = x_shape[0]
    out_size = n + abs(offset)
    if out_size <= 0:
        raise ValueError(f"diag node '{node.output}' produced non-positive output size.")

    if offset >= 0:
        rows = np.arange(n, dtype=np.int32)
        cols = rows + np.int32(offset)
    else:
        cols = np.arange(n, dtype=np.int32)
        rows = cols + np.int32(-offset)
    indices = np.stack([rows, cols], axis=1).astype(np.int32)

    dtype_name = _dtype_name(x)
    if "int" in dtype_name:
        zero_value: int | float = 0
    elif "bool" in dtype_name:
        raise ValueError(f"diag node '{node.output}' does not support bool vectors yet.")
    else:
        zero_value = 0.0
    data = mb.fill(shape=[out_size, out_size], value=zero_value, name=f"{node.output}_base")
    return mb.scatter_nd(
        data=data,
        indices=indices,
        updates=x,
        mode="update",
        name=node.output,
    )


def _zero_scalar_for_dtype_name(dtype_name: str) -> int | float | bool:
    if "bool" in dtype_name:
        return False
    if "int" in dtype_name:
        return 0
    return 0.0


def _build_tri_mask(rows: int, cols: int, k: int, lower: bool, prefix: str) -> object:
    row_idx = mb.range_1d(start=0, end=rows, step=1, name=f"{prefix}_rows")
    row_idx = mb.reshape(x=row_idx, shape=[rows, 1], name=f"{prefix}_rows_col")
    row_idx = mb.tile(x=row_idx, reps=[1, cols], name=f"{prefix}_rows_mat")

    col_idx = mb.range_1d(start=0, end=cols, step=1, name=f"{prefix}_cols")
    col_idx = mb.reshape(x=col_idx, shape=[1, cols], name=f"{prefix}_cols_row")
    col_idx = mb.tile(x=col_idx, reps=[rows, 1], name=f"{prefix}_cols_mat")

    cutoff = mb.add(x=row_idx, y=int(k), name=f"{prefix}_cutoff")
    if lower:
        return mb.less_equal(x=col_idx, y=cutoff, name=f"{prefix}_mask")
    return mb.greater_equal(x=col_idx, y=cutoff, name=f"{prefix}_mask")


def _bool_scalar(value: bool, name: str) -> object:
    vec = mb.fill(shape=[1], value=bool(value), name=f"{name}_vec")
    return mb.squeeze(x=vec, axes=[0], name=name)


def _as_bool_tensor(x: object, prefix: str) -> object:
    if "bool" in _dtype_name(x):
        return x
    return mb.not_equal(x=x, y=0, name=f"{prefix}_nonzero")


def _parse_reduction_axes(node: Node, rank: int, key: str = "axes") -> list[int]:
    axes_raw = node.attrs.get(key)
    if axes_raw is None:
        return list(range(rank))
    if isinstance(axes_raw, int):
        axes_raw = [int(axes_raw)]
    if not isinstance(axes_raw, (list, tuple)):
        raise ValueError(f"{node.op} node '{node.output}' expects '{key}' as int or list.")
    return _normalize_axes(rank, [int(v) for v in axes_raw], key, node)


def _reduction_element_count(x_shape: list[int], axes: list[int], node: Node) -> int:
    if len(axes) == 0:
        return 1
    count = 1
    for axis in axes:
        dim = int(x_shape[axis])
        if dim < 0:
            raise ValueError(
                f"{node.op} node '{node.output}' requires static non-negative dims, got {x_shape}."
            )
        count *= dim
    return count


def _reduce_bool_mask(mask: object, axes: list[int], keep_dims: bool, any_mode: bool, name: str) -> object:
    rank = len(mask.shape)
    if rank == 0 or len(axes) == 0:
        return mb.identity(x=mask, name=name)

    mask_i = mb.cast(x=mask, dtype="int32", name=f"{name}_i")
    if any_mode:
        summed = mb.reduce_sum(x=mask_i, axes=axes, keep_dims=keep_dims, name=f"{name}_sum")
        return mb.greater(x=summed, y=0, name=name)
    prod = mb.reduce_prod(x=mask_i, axes=axes, keep_dims=keep_dims, name=f"{name}_prod")
    return mb.cast(x=prod, dtype="bool", name=name)


def _to_float_for_inf_checks(x: object, prefix: str) -> object:
    dtype_name = _dtype_name(x)
    if "fp" in dtype_name or "float" in dtype_name:
        return x
    return mb.cast(x=x, dtype="fp32", name=f"{prefix}_fp32")


def _coerce_float_tensor(x: object, prefix: str) -> object:
    dtype_name = _dtype_name(x)
    if "fp" in dtype_name or "float" in dtype_name:
        return x
    return mb.cast(x=x, dtype="fp32", name=f"{prefix}_fp32")


def _lower_var_or_std(node: Node, x: object, compute_std: bool) -> object:
    x_work = _coerce_float_tensor(x, node.output)
    rank = len(x_work.shape)
    axes = _parse_reduction_axes(node, rank)
    keep_dims = bool(node.attrs.get("keep_dims", False))
    correction = float(node.attrs.get("correction", node.attrs.get("ddof", 0.0)))

    if len(axes) == 0:
        zeros = mb.mul(x=x_work, y=0.0, name=f"{node.output}_zeros")
        return mb.sqrt(x=zeros, name=node.output) if compute_std else mb.identity(
            x=zeros, name=node.output
        )

    mean = mb.reduce_mean(x=x_work, axes=axes, keep_dims=True, name=f"{node.output}_mean")
    centered = mb.sub(x=x_work, y=mean, name=f"{node.output}_centered")
    sq = mb.mul(x=centered, y=centered, name=f"{node.output}_sq")
    var_out = mb.reduce_mean(x=sq, axes=axes, keep_dims=keep_dims, name=f"{node.output}_var_pop")

    if correction != 0.0:
        x_shape = _static_shape_list(x_work, "x", node)
        n = _reduction_element_count(x_shape, axes, node)
        denom = float(n) - correction
        if denom <= 0.0:
            raise ValueError(
                f"{node.op} node '{node.output}' has invalid correction={correction} for reduction size {n}."
            )
        scale = float(n) / denom
        var_out = mb.mul(x=var_out, y=scale, name=f"{node.output}_var_scaled")

    if compute_std:
        return mb.sqrt(x=var_out, name=node.output)
    return mb.identity(x=var_out, name=node.output)


def _float_log_tensor(x: object, prefix: str) -> object:
    return _coerce_float_tensor(x, prefix)


def _slice_last_dim(x: object, x_shape: list[int], start: int, end: int, name: str) -> object:
    begin = [0] * len(x_shape)
    finish = list(x_shape)
    begin[-1] = int(start)
    finish[-1] = int(end)
    return mb.slice_by_index(x=x, begin=begin, end=finish, name=name)


def _lower_rope(node: Node, env: dict[str, object]) -> object:
    if len(node.inputs) < 1:
        raise ValueError(f"rope node '{node.output}' requires at least 1 input.")

    x_input = env[node.inputs[0]]
    x_dtype_name = _dtype_name(x_input)
    cast_back_to_fp16 = "fp16" in x_dtype_name or "float16" in x_dtype_name
    x = (
        mb.cast(x=x_input, dtype="fp32", name=f"{node.output}_x_fp32")
        if cast_back_to_fp16
        else x_input
    )
    x_shape = _static_shape_list(x, node.inputs[0], node)
    if len(x_shape) < 3:
        raise ValueError(
            f"rope node '{node.output}' expects input rank >= 3 with shape (B, *, T, D), got {x_shape}."
        )
    seq_len = int(x_shape[-2])
    feature_dim = int(x_shape[-1])
    if seq_len <= 0 or feature_dim <= 0:
        raise ValueError(
            f"rope node '{node.output}' requires positive sequence/feature dims, got {x_shape}."
        )

    dims_attr = node.attrs.get("dims")
    dims = int(dims_attr) if dims_attr is not None else feature_dim
    traditional = bool(node.attrs.get("traditional", False))
    scale = float(node.attrs.get("scale", 1.0))
    base_attr = node.attrs.get("base", 10000.0)
    base = None if base_attr is None else float(base_attr)

    offset_term: object | None = None
    if len(node.inputs) >= 2:
        offset = env[node.inputs[1]]
        offset_shape = _static_shape_list(offset, node.inputs[1], node)
        offset_fp32 = _coerce_float_tensor(offset, f"{node.output}_offset")
        if len(offset_shape) == 0:
            offset_term = offset_fp32
        elif len(offset_shape) == 1 and offset_shape[0] == 1:
            offset_term = mb.squeeze(x=offset_fp32, axes=[0], name=f"{node.output}_offset_scalar")
        else:
            raise ValueError(
                f"rope node '{node.output}' currently supports only scalar offset; got shape {offset_shape}."
            )

    freqs_tensor: object | None = None
    if len(node.inputs) >= 3:
        freqs_candidate = env[node.inputs[2]]
        freqs_shape = _static_shape_list(freqs_candidate, node.inputs[2], node)
        if len(freqs_shape) == 1:
            inferred_dims = int(freqs_shape[0]) * 2
            if dims_attr is None:
                dims = inferred_dims
            elif int(dims_attr) != inferred_dims:
                raise ValueError(
                    f"rope node '{node.output}' has dims={int(dims_attr)} but freqs imply dims={inferred_dims}."
                )
            freqs_tensor = _coerce_float_tensor(freqs_candidate, f"{node.output}_freqs")
        elif len(freqs_shape) == 0:
            # DOT capture may expose constant freqs as unknown scalar source; fall back to base path.
            freqs_tensor = None
        else:
            raise ValueError(
                f"rope node '{node.output}' expects freqs to be rank-1 if provided, got shape {freqs_shape}."
            )

    if dims <= 0 or dims > feature_dim or dims % 2 != 0:
        raise ValueError(
            f"rope node '{node.output}' requires even dims in (0, {feature_dim}], got dims={dims}."
        )
    half_dims = dims // 2

    positions = mb.range_1d(start=0, end=seq_len, step=1, name=f"{node.output}_positions_i")
    positions = mb.cast(x=positions, dtype="fp32", name=f"{node.output}_positions")
    if scale != 1.0:
        positions = mb.mul(x=positions, y=scale, name=f"{node.output}_positions_scaled")
    if offset_term is not None:
        positions = mb.add(x=positions, y=offset_term, name=f"{node.output}_positions_offset")
    positions_2d = mb.reshape(x=positions, shape=[seq_len, 1], name=f"{node.output}_positions_2d")

    if freqs_tensor is None:
        if base is None:
            base = 10000.0
        freq_values = np.power(
            np.float32(base),
            np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims),
        ).astype(np.float32)
        freqs_tensor = mb.const(val=freq_values, name=f"{node.output}_freqs_const")
    freqs_2d = mb.reshape(x=freqs_tensor, shape=[1, half_dims], name=f"{node.output}_freqs_2d")
    angles = mb.real_div(x=positions_2d, y=freqs_2d, name=f"{node.output}_angles")
    cos_angles = mb.cos(x=angles, name=f"{node.output}_cos")
    sin_angles = mb.sin(x=angles, name=f"{node.output}_sin")

    prefix_shape = [int(v) for v in x_shape[:-2]]
    trig_shape = [1] * len(prefix_shape) + [seq_len, half_dims]
    cos_b = mb.reshape(x=cos_angles, shape=trig_shape, name=f"{node.output}_cos_b")
    sin_b = mb.reshape(x=sin_angles, shape=trig_shape, name=f"{node.output}_sin_b")
    if any(dim != 1 for dim in prefix_shape):
        reps = prefix_shape + [1, 1]
        cos_b = mb.tile(x=cos_b, reps=reps, name=f"{node.output}_cos_tile")
        sin_b = mb.tile(x=sin_b, reps=reps, name=f"{node.output}_sin_tile")

    x_rot = _slice_last_dim(x=x, x_shape=x_shape, start=0, end=dims, name=f"{node.output}_x_rot")

    if traditional:
        pairs_shape = prefix_shape + [seq_len, half_dims, 2]
        x_pairs = mb.reshape(x=x_rot, shape=pairs_shape, name=f"{node.output}_pairs")
        pairs_rank = len(pairs_shape)

        begin_even = [0] * pairs_rank
        end_even = list(pairs_shape)
        end_even[-1] = 1
        x_even = mb.slice_by_index(
            x=x_pairs, begin=begin_even, end=end_even, name=f"{node.output}_pairs_even"
        )
        x_even = mb.squeeze(x=x_even, axes=[pairs_rank - 1], name=f"{node.output}_even")

        begin_odd = [0] * pairs_rank
        end_odd = list(pairs_shape)
        begin_odd[-1] = 1
        x_odd = mb.slice_by_index(
            x=x_pairs, begin=begin_odd, end=end_odd, name=f"{node.output}_pairs_odd"
        )
        x_odd = mb.squeeze(x=x_odd, axes=[pairs_rank - 1], name=f"{node.output}_odd")

        rot_even = mb.sub(
            x=mb.mul(x=x_even, y=cos_b, name=f"{node.output}_even_cos"),
            y=mb.mul(x=x_odd, y=sin_b, name=f"{node.output}_odd_sin"),
            name=f"{node.output}_rot_even",
        )
        rot_odd = mb.add(
            x=mb.mul(x=x_even, y=sin_b, name=f"{node.output}_even_sin"),
            y=mb.mul(x=x_odd, y=cos_b, name=f"{node.output}_odd_cos"),
            name=f"{node.output}_rot_odd",
        )
        rot_even = mb.expand_dims(x=rot_even, axes=[pairs_rank - 1], name=f"{node.output}_rot_even_e")
        rot_odd = mb.expand_dims(x=rot_odd, axes=[pairs_rank - 1], name=f"{node.output}_rot_odd_e")
        interleaved = mb.concat(
            values=[rot_even, rot_odd], axis=pairs_rank - 1, name=f"{node.output}_pairs_rot"
        )
        rotated = mb.reshape(
            x=interleaved,
            shape=prefix_shape + [seq_len, dims],
            name=f"{node.output}_rotated",
        )
    else:
        x_rot_shape = prefix_shape + [seq_len, dims]
        first = _slice_last_dim(
            x=x_rot,
            x_shape=x_rot_shape,
            start=0,
            end=half_dims,
            name=f"{node.output}_first",
        )
        second = _slice_last_dim(
            x=x_rot,
            x_shape=x_rot_shape,
            start=half_dims,
            end=dims,
            name=f"{node.output}_second",
        )
        rot_first = mb.sub(
            x=mb.mul(x=first, y=cos_b, name=f"{node.output}_first_cos"),
            y=mb.mul(x=second, y=sin_b, name=f"{node.output}_second_sin"),
            name=f"{node.output}_rot_first",
        )
        rot_second = mb.add(
            x=mb.mul(x=first, y=sin_b, name=f"{node.output}_first_sin"),
            y=mb.mul(x=second, y=cos_b, name=f"{node.output}_second_cos"),
            name=f"{node.output}_rot_second",
        )
        rotated = mb.concat(values=[rot_first, rot_second], axis=len(x_shape) - 1, name=f"{node.output}_rotated")

    final_name = f"{node.output}_fp32" if cast_back_to_fp16 else node.output
    if dims < feature_dim:
        tail = _slice_last_dim(
            x=x,
            x_shape=x_shape,
            start=dims,
            end=feature_dim,
            name=f"{node.output}_tail",
        )
        out = mb.concat(values=[rotated, tail], axis=len(x_shape) - 1, name=final_name)
    else:
        out = mb.identity(x=rotated, name=final_name)

    if cast_back_to_fp16:
        return mb.cast(x=out, dtype="fp16", name=node.output)
    return out


def _build_sdpa_causal_mask(
    *,
    q_shape: list[int],
    k_shape: list[int],
    node: Node,
) -> object:
    target_seq = int(q_shape[-2])
    source_seq = int(k_shape[-2])
    if target_seq <= 0 or source_seq <= 0:
        raise ValueError(
            f"scaled_dot_product_attention node '{node.output}' requires static positive "
            f"sequence dims for causal mask, got q={q_shape}, k={k_shape}."
        )
    return mb.const(
        val=np.tril(np.ones((target_seq, source_seq), dtype=np.bool_)),
        name=f"{node.output}_causal_mask",
    )


def _merge_sdpa_masks(explicit_mask: object, causal_mask: object, node: Node) -> object:
    mask_dtype = _dtype_name(explicit_mask)
    if "bool" in mask_dtype:
        return mb.logical_and(
            x=explicit_mask,
            y=causal_mask,
            name=f"{node.output}_mask_combined",
        )

    float_dtype = "fp16" if ("fp16" in mask_dtype or "float16" in mask_dtype) else "fp32"
    if any(token in mask_dtype for token in ("fp16", "float16", "fp32", "float32")):
        explicit_mask_f = explicit_mask
        if float_dtype == "fp32" and ("fp16" in mask_dtype or "float16" in mask_dtype):
            explicit_mask_f = mb.cast(
                x=explicit_mask,
                dtype="fp32",
                name=f"{node.output}_mask_fp",
            )
    else:
        explicit_mask_f = mb.cast(
            x=explicit_mask,
            dtype=float_dtype,
            name=f"{node.output}_mask_fp",
        )

    causal_f = mb.cast(
        x=causal_mask,
        dtype=float_dtype,
        name=f"{node.output}_causal_fp",
    )
    inv_causal = mb.sub(x=1.0, y=causal_f, name=f"{node.output}_causal_inv")
    causal_add = mb.mul(x=-3e4, y=inv_causal, name=f"{node.output}_causal_add")
    return mb.add(x=explicit_mask_f, y=causal_add, name=f"{node.output}_mask_combined")


def _lower_sdpa(node: Node, env: dict[str, object]) -> object:
    if len(node.inputs) < 3:
        raise ValueError(
            f"scaled_dot_product_attention node '{node.output}' requires at least 3 inputs (q, k, v)."
        )

    q = env[node.inputs[0]]
    k = env[node.inputs[1]]
    v = env[node.inputs[2]]
    mask = env[node.inputs[3]] if len(node.inputs) >= 4 else None

    has_sinks = bool(node.attrs.get("has_sinks", False))
    output_logsumexp = bool(node.attrs.get("output_logsumexp", False))
    do_causal = bool(node.attrs.get("do_causal", False))
    if has_sinks:
        raise ValueError(
            f"scaled_dot_product_attention node '{node.output}' has has_sinks=True, which is not supported yet."
        )
    if output_logsumexp:
        raise ValueError(
            f"scaled_dot_product_attention node '{node.output}' has output_logsumexp=True, "
            "which is training-only and unsupported in inference lowering."
        )
    if int(node.attrs.get("output_index", 0)) != 0:
        raise ValueError(
            f"scaled_dot_product_attention node '{node.output}' has output_index={node.attrs.get('output_index')}; "
            "only the attention output (index 0) is supported."
        )

    scale_attr = node.attrs.get("scale")
    scale = None if scale_attr is None else float(scale_attr)
    q_shape = _shape_list_if_static(q)
    k_shape = _shape_list_if_static(k)
    v_shape = _shape_list_if_static(v)
    default_scale: float | None = None
    if q_shape is not None:
        if q_shape[-1] <= 0:
            raise ValueError(
                f"scaled_dot_product_attention node '{node.output}' has invalid query embedding dim {q_shape[-1]}."
            )
        default_scale = float(1.0 / np.sqrt(float(q_shape[-1])))

    # Handle grouped-query attention by repeating key/value heads when needed.
    if q_shape is not None and k_shape is not None and v_shape is not None and len(q_shape) >= 3 and len(k_shape) >= 3 and len(v_shape) >= 3:
        q_heads = int(q_shape[-3])
        k_heads = int(k_shape[-3])
        v_heads = int(v_shape[-3])
        if k_heads != v_heads:
            raise ValueError(
                f"scaled_dot_product_attention node '{node.output}' requires matching "
                f"key/value head dims, got k={k_shape}, v={v_shape}."
            )
        if q_heads != k_heads:
            if k_heads <= 0 or q_heads <= 0 or (q_heads % k_heads) != 0:
                raise ValueError(
                    f"scaled_dot_product_attention node '{node.output}' cannot align grouped heads: "
                    f"q={q_shape}, k={k_shape}, v={v_shape}."
                )
            repeat_heads = q_heads // k_heads
            if repeat_heads > 1:
                # Llama-style GQA expects per-head repeat_interleave order
                # (0,0,0,0,1,1,1,1,...) rather than a tiled block order.
                repeat_indices = np.repeat(np.arange(k_heads, dtype=np.int32), repeat_heads)
                k = mb.gather(
                    x=k,
                    indices=repeat_indices,
                    axis=-3,
                    name=f"{node.output}_k_repeat_heads",
                )
                v = mb.gather(
                    x=v,
                    indices=repeat_indices,
                    axis=-3,
                    name=f"{node.output}_v_repeat_heads",
                )
                k_shape[-3] = q_heads
                v_shape[-3] = q_heads

    can_use_fused = hasattr(mb, "scaled_dot_product_attention")
    if can_use_fused:
        if scale is not None and default_scale is None:
            # We cannot prove custom scale equals the implicit default when symbolic dims are present.
            can_use_fused = False
        elif scale is not None and default_scale is not None and not np.isclose(
            scale, default_scale, rtol=1e-6, atol=1e-8
        ):
            can_use_fused = False

    merged_mask = mask
    if do_causal:
        if q_shape is None or k_shape is None:
            raise ValueError(
                f"scaled_dot_product_attention node '{node.output}' requires static shape for "
                "causal masking when do_causal=True."
            )
        causal_mask = _build_sdpa_causal_mask(q_shape=q_shape, k_shape=k_shape, node=node)
        if merged_mask is None:
            merged_mask = causal_mask
        else:
            merged_mask = _merge_sdpa_masks(merged_mask, causal_mask, node)

    if can_use_fused:
        kwargs: dict[str, object] = {
            "query": q,
            "key": k,
            "value": v,
            "name": node.output,
        }
        if merged_mask is not None:
            kwargs["attn_mask"] = merged_mask
        return mb.scaled_dot_product_attention(**kwargs)

    scores = mb.matmul(x=q, y=k, transpose_y=True, name=f"{node.output}_scores")
    if scale is None:
        if default_scale is None:
            raise ValueError(
                f"scaled_dot_product_attention node '{node.output}' requires explicit 'scale' "
                "when query head dimension is symbolic and fused SDPA path is unavailable."
            )
        scale_to_use = default_scale
    else:
        scale_to_use = float(scale)
    if not np.isclose(scale_to_use, 1.0, rtol=1e-6, atol=1e-8):
        scores = mb.mul(x=scores, y=float(scale_to_use), name=f"{node.output}_scores_scaled")

    if merged_mask is not None:
        mask_dtype = _dtype_name(merged_mask)
        if "bool" in mask_dtype:
            mask_f = mb.cast(x=merged_mask, dtype="fp32", name=f"{node.output}_mask_fp")
            inv_mask = mb.sub(x=1.0, y=mask_f, name=f"{node.output}_mask_inv")
            additive_mask = mb.mul(x=-3e4, y=inv_mask, name=f"{node.output}_mask_add")
            scores = mb.add(x=scores, y=additive_mask, name=f"{node.output}_scores_masked")
        else:
            scores = mb.add(x=scores, y=merged_mask, name=f"{node.output}_scores_masked")

    weights = mb.softmax(x=scores, axis=-1, name=f"{node.output}_weights")
    return mb.matmul(x=weights, y=v, name=node.output)


def _meshgrid_dims_and_axis(
    dims: list[int], input_index: int, indexing: str, node: Node
) -> tuple[list[int], int]:
    rank = len(dims)
    if input_index < 0 or input_index >= rank:
        raise ValueError(
            f"{node.op} node '{node.output}' input_index={input_index} is out of range for {rank} input(s)."
        )

    if indexing == "ij" or rank < 2:
        return list(dims), input_index
    if indexing != "xy":
        raise ValueError(
            f"{node.op} node '{node.output}' only supports indexing='xy' or indexing='ij', got {indexing!r}."
        )

    out_dims = [dims[1], dims[0], *dims[2:]]
    if input_index == 0:
        axis = 1
    elif input_index == 1:
        axis = 0
    else:
        axis = input_index
    return out_dims, axis


def _num_spatial_dims(x: object, node: Node) -> int:
    rank = len(x.shape)
    spatial = rank - 2
    if spatial < 1 or spatial > 3:
        raise ValueError(
            f"{node.op} node '{node.output}' requires input rank in [3,5], got rank={rank}."
        )
    return spatial


def _parse_spatial_attr(
    value: object | None, num_spatial_dims: int, name: str, node: Node
) -> list[int]:
    if value is None:
        return [1] * num_spatial_dims
    if isinstance(value, int):
        return [int(value)] * num_spatial_dims
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{node.op} node '{node.output}' expects '{name}' as int or list, got {value!r}."
        )
    parsed = [int(v) for v in value]
    if len(parsed) == 1:
        return parsed * num_spatial_dims
    if len(parsed) != num_spatial_dims:
        raise ValueError(
            f"{node.op} node '{node.output}' expects '{name}' length {num_spatial_dims}, got {parsed}."
        )
    return parsed


def _parse_conv_pad(value: object | None, num_spatial_dims: int, node: Node) -> list[int]:
    if value is None:
        return [0] * (2 * num_spatial_dims)
    if isinstance(value, int):
        v = int(value)
        return [v, v] * num_spatial_dims
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{node.op} node '{node.output}' expects pad/padding as int or list, got {value!r}."
        )
    parsed = [int(v) for v in value]
    if len(parsed) == num_spatial_dims:
        if node.op == "convolution":
            out: list[int] = []
            for total in parsed:
                before = int(total) // 2
                after = int(total) - before
                out.extend([before, after])
            return out
        out: list[int] = []
        for p in parsed:
            out.extend([p, p])
        return out
    if len(parsed) != 2 * num_spatial_dims:
        raise ValueError(
            f"{node.op} node '{node.output}' expects padding length {num_spatial_dims} or "
            f"{2 * num_spatial_dims}, got {parsed}."
        )
    return parsed


def _parse_conv_output_shape(
    raw_output_shape: object | None,
    x: object,
    weight: object,
    groups: int,
    num_spatial_dims: int,
    node: Node,
) -> list[int] | None:
    if raw_output_shape is None:
        return None
    if not isinstance(raw_output_shape, (list, tuple)):
        raise ValueError(
            f"{node.op} node '{node.output}' expects output_shape as list/tuple, got {raw_output_shape!r}."
        )
    parsed = [int(v) for v in raw_output_shape]
    if len(parsed) == num_spatial_dims + 2:
        return parsed
    if len(parsed) == num_spatial_dims:
        x_shape = _static_shape_list(x, "x", node)
        w_shape = _static_shape_list(weight, "weight", node)
        batch = int(x_shape[0])
        c_out = int(w_shape[1]) * int(groups)
        return [batch, c_out] + parsed
    raise ValueError(
        f"{node.op} node '{node.output}' expects output_shape length {num_spatial_dims} "
        f"or {num_spatial_dims + 2}, got {parsed}."
    )


def _lower_conv_op(node: Node, env: dict[str, object], transpose: bool) -> object:
    if len(node.inputs) not in {2, 3}:
        raise ValueError(
            f"{node.op} node '{node.output}' requires 2 or 3 inputs (x, weight[, bias])."
        )
    x = env[node.inputs[0]]
    weight = env[node.inputs[1]]
    bias = env[node.inputs[2]] if len(node.inputs) == 3 else None
    num_spatial_dims = _num_spatial_dims(x, node)

    strides = _parse_spatial_attr(
        node.attrs.get("strides", node.attrs.get("stride")),
        num_spatial_dims,
        "strides",
        node,
    )
    dilations = _parse_spatial_attr(
        node.attrs.get("dilations", node.attrs.get("dilation")),
        num_spatial_dims,
        "dilations",
        node,
    )
    groups = int(node.attrs.get("groups", 1))
    if groups <= 0:
        raise ValueError(f"{node.op} node '{node.output}' requires groups > 0, got {groups}.")

    raw_pad = node.attrs.get("pad", node.attrs.get("padding", node.attrs.get("pads")))
    pad_type = str(
        node.attrs.get("pad_type", "custom" if raw_pad is not None else "valid")
    ).strip().lower()
    if pad_type not in {"valid", "custom", "same", "same_lower"}:
        raise ValueError(
            f"{node.op} node '{node.output}' has unsupported pad_type '{pad_type}'."
        )
    pad = _parse_conv_pad(raw_pad, num_spatial_dims, node)

    # MLX callback capture frequently emits 1-D convolution in channels-last form:
    # x: [N, L, C]. For regular conv, weight is [O, K, C_in/groups].
    # For some depthwise captures, weight is [C, K, 1] with groups omitted.
    # Convert these cases to Core ML's channels-first conv and transpose back.
    x_shape_static = _shape_list_if_static(x)
    w_shape_static = _shape_list_if_static(weight)
    if (
        not transpose
        and num_spatial_dims == 1
        and x_shape_static is not None
        and w_shape_static is not None
        and len(x_shape_static) == 3
        and len(w_shape_static) == 3
    ):
        channels_first = int(x_shape_static[1])
        channels_last = int(x_shape_static[2])
        depthwise_channels_last = (
            int(w_shape_static[0]) == channels_last
            and int(w_shape_static[2]) == 1
            and (groups == 1 or groups == channels_last)
        )
        generic_channels_last = (
            int(w_shape_static[2]) * groups == channels_last
            and int(w_shape_static[1]) * groups != channels_first
        )
        if depthwise_channels_last or generic_channels_last:
            if depthwise_channels_last:
                groups = channels_last
            x_ncl = mb.transpose(x=x, perm=[0, 2, 1], name=f"{node.output}_x_ncl")

            weight_val = getattr(weight, "val", None)
            if weight_val is not None:
                w_oik = np.transpose(np.asarray(weight_val), (0, 2, 1))
                weight_ncl = mb.const(val=w_oik, name=f"{node.output}_w_oik")
            else:
                weight_ncl = mb.transpose(x=weight, perm=[0, 2, 1], name=f"{node.output}_w_oik")

            conv_ncl = mb.conv(
                x=x_ncl,
                weight=weight_ncl,
                bias=bias,
                strides=strides,
                pad_type=pad_type,
                pad=pad,
                dilations=dilations,
                groups=groups,
                name=f"{node.output}_ncl",
            )
            return mb.transpose(x=conv_ncl, perm=[0, 2, 1], name=node.output)

    if transpose:
        output_shape = _parse_conv_output_shape(
            node.attrs.get("output_shape"),
            x=x,
            weight=weight,
            groups=groups,
            num_spatial_dims=num_spatial_dims,
            node=node,
        )
        kwargs = {
            "x": x,
            "weight": weight,
            "bias": None,
            "pad": pad,
            "output_shape": output_shape,
            "pad_type": pad_type,
            "strides": strides,
            "dilations": dilations,
            "groups": groups,
            "name": node.output if bias is None else f"{node.output}_convt",
        }
        out = mb.conv_transpose(**kwargs)
        if bias is None:
            return out
        bias_shape = [1, -1] + [1] * num_spatial_dims
        bias_term = mb.reshape(x=bias, shape=bias_shape, name=f"{node.output}_bias")
        return mb.add(x=out, y=bias_term, name=node.output)

    return mb.conv(
        x=x,
        weight=weight,
        bias=bias,
        strides=strides,
        pad_type=pad_type,
        pad=pad,
        dilations=dilations,
        groups=groups,
        name=node.output,
    )


def _lower_node(
    node: Node,
    env: dict[str, object],
    profile: LoweringProfile,
) -> object:
    mil_op = mil_op_for_mlx(node.op)
    if mil_op is None:
        raise ValueError(f"Unsupported op while lowering: {node.op}")

    if mil_op == "const":
        if "value" not in node.attrs:
            raise ValueError(f"constant node '{node.output}' requires 'value' attr.")
        return mb.const(val=node.attrs["value"], name=node.output)

    if mil_op == "read_state":
        if len(node.inputs) != 1:
            raise ValueError(f"read_state node '{node.output}' requires exactly 1 state input.")
        return mb.read_state(input=env[node.inputs[0]], name=node.output)
    if mil_op == "coreml_update_state":
        if len(node.inputs) != 2:
            raise ValueError(
                f"write_state node '{node.output}' requires exactly 2 inputs (state, value)."
            )
        return mb.coreml_update_state(
            state=env[node.inputs[0]],
            value=env[node.inputs[1]],
            name=node.output,
        )
    if mil_op == "state_update_masked":
        if len(node.inputs) != 3:
            raise ValueError(
                f"state_update_masked node '{node.output}' requires 3 inputs (state, value, mask)."
            )
        state = env[node.inputs[0]]
        value = env[node.inputs[1]]
        mask = env[node.inputs[2]]
        current = mb.read_state(input=state, name=f"{node.output}_current")
        mask_dtype = _dtype_name(mask)
        mask_bool = mask if "bool" in mask_dtype else mb.cast(
            x=mask, dtype="bool", name=f"{node.output}_mask_bool"
        )
        merged = mb.select(cond=mask_bool, a=value, b=current, name=f"{node.output}_merged")
        return mb.coreml_update_state(state=state, value=merged, name=node.output)

    if mil_op == "matmul":
        x = env[node.inputs[0]]
        y = env[node.inputs[1]]
        linear = _maybe_lower_matmul_as_linear(
            node,
            x,
            y,
            allow_rank_gt2=profile.linearize_matmul_rank_gt2,
        )
        if linear is not None:
            return linear
        return mb.matmul(x=x, y=y, name=node.output)
    if mil_op == "add":
        return mb.add(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "maximum":
        return mb.maximum(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "sub":
        return mb.sub(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "mul":
        return mb.mul(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "real_div":
        return mb.real_div(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "pow":
        return mb.pow(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "inverse":
        epsilon = float(node.attrs.get("epsilon", 1e-4))
        return mb.inverse(x=env[node.inputs[0]], epsilon=epsilon, name=node.output)
    if mil_op == "mod":
        return mb.mod(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "rope":
        return _lower_rope(node, env)
    if mil_op == "scaled_dot_product_attention":
        return _lower_sdpa(node, env)
    if mil_op == "softmax":
        axis = int(node.attrs.get("axis", -1))
        return mb.softmax(x=env[node.inputs[0]], axis=axis, name=node.output)
    if mil_op == "sigmoid":
        return mb.sigmoid(x=env[node.inputs[0]], name=node.output)
    if mil_op == "tanh":
        return mb.tanh(x=_coerce_float_tensor(env[node.inputs[0]], node.output), name=node.output)
    if mil_op == "erf":
        return mb.erf(x=_coerce_float_tensor(env[node.inputs[0]], node.output), name=node.output)
    if mil_op == "layernorm":
        if len(node.inputs) < 2:
            raise ValueError(
                f"layernorm node '{node.output}' requires at least 2 inputs (x, weight)."
            )
        eps = float(node.attrs.get("eps", node.attrs.get("epsilon", 1e-5)))
        x = env[node.inputs[0]]
        w = env[node.inputs[1]]
        x_work = mb.cast(x=x, dtype="fp32", name=f"{node.output}_x_fp32")
        w_work = mb.cast(x=w, dtype="fp32", name=f"{node.output}_w_fp32")
        rank = len(x_work.shape)
        if rank == 0:
            raise ValueError(f"layernorm node '{node.output}' requires rank >= 1 input.")
        axes = [rank - 1]
        mean = mb.reduce_mean(x=x_work, axes=axes, keep_dims=True, name=f"{node.output}_mean")
        centered = mb.sub(x=x_work, y=mean, name=f"{node.output}_centered")
        sq = mb.mul(x=centered, y=centered, name=f"{node.output}_sq")
        var = mb.reduce_mean(x=sq, axes=axes, keep_dims=True, name=f"{node.output}_var")
        denom = mb.add(x=var, y=eps, name=f"{node.output}_denom")
        inv = mb.rsqrt(x=denom, name=f"{node.output}_inv")
        norm = mb.mul(x=centered, y=inv, name=f"{node.output}_norm")
        out = mb.mul(x=norm, y=w_work, name=f"{node.output}_scaled")
        if len(node.inputs) >= 3:
            b = mb.cast(x=env[node.inputs[2]], dtype="fp32", name=f"{node.output}_b_fp32")
            out = mb.add(x=out, y=b, name=f"{node.output}_biased")
        x_dtype = _dtype_name(x)
        if "fp16" in x_dtype or "float16" in x_dtype:
            return mb.cast(x=out, dtype="fp16", name=node.output)
        return mb.identity(x=out, name=node.output)
    if mil_op == "rmsnorm":
        if len(node.inputs) < 2:
            raise ValueError(f"rmsnorm node '{node.output}' requires at least 2 inputs (x, weight).")
        eps = float(node.attrs.get("eps", node.attrs.get("epsilon", 1e-5)))
        x = env[node.inputs[0]]
        w = env[node.inputs[1]]
        # Match MLX RMSNorm behavior: accumulate in fp32 even when inputs are fp16.
        x_work = mb.cast(x=x, dtype="fp32", name=f"{node.output}_x_fp32")
        w_work = mb.cast(x=w, dtype="fp32", name=f"{node.output}_w_fp32")
        sq = mb.mul(x=x_work, y=x_work, name=f"{node.output}_sq")
        mean_sq = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True, name=f"{node.output}_mean_sq")
        denom = mb.add(x=mean_sq, y=eps, name=f"{node.output}_denom")
        inv = mb.rsqrt(x=denom, name=f"{node.output}_inv")
        norm = mb.mul(x=x_work, y=inv, name=f"{node.output}_norm")
        out = mb.mul(x=norm, y=w_work, name=f"{node.output}_scaled")
        if len(node.inputs) >= 3:
            b = mb.cast(x=env[node.inputs[2]], dtype="fp32", name=f"{node.output}_b_fp32")
            out = mb.add(x=out, y=b, name=f"{node.output}_biased")
        x_dtype = _dtype_name(x)
        if "fp16" in x_dtype or "float16" in x_dtype:
            return mb.cast(x=out, dtype="fp16", name=node.output)
        return mb.identity(x=out, name=node.output)
    if mil_op == "greater":
        return mb.greater(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "greater_equal":
        return mb.greater_equal(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "less":
        return mb.less(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "bitwisebinary":
        mode = int(node.attrs.get("mode", 0))
        x = env[node.inputs[0]]
        y = env[node.inputs[1]]
        if mode == 0:
            return mb.logical_and(x=x, y=y, name=node.output)
        if mode == 1:
            return mb.logical_or(x=x, y=y, name=node.output)
        raise ValueError(
            f"bitwisebinary node '{node.output}' has unsupported mode={mode}. "
            "Supported modes: 0 (and), 1 (or)."
        )
    if mil_op == "reduce":
        if len(node.inputs) != 1:
            raise ValueError(f"reduce node '{node.output}' requires exactly 1 input.")
        mode = int(node.attrs.get("mode", 2))
        x = env[node.inputs[0]]
        rank = len(x.shape)
        axes = _parse_reduction_axes(node, rank)
        keep_dims = bool(node.attrs.get("keep_dims", True))
        if mode == 0:
            mask = _as_bool_tensor(x, node.output)
            return _reduce_bool_mask(mask, axes, keep_dims, any_mode=False, name=node.output)
        if mode == 1:
            mask = _as_bool_tensor(x, node.output)
            return _reduce_bool_mask(mask, axes, keep_dims, any_mode=True, name=node.output)
        if mode == 2:
            return mb.reduce_sum(x=x, axes=axes, keep_dims=keep_dims, name=node.output)
        if mode == 3:
            return mb.reduce_prod(x=x, axes=axes, keep_dims=keep_dims, name=node.output)
        if mode == 4:
            return mb.reduce_min(x=x, axes=axes, keep_dims=keep_dims, name=node.output)
        if mode == 5:
            return mb.reduce_max(x=x, axes=axes, keep_dims=keep_dims, name=node.output)
        raise ValueError(
            f"reduce node '{node.output}' has unsupported mode={mode}. "
            "Supported modes: 0 (all), 1 (any), 2 (sum), 3 (prod), 4 (min), 5 (max)."
        )
    if mil_op in {
        "reduce_sum",
        "reduce_mean",
        "reduce_min",
        "reduce_max",
        "reduce_prod",
        "reduce_log_sum_exp",
    }:
        axes = node.attrs.get("axes")
        keep_dims = bool(node.attrs.get("keep_dims", False))
        op = getattr(mb, mil_op)
        return op(x=env[node.inputs[0]], axes=axes, keep_dims=keep_dims, name=node.output)
    if mil_op in {"reduce_argmax", "reduce_argmin"}:
        axis = node.attrs.get("axis")
        if axis is None:
            axes = node.attrs.get("axes")
            if axes is not None:
                if len(axes) != 1:
                    raise ValueError(
                        f"{mil_op} expects a single axis. Got axes={axes} for node {node.output}"
                    )
                axis = int(axes[0])
        if axis is None:
            raise ValueError(f"{mil_op} requires 'axis' (or single-entry 'axes') attr.")
        keep_dims = bool(node.attrs.get("keep_dims", False))
        output_dtype = str(node.attrs.get("output_dtype", "int32"))
        op = getattr(mb, mil_op)
        return op(
            x=env[node.inputs[0]],
            axis=int(axis),
            keep_dims=keep_dims,
            output_dtype=output_dtype,
            name=node.output,
        )
    if mil_op == "reshape":
        shape = _int_list(node.attrs.get("shape"), "shape", node)
        return mb.reshape(x=env[node.inputs[0]], shape=shape, name=node.output)
    if mil_op == "flatten":
        shape = node.attrs.get("shape")
        if shape is None:
            shape = [-1]
        return mb.reshape(x=env[node.inputs[0]], shape=[int(v) for v in shape], name=node.output)
    if mil_op == "unflatten":
        shape = _int_list(node.attrs.get("shape"), "shape", node)
        return mb.reshape(x=env[node.inputs[0]], shape=shape, name=node.output)
    if mil_op == "transpose":
        perm = _int_list(node.attrs.get("perm"), "perm", node)
        return mb.transpose(x=env[node.inputs[0]], perm=perm, name=node.output)
    if mil_op == "expand_dims":
        if len(node.inputs) != 1:
            raise ValueError(f"expand_dims node '{node.output}' requires exactly 1 input.")
        axes_raw = node.attrs.get("axes", node.attrs.get("axis"))
        if axes_raw is None:
            raise ValueError(f"expand_dims node '{node.output}' requires 'axes' (or 'axis') attr.")
        if isinstance(axes_raw, int):
            axes = [int(axes_raw)]
        elif isinstance(axes_raw, (list, tuple)):
            axes = [int(v) for v in axes_raw]
        else:
            raise ValueError(
                f"expand_dims node '{node.output}' expects axes as int/list, got {axes_raw!r}."
            )
        return mb.expand_dims(x=env[node.inputs[0]], axes=axes, name=node.output)
    if mil_op == "atleast_1d":
        x = env[node.inputs[0]]
        if len(x.shape) >= 1:
            return mb.identity(x=x, name=node.output)
        return mb.reshape(x=x, shape=[1], name=node.output)
    if mil_op == "atleast_2d":
        x = env[node.inputs[0]]
        rank = len(x.shape)
        if rank >= 2:
            return mb.identity(x=x, name=node.output)
        if rank == 1:
            return mb.expand_dims(x=x, axes=[0], name=node.output)
        return mb.reshape(x=x, shape=[1, 1], name=node.output)
    if mil_op == "atleast_3d":
        x = env[node.inputs[0]]
        rank = len(x.shape)
        if rank >= 3:
            return mb.identity(x=x, name=node.output)
        if rank == 2:
            return mb.expand_dims(x=x, axes=[2], name=node.output)
        if rank == 1:
            return mb.expand_dims(x=x, axes=[0, 2], name=node.output)
        return mb.reshape(x=x, shape=[1, 1, 1], name=node.output)
    if mil_op == "moveaxis":
        x = env[node.inputs[0]]
        rank = len(x.shape)
        source = node.attrs.get("source")
        destination = node.attrs.get("destination")
        if source is None or destination is None:
            raise ValueError(
                f"moveaxis node '{node.output}' requires both 'source' and 'destination' attrs."
            )
        source = int(source)
        destination = int(destination)
        if source < 0:
            source += rank
        if destination < 0:
            destination += rank
        perm = [i for i in range(rank) if i != source]
        perm.insert(destination, source)
        return mb.transpose(x=x, perm=perm, name=node.output)
    if mil_op == "swapaxes":
        x = env[node.inputs[0]]
        rank = len(x.shape)
        axis1 = int(node.attrs.get("axis1"))
        axis2 = int(node.attrs.get("axis2"))
        if axis1 < 0:
            axis1 += rank
        if axis2 < 0:
            axis2 += rank
        perm = list(range(rank))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
        return mb.transpose(x=x, perm=perm, name=node.output)
    if mil_op == "slice_by_index":
        begin = _int_list(node.attrs.get("begin"), "begin", node)
        end = _int_list(node.attrs.get("end"), "end", node)
        stride = node.attrs.get("stride")
        if stride is not None:
            stride = [int(v) for v in stride]
        return mb.slice_by_index(
            x=env[node.inputs[0]],
            begin=begin,
            end=end,
            stride=stride,
            name=node.output,
        )
    if mil_op == "slice_update":
        if len(node.inputs) != 2:
            raise ValueError(
                f"slice_update node '{node.output}' requires exactly 2 inputs (x, update)."
            )
        begin = _int_list(node.attrs.get("begin"), "begin", node)
        end = _int_list(node.attrs.get("end"), "end", node)
        stride = node.attrs.get("stride")
        if stride is not None:
            stride = [int(v) for v in stride]

        begin_mask = node.attrs.get("begin_mask")
        if begin_mask is not None:
            begin_mask = [bool(v) for v in begin_mask]
        end_mask = node.attrs.get("end_mask")
        if end_mask is not None:
            end_mask = [bool(v) for v in end_mask]
        squeeze_mask = node.attrs.get("squeeze_mask")
        if squeeze_mask is not None:
            squeeze_mask = [bool(v) for v in squeeze_mask]

        return mb.slice_update(
            x=env[node.inputs[0]],
            update=env[node.inputs[1]],
            begin=begin,
            end=end,
            stride=stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            squeeze_mask=squeeze_mask,
            name=node.output,
        )
    if mil_op == "split":
        if len(node.inputs) != 1:
            raise ValueError(f"split node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        rank = len(x.shape)
        axis = int(node.attrs.get("axis", 0))
        axis = _normalize_axes(rank, [axis], "axis", node)[0]
        output_index = int(node.attrs.get("output_index", 0))

        def _parse_split_sizes() -> list[int] | None:
            explicit = node.attrs.get("split_sizes")
            if isinstance(explicit, (list, tuple)):
                parsed = [int(v) for v in explicit]
                if any(v < 0 for v in parsed):
                    raise ValueError(
                        f"split node '{node.output}' requires non-negative split_sizes, got {parsed}."
                    )
                return parsed
            return None

        split_sizes = _parse_split_sizes()
        num_splits_attr = node.attrs.get("num_splits", node.attrs.get("num_outputs"))
        num_splits = int(num_splits_attr) if num_splits_attr is not None else None
        if num_splits is not None and num_splits <= 0:
            raise ValueError(f"split node '{node.output}' requires num_splits > 0, got {num_splits}.")

        if split_sizes is None:
            split_indices = node.attrs.get("split_indices")
            if isinstance(split_indices, (list, tuple)):
                x_shape_static = _shape_list_if_static(x)
                if x_shape_static is None:
                    raise ValueError(
                        f"split node '{node.output}' requires static input shape when using split_indices."
                    )
                axis_dim = int(x_shape_static[axis])
                indices = [int(v) for v in split_indices]
                prev = 0
                split_sizes = []
                for idx in indices:
                    if idx < prev or idx > axis_dim:
                        raise ValueError(
                            f"split node '{node.output}' has invalid split index {idx} for axis size {axis_dim}."
                        )
                    split_sizes.append(idx - prev)
                    prev = idx
                split_sizes.append(axis_dim - prev)

        if split_sizes is not None:
            if output_index < 0 or output_index >= len(split_sizes):
                raise ValueError(
                    f"split node '{node.output}' output_index={output_index} is out of range "
                    f"for {len(split_sizes)} split outputs."
                )
            splits = mb.split(
                x=x,
                split_sizes=split_sizes,
                axis=axis,
                name=f"{node.output}_split",
            )
            return mb.identity(x=splits[output_index], name=node.output)

        if num_splits is None:
            raise ValueError(
                f"split node '{node.output}' requires split_sizes/split_indices/num_splits attrs."
            )
        if output_index < 0 or output_index >= num_splits:
            raise ValueError(
                f"split node '{node.output}' output_index={output_index} is out of range "
                f"for {num_splits} split outputs."
            )
        splits = mb.split(
            x=x,
            num_splits=num_splits,
            axis=axis,
            name=f"{node.output}_split",
        )
        return mb.identity(x=splits[output_index], name=node.output)
    if mil_op == "gather":
        axis = int(node.attrs.get("axis", 0))
        gathered = mb.gather(
            x=env[node.inputs[0]],
            indices=env[node.inputs[1]],
            axis=axis,
        )
        target_shape = node.attrs.get("shape")
        if target_shape is not None:
            return mb.reshape(
                x=gathered,
                shape=[int(v) for v in target_shape],
                name=node.output,
            )
        return mb.identity(x=gathered, name=node.output)
    if mil_op == "gather_along_axis":
        axis = int(node.attrs.get("axis", 0))
        return mb.gather_along_axis(
            x=env[node.inputs[0]],
            indices=env[node.inputs[1]],
            axis=axis,
            name=node.output,
        )
    if mil_op == "squeeze":
        axes_attr = node.attrs.get("axes")
        axes: list[int] | None = None
        if axes_attr is not None:
            axes = [int(v) for v in axes_attr]
        return mb.squeeze(x=env[node.inputs[0]], axes=axes, name=node.output)
    if mil_op == "concat":
        if len(node.inputs) == 0:
            raise ValueError(f"concatenate node '{node.output}' requires at least 1 input.")
        axis = int(node.attrs.get("axis", 0))
        return mb.concat(values=[env[name] for name in node.inputs], axis=axis, name=node.output)
    if mil_op in {"zeros", "ones", "full"}:
        if mil_op == "full" and len(node.inputs) == 1 and "shape" not in node.attrs and "value" not in node.attrs:
            # MLX callback export can emit Full as a materialization op on a pre-broadcast tensor.
            return mb.identity(x=env[node.inputs[0]], name=node.output)
        shape = _int_list(node.attrs.get("shape"), "shape", node)
        if mil_op == "zeros":
            value = float(node.attrs.get("value", 0.0))
        elif mil_op == "ones":
            value = float(node.attrs.get("value", 1.0))
        else:
            if "value" not in node.attrs:
                raise ValueError(f"full node '{node.output}' requires 'value' attr.")
            value = float(node.attrs["value"])
        return mb.fill(shape=shape, value=value, name=node.output)
    if mil_op in {"zeros_like", "ones_like", "full_like"}:
        shape = mb.shape(x=env[node.inputs[0]])
        if mil_op == "zeros_like":
            value = float(node.attrs.get("value", 0.0))
        elif mil_op == "ones_like":
            value = float(node.attrs.get("value", 1.0))
        else:
            if "value" not in node.attrs:
                raise ValueError(f"full_like node '{node.output}' requires 'value' attr.")
            value = float(node.attrs["value"])
        return mb.fill(shape=shape, value=value, name=node.output)
    if mil_op == "arange":
        if "end" not in node.attrs:
            raise ValueError(f"arange node '{node.output}' requires 'end' attr.")
        start = node.attrs.get("start", 0)
        end = node.attrs["end"]
        step = node.attrs.get("step", 1)
        return mb.range_1d(start=start, end=end, step=step, name=node.output)
    if mil_op == "linspace":
        if "start" not in node.attrs or "stop" not in node.attrs:
            raise ValueError(f"linspace node '{node.output}' requires 'start' and 'stop' attrs.")
        num = int(node.attrs.get("num", 50))
        endpoint = bool(node.attrs.get("endpoint", True))
        dtype = _normalize_cast_dtype(str(node.attrs.get("dtype", "fp32")))
        start = float(node.attrs["start"])
        stop = float(node.attrs["stop"])
        if num <= 0:
            raise ValueError(f"linspace node '{node.output}' requires num > 0, got {num}.")
        if num == 1:
            return mb.fill(shape=[1], value=start, name=node.output)
        denom = num - 1 if endpoint else num
        step = (stop - start) / float(denom)
        idx = mb.range_1d(start=0, end=num, step=1, name=f"{node.output}_idx")
        idx = mb.cast(x=idx, dtype=dtype, name=f"{node.output}_idx_cast")
        scaled = mb.mul(x=idx, y=step, name=f"{node.output}_scaled")
        return mb.add(x=scaled, y=start, name=node.output)
    if mil_op == "select":
        return mb.select(
            cond=env[node.inputs[0]],
            a=env[node.inputs[1]],
            b=env[node.inputs[2]],
            name=node.output,
        )
    if mil_op == "cast":
        if "dtype" not in node.attrs:
            raise ValueError(f"cast node '{node.output}' requires 'dtype' attr.")
        dtype = _normalize_cast_dtype(str(node.attrs["dtype"]))
        return mb.cast(x=env[node.inputs[0]], dtype=dtype, name=node.output)
    if mil_op == "number_of_elements":
        shape = mb.shape(x=env[node.inputs[0]], name=f"{node.output}_shape")
        return mb.reduce_prod(x=shape, axes=[0], keep_dims=False, name=node.output)
    if mil_op == "acos":
        return mb.acos(x=_coerce_float_tensor(env[node.inputs[0]], node.output), name=node.output)
    if mil_op == "asin":
        return mb.asin(x=_coerce_float_tensor(env[node.inputs[0]], node.output), name=node.output)
    if mil_op == "atan":
        return mb.atan(x=_coerce_float_tensor(env[node.inputs[0]], node.output), name=node.output)
    if mil_op == "atanh":
        return mb.atanh(x=_coerce_float_tensor(env[node.inputs[0]], node.output), name=node.output)
    if mil_op == "negative":
        x = env[node.inputs[0]]
        dtype_name = _dtype_name(x)
        if "bool" in dtype_name:
            raise ValueError(f"negative node '{node.output}' does not support bool input.")
        factor: int | float = -1 if "int" in dtype_name else -1.0
        return mb.mul(x=x, y=factor, name=node.output)
    if mil_op == "degrees":
        scale = 180.0 / float(np.pi)
        x = _coerce_float_tensor(env[node.inputs[0]], node.output)
        return mb.mul(x=x, y=scale, name=node.output)
    if mil_op == "radians":
        scale = float(np.pi) / 180.0
        x = _coerce_float_tensor(env[node.inputs[0]], node.output)
        return mb.mul(x=x, y=scale, name=node.output)
    if mil_op == "expm1":
        x = _coerce_float_tensor(env[node.inputs[0]], node.output)
        ex = mb.exp(x=x, name=f"{node.output}_exp")
        return mb.sub(x=ex, y=1.0, name=node.output)
    if mil_op == "exp":
        x = _coerce_float_tensor(env[node.inputs[0]], node.output)
        return mb.exp(x=x, name=node.output)
    if mil_op == "log1p":
        x = _float_log_tensor(env[node.inputs[0]], node.output)
        return mb.log(x=mb.add(x=x, y=1.0, name=f"{node.output}_plus1"), name=node.output)
    if mil_op == "log2":
        x = _float_log_tensor(env[node.inputs[0]], node.output)
        return mb.mul(
            x=mb.log(x=x, name=f"{node.output}_ln"),
            y=(1.0 / float(np.log(2.0))),
            name=node.output,
        )
    if mil_op == "log10":
        x = _float_log_tensor(env[node.inputs[0]], node.output)
        return mb.mul(
            x=mb.log(x=x, name=f"{node.output}_ln"),
            y=(1.0 / float(np.log(10.0))),
            name=node.output,
        )
    if mil_op == "floor_div":
        return mb.floor_div(x=env[node.inputs[0]], y=env[node.inputs[1]], name=node.output)
    if mil_op == "var":
        if len(node.inputs) != 1:
            raise ValueError(f"var node '{node.output}' requires exactly 1 input.")
        return _lower_var_or_std(node, env[node.inputs[0]], compute_std=False)
    if mil_op == "std":
        if len(node.inputs) != 1:
            raise ValueError(f"std node '{node.output}' requires exactly 1 input.")
        return _lower_var_or_std(node, env[node.inputs[0]], compute_std=True)
    if mil_op == "divmod":
        if len(node.inputs) != 2:
            raise ValueError(f"divmod node '{node.output}' requires exactly 2 inputs.")
        which = str(node.attrs.get("output", node.attrs.get("which", "quotient"))).strip().lower()
        x = env[node.inputs[0]]
        y = env[node.inputs[1]]
        q = mb.floor_div(x=x, y=y, name=f"{node.output}_q")
        if which in {"quotient", "q", "0"}:
            return mb.identity(x=q, name=node.output)
        if which in {"remainder", "r", "1"}:
            return mb.mod(x=x, y=y, name=node.output)
        raise ValueError(
            f"divmod node '{node.output}' expects output in {{'quotient','remainder'}}; got {which!r}."
        )
    if mil_op == "conv":
        is_transpose = bool(node.attrs.get("transpose", node.attrs.get("transposed", False)))
        return _lower_conv_op(node, env, transpose=is_transpose)
    if mil_op == "conv_transpose":
        return _lower_conv_op(node, env, transpose=True)
    if mil_op == "addmm":
        if len(node.inputs) != 3:
            raise ValueError(
                f"addmm node '{node.output}' requires exactly 3 inputs (input, mat1, mat2)."
            )
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
        transpose_mat1 = bool(node.attrs.get("transpose_mat1", False))
        transpose_mat2 = bool(node.attrs.get("transpose_mat2", False))

        terms = [env[name] for name in node.inputs]
        resolved_bias = terms[0]
        resolved_mat1 = terms[1]
        resolved_mat2 = terms[2]
        term_shapes = [_shape_list_if_static(term) for term in terms]
        candidate_orders = [
            (0, 1, 2),
            (0, 2, 1),
            (2, 0, 1),
            (2, 1, 0),
            (1, 0, 2),
            (1, 2, 0),
        ]
        for bias_idx, mat1_idx, mat2_idx in candidate_orders:
            bias_shape = term_shapes[bias_idx]
            mat1_shape = term_shapes[mat1_idx]
            mat2_shape = term_shapes[mat2_idx]
            if bias_shape is None or mat1_shape is None or mat2_shape is None:
                continue
            out_shape = _matmul_output_shape(
                mat1_shape,
                mat2_shape,
                transpose_lhs=transpose_mat1,
                transpose_rhs=transpose_mat2,
            )
            if out_shape == bias_shape:
                resolved_bias = terms[bias_idx]
                resolved_mat1 = terms[mat1_idx]
                resolved_mat2 = terms[mat2_idx]
                break
        mm = mb.matmul(
            x=resolved_mat1,
            y=resolved_mat2,
            transpose_x=transpose_mat1,
            transpose_y=transpose_mat2,
            name=f"{node.output}_mm",
        )
        if alpha != 1.0:
            mm = mb.mul(x=mm, y=alpha, name=f"{node.output}_alpha")
        if beta != 1.0:
            resolved_bias = mb.mul(x=resolved_bias, y=beta, name=f"{node.output}_beta")
        return mb.add(x=resolved_bias, y=mm, name=node.output)
    if mil_op == "broadcast_to":
        if len(node.inputs) != 1:
            raise ValueError(f"broadcast_to node '{node.output}' requires exactly 1 input.")
        target_shape = _int_list(node.attrs.get("shape"), "shape", node)
        if any(dim < 0 for dim in target_shape):
            raise ValueError(
                f"broadcast_to node '{node.output}' requires non-negative target shape, got {target_shape}."
            )
        x = env[node.inputs[0]]
        x_shape = _static_shape_list(x, "x", node)
        return _broadcast_tensor_to_shape(x, x_shape, target_shape, node)
    if mil_op == "broadcast_arrays":
        if len(node.inputs) == 0:
            raise ValueError(f"broadcast_arrays node '{node.output}' requires at least 1 input.")
        input_index = int(node.attrs.get("input_index", 0))
        if input_index < 0 or input_index >= len(node.inputs):
            raise ValueError(
                f"broadcast_arrays node '{node.output}' input_index={input_index} is out of range "
                f"for {len(node.inputs)} input(s)."
            )

        all_shapes = [_static_shape_list(env[name], name, node) for name in node.inputs]
        if "shape" in node.attrs:
            target_shape = _int_list(node.attrs.get("shape"), "shape", node)
        else:
            target_shape = _infer_broadcast_shape(all_shapes, node)
        if any(dim < 0 for dim in target_shape):
            raise ValueError(
                f"broadcast_arrays node '{node.output}' requires non-negative target shape, got {target_shape}."
            )

        selected_name = node.inputs[input_index]
        x = env[selected_name]
        x_shape = _static_shape_list(x, selected_name, node)
        return _broadcast_tensor_to_shape(x, x_shape, target_shape, node)
    if mil_op == "outer":
        if len(node.inputs) != 2:
            raise ValueError(f"outer node '{node.output}' requires exactly 2 inputs.")
        x_flat = mb.reshape(x=env[node.inputs[0]], shape=[-1], name=f"{node.output}_x_flat")
        y_flat = mb.reshape(x=env[node.inputs[1]], shape=[-1], name=f"{node.output}_y_flat")
        x_mat = mb.reshape(x=x_flat, shape=[-1, 1], name=f"{node.output}_x_mat")
        y_mat = mb.reshape(x=y_flat, shape=[1, -1], name=f"{node.output}_y_mat")
        return mb.matmul(x=x_mat, y=y_mat, name=node.output)
    if mil_op == "inner":
        if len(node.inputs) != 2:
            raise ValueError(f"inner node '{node.output}' requires exactly 2 inputs.")
        x = env[node.inputs[0]]
        y = env[node.inputs[1]]
        x_shape = _static_shape_list(x, "x", node)
        y_shape = _static_shape_list(y, "y", node)
        if not x_shape or not y_shape:
            raise ValueError(f"inner node '{node.output}' requires rank >= 1 inputs.")
        if x_shape[-1] != y_shape[-1]:
            raise ValueError(
                f"inner node '{node.output}' last dimensions must match: "
                f"{x_shape[-1]} vs {y_shape[-1]}."
            )

        k_dim = x_shape[-1]
        x_rows = _prod_ints(x_shape[:-1]) if len(x_shape) > 1 else 1
        y_rows = _prod_ints(y_shape[:-1]) if len(y_shape) > 1 else 1
        x2d = mb.reshape(x=x, shape=[x_rows, k_dim], name=f"{node.output}_x2d")
        y2d = mb.reshape(x=y, shape=[y_rows, k_dim], name=f"{node.output}_y2d")
        dot = mb.matmul(x=x2d, y=y2d, transpose_y=True, name=f"{node.output}_mat")
        out_shape = x_shape[:-1] + y_shape[:-1]
        if len(out_shape) == 0:
            return mb.squeeze(x=dot, axes=[0, 1], name=node.output)
        return mb.reshape(x=dot, shape=out_shape, name=node.output)
    if mil_op == "tensordot":
        if len(node.inputs) != 2:
            raise ValueError(f"tensordot node '{node.output}' requires exactly 2 inputs.")
        x = env[node.inputs[0]]
        y = env[node.inputs[1]]
        x_shape = _static_shape_list(x, "x", node)
        y_shape = _static_shape_list(y, "y", node)

        x_axes, y_axes = _parse_tensordot_axes(node, len(x_shape), len(y_shape))
        x_free_axes = [axis for axis in range(len(x_shape)) if axis not in x_axes]
        y_free_axes = [axis for axis in range(len(y_shape)) if axis not in y_axes]

        for xa, ya in zip(x_axes, y_axes):
            if x_shape[xa] != y_shape[ya]:
                raise ValueError(
                    f"tensordot node '{node.output}' contraction dim mismatch: "
                    f"x axis {xa}={x_shape[xa]} vs y axis {ya}={y_shape[ya]}."
                )

        x_perm = x_free_axes + x_axes
        y_perm = y_axes + y_free_axes
        x_work = x if x_perm == list(range(len(x_shape))) else mb.transpose(x=x, perm=x_perm, name=f"{node.output}_x_t")
        y_work = y if y_perm == list(range(len(y_shape))) else mb.transpose(x=y, perm=y_perm, name=f"{node.output}_y_t")

        x_free_shape = [x_shape[axis] for axis in x_free_axes]
        y_free_shape = [y_shape[axis] for axis in y_free_axes]
        contract_shape = [x_shape[axis] for axis in x_axes]
        contract_size = _prod_ints(contract_shape) if contract_shape else 1
        x_rows = _prod_ints(x_free_shape) if x_free_shape else 1
        y_cols = _prod_ints(y_free_shape) if y_free_shape else 1

        x_2d = mb.reshape(x=x_work, shape=[x_rows, contract_size], name=f"{node.output}_x_2d")
        y_2d = mb.reshape(x=y_work, shape=[contract_size, y_cols], name=f"{node.output}_y_2d")
        out_2d = mb.matmul(x=x_2d, y=y_2d, name=f"{node.output}_mat")

        out_shape = x_free_shape + y_free_shape
        if len(out_shape) == 0:
            return mb.squeeze(x=out_2d, axes=[0, 1], name=node.output)
        return mb.reshape(x=out_2d, shape=out_shape, name=node.output)
    if mil_op == "isclose":
        if len(node.inputs) != 2:
            raise ValueError(f"isclose node '{node.output}' requires exactly 2 inputs.")
        close = _build_isclose_mask(node, env[node.inputs[0]], env[node.inputs[1]], node.output)
        return mb.identity(x=close, name=node.output)
    if mil_op == "allclose":
        if len(node.inputs) != 2:
            raise ValueError(f"allclose node '{node.output}' requires exactly 2 inputs.")
        close = _build_isclose_mask(node, env[node.inputs[0]], env[node.inputs[1]], node.output)
        rank = len(close.shape)
        if rank == 0:
            return mb.identity(x=close, name=node.output)
        close_i = mb.cast(x=close, dtype="int32", name=f"{node.output}_close_i")
        all_i = mb.reduce_prod(
            x=close_i,
            axes=list(range(rank)),
            keep_dims=False,
            name=f"{node.output}_all_i",
        )
        return mb.cast(x=all_i, dtype="bool", name=node.output)
    if mil_op == "nan_to_num":
        if len(node.inputs) != 1:
            raise ValueError(f"nan_to_num node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        defaults = _nan_to_num_defaults(_dtype_name(x))
        if defaults is None:
            return mb.identity(x=x, name=node.output)

        nan_default, posinf_default, neginf_default = defaults
        nan_value = float(node.attrs.get("nan", nan_default))
        posinf_value = float(node.attrs.get("posinf", posinf_default))
        neginf_value = float(node.attrs.get("neginf", neginf_default))

        nan_mask = mb.not_equal(x=x, y=x, name=f"{node.output}_nan_mask")
        out = mb.select(cond=nan_mask, a=nan_value, b=x, name=f"{node.output}_nan")
        posinf_mask = mb.equal(x=out, y=float("inf"), name=f"{node.output}_posinf_mask")
        out = mb.select(cond=posinf_mask, a=posinf_value, b=out, name=f"{node.output}_posinf")
        neginf_mask = mb.equal(x=out, y=float("-inf"), name=f"{node.output}_neginf_mask")
        return mb.select(cond=neginf_mask, a=neginf_value, b=out, name=node.output)
    if mil_op == "diag":
        if len(node.inputs) != 1:
            raise ValueError(f"diag node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        x_shape = _static_shape_list(x, "x", node)
        offset = int(node.attrs.get("k", node.attrs.get("offset", 0)))

        if len(x_shape) == 1:
            return _diag_from_vector(x, x_shape, offset, node)
        if len(x_shape) == 2:
            return _extract_diagonal(
                x=x,
                x_shape=x_shape,
                axis1=0,
                axis2=1,
                offset=offset,
                name_prefix=node.output,
            )
        raise ValueError(f"diag node '{node.output}' supports rank-1 or rank-2 input, got rank {len(x_shape)}.")
    if mil_op == "diagonal":
        if len(node.inputs) != 1:
            raise ValueError(f"diagonal node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        x_shape = _static_shape_list(x, "x", node)
        rank = len(x_shape)
        axis1 = int(node.attrs.get("axis1", 0))
        axis2 = int(node.attrs.get("axis2", 1))
        axis1 = _normalize_axes(rank, [axis1], "axis1", node)[0]
        axis2 = _normalize_axes(rank, [axis2], "axis2", node)[0]
        offset = int(node.attrs.get("offset", node.attrs.get("k", 0)))
        return _extract_diagonal(
            x=x,
            x_shape=x_shape,
            axis1=axis1,
            axis2=axis2,
            offset=offset,
            name_prefix=node.output,
        )
    if mil_op == "trace":
        if len(node.inputs) != 1:
            raise ValueError(f"trace node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        x_shape = _static_shape_list(x, "x", node)
        rank = len(x_shape)
        axis1 = int(node.attrs.get("axis1", 0))
        axis2 = int(node.attrs.get("axis2", 1))
        axis1 = _normalize_axes(rank, [axis1], "axis1", node)[0]
        axis2 = _normalize_axes(rank, [axis2], "axis2", node)[0]
        offset = int(node.attrs.get("offset", node.attrs.get("k", 0)))
        diagonal = _extract_diagonal(
            x=x,
            x_shape=x_shape,
            axis1=axis1,
            axis2=axis2,
            offset=offset,
            name_prefix=f"{node.output}_diag",
        )
        diag_rank = len(diagonal.shape)
        return mb.reduce_sum(
            x=diagonal,
            axes=[diag_rank - 1],
            keep_dims=False,
            name=node.output,
        )
    if mil_op == "tri":
        if len(node.inputs) != 0:
            raise ValueError(f"tri node '{node.output}' expects no inputs.")
        if "n" not in node.attrs:
            raise ValueError(f"tri node '{node.output}' requires 'n' attr.")
        n = int(node.attrs["n"])
        m = int(node.attrs.get("m", n))
        k = int(node.attrs.get("k", 0))
        if n < 0 or m < 0:
            raise ValueError(f"tri node '{node.output}' requires non-negative n/m, got n={n}, m={m}.")

        mask = _build_tri_mask(rows=n, cols=m, k=k, lower=True, prefix=node.output)
        dtype = _normalize_cast_dtype(str(node.attrs.get("dtype", "fp32")))
        if dtype == "bool":
            return mb.identity(x=mask, name=node.output)
        return mb.cast(x=mask, dtype=dtype, name=node.output)
    if mil_op in {"tril", "triu"}:
        if len(node.inputs) != 1:
            raise ValueError(f"{mil_op} node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        x_shape = _static_shape_list(x, "x", node)
        if len(x_shape) < 2:
            raise ValueError(f"{mil_op} node '{node.output}' requires rank >= 2 input.")
        rows = x_shape[-2]
        cols = x_shape[-1]
        k = int(node.attrs.get("k", 0))
        lower = mil_op == "tril"
        mask2d = _build_tri_mask(rows=rows, cols=cols, k=k, lower=lower, prefix=f"{node.output}_mask")

        if len(x_shape) > 2:
            prefix_size = _prod_ints(x_shape[:-2])
            x_work = mb.reshape(x=x, shape=[prefix_size, rows, cols], name=f"{node.output}_x3")
            mask_work = mb.reshape(x=mask2d, shape=[1, rows, cols], name=f"{node.output}_mask3")
            mask_work = mb.tile(
                x=mask_work, reps=[prefix_size, 1, 1], name=f"{node.output}_mask_tile"
            )
        else:
            x_work = x
            mask_work = mask2d

        zero_value = _zero_scalar_for_dtype_name(_dtype_name(x))
        masked = mb.select(cond=mask_work, a=x_work, b=zero_value, name=f"{node.output}_masked")
        if len(x_shape) > 2:
            return mb.reshape(x=masked, shape=x_shape, name=node.output)
        return mb.identity(x=masked, name=node.output)
    if mil_op == "all":
        if len(node.inputs) != 1:
            raise ValueError(f"all node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        mask = _as_bool_tensor(x, node.output)
        axes = _parse_reduction_axes(node, len(mask.shape))
        keep_dims = bool(node.attrs.get("keep_dims", False))
        return _reduce_bool_mask(mask, axes, keep_dims, any_mode=False, name=node.output)
    if mil_op == "any":
        if len(node.inputs) != 1:
            raise ValueError(f"any node '{node.output}' requires exactly 1 input.")
        x = env[node.inputs[0]]
        mask = _as_bool_tensor(x, node.output)
        axes = _parse_reduction_axes(node, len(mask.shape))
        keep_dims = bool(node.attrs.get("keep_dims", False))
        return _reduce_bool_mask(mask, axes, keep_dims, any_mode=True, name=node.output)
    if mil_op == "array_equal":
        if len(node.inputs) != 2:
            raise ValueError(f"array_equal node '{node.output}' requires exactly 2 inputs.")
        x = env[node.inputs[0]]
        y = env[node.inputs[1]]
        x_shape = _static_shape_list(x, "x", node)
        y_shape = _static_shape_list(y, "y", node)
        if x_shape != y_shape:
            return _bool_scalar(False, node.output)
        eq = mb.equal(x=x, y=y, name=f"{node.output}_eq")
        return _reduce_bool_mask(
            mask=eq,
            axes=list(range(len(eq.shape))),
            keep_dims=False,
            any_mode=False,
            name=node.output,
        )
    if mil_op == "isnan":
        if len(node.inputs) != 1:
            raise ValueError(f"isnan node '{node.output}' requires exactly 1 input.")
        x = _to_float_for_inf_checks(env[node.inputs[0]], node.output)
        return mb.not_equal(x=x, y=x, name=node.output)
    if mil_op == "isinf":
        if len(node.inputs) != 1:
            raise ValueError(f"isinf node '{node.output}' requires exactly 1 input.")
        x = _to_float_for_inf_checks(env[node.inputs[0]], node.output)
        abs_x = mb.abs(x=x, name=f"{node.output}_abs")
        return mb.equal(x=abs_x, y=float("inf"), name=node.output)
    if mil_op == "isfinite":
        if len(node.inputs) != 1:
            raise ValueError(f"isfinite node '{node.output}' requires exactly 1 input.")
        x = _to_float_for_inf_checks(env[node.inputs[0]], node.output)
        isnan = mb.not_equal(x=x, y=x, name=f"{node.output}_isnan")
        abs_x = mb.abs(x=x, name=f"{node.output}_abs")
        isinf = mb.equal(x=abs_x, y=float("inf"), name=f"{node.output}_isinf")
        nan_or_inf = mb.logical_or(x=isnan, y=isinf, name=f"{node.output}_nan_or_inf")
        return mb.logical_not(x=nan_or_inf, name=node.output)
    if mil_op == "isneginf":
        if len(node.inputs) != 1:
            raise ValueError(f"isneginf node '{node.output}' requires exactly 1 input.")
        x = _to_float_for_inf_checks(env[node.inputs[0]], node.output)
        return mb.equal(x=x, y=float("-inf"), name=node.output)
    if mil_op == "isposinf":
        if len(node.inputs) != 1:
            raise ValueError(f"isposinf node '{node.output}' requires exactly 1 input.")
        x = _to_float_for_inf_checks(env[node.inputs[0]], node.output)
        return mb.equal(x=x, y=float("inf"), name=node.output)
    if mil_op == "eye":
        if len(node.inputs) != 0:
            raise ValueError(f"eye node '{node.output}' expects no inputs.")
        if "n" not in node.attrs:
            raise ValueError(f"eye node '{node.output}' requires 'n' attr.")
        n = int(node.attrs["n"])
        m = int(node.attrs.get("m", n))
        k = int(node.attrs.get("k", 0))
        if n < 0 or m < 0:
            raise ValueError(f"eye node '{node.output}' requires non-negative n/m, got n={n}, m={m}.")

        row_idx = mb.range_1d(start=0, end=n, step=1, name=f"{node.output}_rows")
        row_idx = mb.reshape(x=row_idx, shape=[n, 1], name=f"{node.output}_rows_col")
        row_idx = mb.tile(x=row_idx, reps=[1, m], name=f"{node.output}_rows_mat")

        col_idx = mb.range_1d(start=0, end=m, step=1, name=f"{node.output}_cols")
        col_idx = mb.reshape(x=col_idx, shape=[1, m], name=f"{node.output}_cols_row")
        col_idx = mb.tile(x=col_idx, reps=[n, 1], name=f"{node.output}_cols_mat")

        diagonal = mb.add(x=row_idx, y=int(k), name=f"{node.output}_diag_idx")
        mask = mb.equal(x=col_idx, y=diagonal, name=f"{node.output}_mask")
        dtype = _normalize_cast_dtype(str(node.attrs.get("dtype", "fp32")))
        if dtype == "bool":
            return mb.identity(x=mask, name=node.output)
        return mb.cast(x=mask, dtype=dtype, name=node.output)
    if mil_op == "meshgrid":
        if len(node.inputs) == 0:
            raise ValueError(f"meshgrid node '{node.output}' requires at least 1 input.")
        if bool(node.attrs.get("sparse", False)):
            raise ValueError(f"meshgrid node '{node.output}' does not support sparse=True yet.")

        input_index = int(node.attrs.get("input_index", 0))
        indexing = str(node.attrs.get("indexing", "xy")).strip().lower()
        vectors = [env[name] for name in node.inputs]
        dims: list[int] = []
        for i, vector in enumerate(vectors):
            vector_shape = _static_shape_list(vector, node.inputs[i], node)
            if len(vector_shape) != 1:
                raise ValueError(
                    f"meshgrid node '{node.output}' requires rank-1 inputs, got shape {vector_shape}."
                )
            dims.append(int(vector_shape[0]))

        out_dims, varying_axis = _meshgrid_dims_and_axis(dims, input_index, indexing, node)
        src = vectors[input_index]
        src_len = dims[input_index]
        base_shape = [1] * len(out_dims)
        base_shape[varying_axis] = src_len
        reshaped = mb.reshape(x=src, shape=base_shape, name=f"{node.output}_reshape")
        reps = [1 if axis == varying_axis else int(dim) for axis, dim in enumerate(out_dims)]
        if all(rep == 1 for rep in reps):
            return mb.identity(x=reshaped, name=node.output)
        return mb.tile(x=reshaped, reps=reps, name=node.output)
    if mil_op == "kron":
        if len(node.inputs) != 2:
            raise ValueError(f"kron node '{node.output}' requires exactly 2 inputs.")
        x = env[node.inputs[0]]
        y = env[node.inputs[1]]
        x_shape = _static_shape_list(x, "x", node)
        y_shape = _static_shape_list(y, "y", node)
        rank = max(len(x_shape), len(y_shape))
        if rank == 0:
            return mb.mul(x=x, y=y, name=node.output)

        x_pad = [1] * (rank - len(x_shape)) + x_shape
        y_pad = [1] * (rank - len(y_shape)) + y_shape
        x_reshape: list[int] = []
        y_reshape: list[int] = []
        out_shape: list[int] = []
        for x_dim, y_dim in zip(x_pad, y_pad):
            x_reshape.extend([x_dim, 1])
            y_reshape.extend([1, y_dim])
            out_shape.append(int(x_dim) * int(y_dim))

        x_work = mb.reshape(x=x, shape=x_reshape, name=f"{node.output}_x")
        y_work = mb.reshape(x=y, shape=y_reshape, name=f"{node.output}_y")
        product = mb.mul(x=x_work, y=y_work, name=f"{node.output}_mul")
        return mb.reshape(x=product, shape=out_shape, name=node.output)
    if mil_op == "logaddexp":
        if len(node.inputs) != 2:
            raise ValueError(f"logaddexp node '{node.output}' requires exactly 2 inputs.")
        x = _coerce_float_tensor(env[node.inputs[0]], f"{node.output}_x")
        y = _coerce_float_tensor(env[node.inputs[1]], f"{node.output}_y")
        m = mb.maximum(x=x, y=y, name=f"{node.output}_max")
        n = mb.minimum(x=x, y=y, name=f"{node.output}_min")
        same = mb.equal(x=m, y=n, name=f"{node.output}_same")
        delta_raw = mb.sub(x=n, y=m, name=f"{node.output}_delta_raw")
        delta = mb.select(cond=same, a=0.0, b=delta_raw, name=f"{node.output}_delta")
        exp_delta = mb.exp(x=delta, name=f"{node.output}_exp_delta")
        log_term = mb.log(x=mb.add(x=1.0, y=exp_delta, name=f"{node.output}_sum"), name=f"{node.output}_log")
        return mb.add(x=m, y=log_term, name=node.output)
    if mil_op == "identity":
        return mb.identity(x=env[node.inputs[0]], name=node.output)

    raise ValueError(f"Internal error: missing lowering implementation for {mil_op}")


def build_mil_program(
    graph: Graph,
    deployment_target: ct.target = ct.target.iOS18,
    normalize: bool = True,
    target_profile: str | None = "default",
    shared_state_specs: list[StateSpec] | None = None,
):
    return build_mil_program_from_graphs(
        {"main": graph},
        deployment_target=deployment_target,
        normalize=normalize,
        target_profile=target_profile,
        shared_input_specs=list(graph.inputs),
        shared_state_specs=shared_state_specs,
    )


def build_mil_program_from_graphs(
    function_graphs: dict[str, Graph],
    deployment_target: ct.target = ct.target.iOS18,
    normalize: bool = True,
    target_profile: str | None = "default",
    shared_input_specs: list[TensorSpec] | None = None,
    shared_state_specs: list[StateSpec] | None = None,
):
    if not function_graphs:
        raise ValueError("build_mil_program_from_graphs requires at least one function graph.")

    profile = resolve_lowering_profile(target_profile)
    program = Program()
    expected_inputs: dict[str, tuple[tuple[int, ...], str]] | None = None
    state_spec_by_name: dict[str, StateSpec] = {}
    state_names: set[str] = set()
    if shared_input_specs is not None:
        expected_inputs = {}
        for spec in shared_input_specs:
            if spec.name in expected_inputs:
                raise ValueError(f"Duplicate shared input spec name: {spec.name}")
            expected_inputs[spec.name] = (tuple(int(v) for v in spec.shape), str(spec.dtype))
    if shared_state_specs is not None:
        seen_state_names: set[str] = set()
        for spec in shared_state_specs:
            if spec.name in seen_state_names:
                raise ValueError(f"Duplicate shared state spec name: {spec.name}")
            seen_state_names.add(spec.name)
            state_spec_by_name[spec.name] = spec
        state_names = seen_state_names

    for function_name, graph in function_graphs.items():
        if not str(function_name).strip():
            raise ValueError("Function name cannot be empty.")
        graph.validate()
        working_graph = normalize_graph(graph) if normalize else graph
        ensure_supported(working_graph)
        graph_input_specs = {
            spec.name: (tuple(int(v) for v in spec.shape), str(spec.dtype))
            for spec in working_graph.inputs
        }
        filtered_graph_inputs = {
            name: spec for name, spec in graph_input_specs.items() if name not in state_names
        }
        filtered_expected_inputs = (
            {name: spec for name, spec in expected_inputs.items() if name not in state_names}
            if expected_inputs is not None
            else None
        )
        if filtered_expected_inputs is not None and filtered_graph_inputs != filtered_expected_inputs:
            raise ValueError(
                f"Function '{function_name}' inputs do not match shared_input_specs. "
                f"Expected {sorted(filtered_expected_inputs.items())}, got {sorted(filtered_graph_inputs.items())}."
            )
        input_list = shared_input_specs if shared_input_specs is not None else working_graph.inputs
        input_specs = {
            spec.name: _to_tensor_spec(spec)
            for spec in input_list
            if spec.name not in state_names
        }
        for state_name in state_names:
            state_spec = state_spec_by_name[state_name]
            input_specs[state_name] = _to_state_tensor_spec(state_spec)

        with Function(input_specs, opset_version=deployment_target) as func:
            env = dict(func.inputs)
            for node in working_graph.nodes:
                env[node.output] = _lower_node(node, env, profile)

            outputs = [env[name] for name in working_graph.outputs]
            func.set_outputs(outputs)
            program.add_function(str(function_name), func)
    return program


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


def convert_program_to_model(
    program,
    deployment_target: ct.target = ct.target.iOS18,
    convert_to: str = "mlprogram",
    compute_precision: str | None = None,
    compute_units: str | None = None,
    inputs: list[object] | None = None,
    state_specs: list[StateSpec] | None = None,
    states: list[object] | None = None,
):
    kwargs: dict[str, object] = {
        "convert_to": convert_to,
        "minimum_deployment_target": deployment_target,
    }
    if compute_precision is not None:
        key = str(compute_precision).strip().lower()
        if key in {"fp16", "float16"}:
            kwargs["compute_precision"] = ct.precision.FLOAT16
        elif key in {"fp32", "float32"}:
            kwargs["compute_precision"] = ct.precision.FLOAT32
        elif key in {"auto", "default"}:
            pass
        else:
            raise ValueError(
                f"Unsupported compute_precision={compute_precision!r}. "
                "Use one of: auto, fp16, fp32."
            )
    if compute_units is not None:
        kwargs["compute_units"] = _parse_compute_units(compute_units)
    if inputs is not None:
        kwargs["inputs"] = inputs
    if states is not None and state_specs is not None:
        raise ValueError("Provide either 'states' or 'state_specs', not both.")
    if state_specs is not None:
        kwargs["states"] = build_coreml_state_types(state_specs)
    elif states is not None:
        kwargs["states"] = states
    try:
        return ct.convert(program, **kwargs)
    except ValueError as exc:
        # ct.convert(program=Program, states=...) is rejected by some coremltools versions.
        # Stateful MIL programs already encode state placeholders via StateTensorSpec.
        if "states" in kwargs and "can only be passed with pytorch source model" in str(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("states", None)
            return ct.convert(program, **retry_kwargs)
        raise
