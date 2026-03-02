import argparse
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import coremltools as ct

# Importing ops triggers installation/registration into the Builder + registry.
from coremltools.converters.mil.mil import ops as _mil_ops  # noqa: F401
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.op_registry import SUPPORTED_MLX_TO_MIL_OPS  # noqa: E402

DEFAULT_MLX_OPS_HEADER = "https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/ops.h"


@dataclass(frozen=True)
class NotYetRow:
    op: str
    lowering: str


@dataclass(frozen=True)
class NotSupportedRow:
    op: str
    reason: str


NOT_YET_IMPLEMENTED: tuple[NotYetRow, ...] = (
    NotYetRow("arccosh", "log/sqrt formula"),
    NotYetRow("arcsinh", "log/sqrt formula"),
    NotYetRow("arctan2", "atan + quadrant correction"),
    NotYetRow("argpartition", "no direct op; approximate via sort/topk"),
    NotYetRow("as_strided", "no safe MIL equivalent"),
    NotYetRow("bitwise_and", "no direct op (emulate expensive)"),
    NotYetRow("bitwise_invert", "no direct op (emulate expensive)"),
    NotYetRow("bitwise_or", "no direct op (emulate expensive)"),
    NotYetRow("bitwise_xor", "no direct op (emulate expensive)"),
    NotYetRow("block_masked_mm", "custom op or decompose with major perf loss"),
    NotYetRow("conjugate", "complex dtype gap"),
    NotYetRow("cummax", "while_loop scan + max state"),
    NotYetRow("cummin", "while_loop scan + min state"),
    NotYetRow("cumprod", "while_loop + running state"),
    NotYetRow("erfinv", "numeric approx (poly/rational)"),
    NotYetRow("gather_mm", "fused gather+matmul (custom likely)"),
    NotYetRow("hadamard_transform", "structured matmul / staged butterflies"),
    NotYetRow("hamming", "arange + cos formula"),
    NotYetRow("hanning", "arange + cos formula"),
    NotYetRow("imag", "complex dtype gap"),
    NotYetRow("left_shift", "bitwise shift gap"),
    NotYetRow("logcumsumexp", "scan op missing; while_loop decomposition"),
    NotYetRow("masked_scatter", "scatter with mask semantics mismatch"),
    NotYetRow("median", "partition/sort needed; expensive"),
    NotYetRow("partition", "partition primitive missing"),
    NotYetRow("put_along_axis", "scatter update semantics mismatch"),
    NotYetRow("real", "complex dtype gap"),
    NotYetRow("repeat", "tile + reshape + slice"),
    NotYetRow("right_shift", "bitwise shift gap"),
    NotYetRow("roll", "slice + concat"),
    NotYetRow("scatter_add", "scatter reduce(add) missing"),
    NotYetRow("scatter_add_axis", "scatter reduce(add) missing"),
    NotYetRow("scatter_max", "scatter reduce(max) missing"),
    NotYetRow("scatter_min", "scatter reduce(min) missing"),
    NotYetRow("scatter_prod", "scatter reduce(prod) missing"),
    NotYetRow("segmented_mm", "segmented/fused matmul primitive missing"),
    NotYetRow("sort", "argsort + gather_along_axis"),
)


NOT_SUPPORTED: tuple[NotSupportedRow, ...] = (
    NotSupportedRow(
        "depends",
        "Control-dependency semantics are not representable in the current MIL lowering model.",
    ),
    NotSupportedRow(
        "from_fp8",
        "**Policy: not supported.** Non-`fp16`/`fp32` quantization path is out of scope for this project.",
    ),
    NotSupportedRow(
        "gather_qmm",
        "**Policy: not supported.** Quantized qmm-family op is out of scope for this project.",
    ),
    NotSupportedRow(
        "qqmm",
        "**Policy: not supported.** Quantized qmm-family op is out of scope for this project.",
    ),
    NotSupportedRow(
        "quantized_matmul",
        "**Policy: not supported.** Non-`fp16`/`fp32` quantization path is out of scope for this project.",
    ),
    NotSupportedRow(
        "to_fp8",
        "**Policy: not supported.** Non-`fp16`/`fp32` quantization path is out of scope for this project.",
    ),
    NotSupportedRow(
        "view",
        "Stride-aware view semantics are not representable in MIL (outside contiguous reshape-compatible cases).",
    ),
)


def render_ops_status_markdown(
    not_yet: list[NotYetRow],
    not_supported: list[NotSupportedRow],
) -> str:
    supported_rows = sorted(SUPPORTED_MLX_TO_MIL_OPS.items())
    not_yet_sorted = sorted(not_yet, key=lambda row: row.op)
    not_supported_sorted = sorted(not_supported, key=lambda row: row.op)

    lines = [
        "# Ops Status (MLX -> MIL)",
        "",
        f"- Supported: **{len(supported_rows)}**",
        f"- Not yet implemented: **{len(not_yet_sorted)}**",
        f"- Not supported: **{len(not_supported_sorted)}**",
        "",
        "## Supported",
        "",
        "| MLX Op | MIL Op |",
        "| --- | --- |",
    ]

    for mlx_op, mil_op in supported_rows:
        lines.append(f"| `{mlx_op}` | `{mil_op}` |")

    lines.extend(
        [
            "",
            "## Not Yet Implemented",
            "",
            "| MLX Op | Proposed MIL Lowering |",
            "| --- | --- |",
        ]
    )

    if not not_yet_sorted:
        lines.append("| _None_ |  |")
    else:
        for row in not_yet_sorted:
            lines.append(f"| `{row.op}` | `{row.lowering}` |")

    lines.extend(
        [
            "",
            "## Not Supported",
            "",
            "| MLX Op | Reason |",
            "| --- | --- |",
        ]
    )

    if not not_supported_sorted:
        lines.append("| _None_ |  |")
    else:
        for row in not_supported_sorted:
            lines.append(f"| `{row.op}` | {row.reason} |")

    return "\n".join(lines) + "\n"


def supported_mil_opsets(variants: dict) -> list[ct.target]:
    return [v for v in SSAOpRegistry.SUPPORTED_OPSET_VERSIONS if v in variants]


def collect_mil_core_op_rows() -> list[tuple[str, ct.target | None, list[ct.target]]]:
    rows = []
    for op_name, variants in SSAOpRegistry.core_ops.items():
        opsets = supported_mil_opsets(variants)
        first_supported = opsets[0] if opsets else None
        rows.append((op_name, first_supported, opsets))

    rows.sort(key=lambda row: (row[1].value if row[1] else 9999, row[0]))
    return rows


def render_mil_markdown(rows: list[tuple[str, ct.target | None, list[ct.target]]]) -> str:
    lines = [
        "# MIL Core Ops by Opset",
        "",
        "Generated from `SSAOpRegistry.core_ops`.",
        "",
        "| Op | First Supported Opset | Supported Opsets |",
        "| --- | --- | --- |",
    ]

    for op_name, first_supported, opsets in rows:
        first_text = first_supported.name if first_supported else "N/A"
        all_text = ", ".join(v.name for v in opsets) if opsets else "N/A"
        lines.append(f"| `{op_name}` | `{first_text}` | `{all_text}` |")

    return "\n".join(lines) + "\n"


def read_text(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        try:
            with urlopen(source) as resp:
                return resp.read().decode("utf-8")
        except Exception as url_error:
            try:
                result = subprocess.run(
                    ["curl", "-fsSL", source],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout
            except Exception as curl_error:
                raise RuntimeError(
                    "Could not download ops header. Pass --ops-header with a local file path. "
                    f"urlopen error: {url_error}; curl error: {curl_error}"
                ) from curl_error
    return Path(source).read_text(encoding="utf-8")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def parse_mlx_ops(header_text: str) -> dict[str, list[str]]:
    body = strip_comments(header_text)
    declarations = re.findall(r"MLX_API\s+(.*?);", body, flags=re.S)
    ops: dict[str, list[str]] = defaultdict(list)

    for decl in declarations:
        normalized = " ".join(decl.split())
        lparen = normalized.find("(")
        if lparen == -1:
            continue

        prefix = normalized[:lparen].strip()
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", prefix)
        if not name_match:
            continue

        name = name_match.group(1)
        ops[name].append(f"{normalized};")

    return dict(ops)


def render_mlx_markdown(
    ops: dict[str, list[str]],
    header_source: str,
    mil_ops: set[str],
) -> str:
    op_names = sorted(ops)
    lines = [
        "# MLX Ops",
        "",
        f"Source: `{header_source}`",
        "",
        f"Total unique ops: **{len(op_names)}**",
        "",
        "| Op | Overloads in `ops.h` |",
        "| --- | ---: |",
    ]

    for name in op_names:
        lines.append(f"| `{name}` | {len(ops[name])} |")

    if mil_ops:
        overlap = sorted(set(op_names) & mil_ops)
        mlx_only = sorted(set(op_names) - mil_ops)
        mil_only = sorted(mil_ops - set(op_names))
        lines.extend(
            [
                "",
                "## MIL Overlap",
                "",
                f"- Shared names: **{len(overlap)}**",
                f"- MLX-only names: **{len(mlx_only)}**",
                f"- MIL-only names: **{len(mil_only)}**",
            ]
        )

    return "\n".join(lines) + "\n"


def resolve_ops_header(user_value: str | None) -> str:
    if user_value:
        return user_value

    for local_path in (Path("mlx_ops.h"), Path("mlx/ops.h")):
        if local_path.exists():
            return str(local_path)
    return DEFAULT_MLX_OPS_HEADER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate docs/ops_status.md, docs/mil_ops.md, and docs/mlx_ops.md "
            "from live registries and in-script status definitions."
        )
    )
    parser.add_argument(
        "--ops-header",
        default=None,
        help=(
            "Path or URL to mlx/ops.h "
            "(default: local mlx_ops.h/mlx/ops.h if present, otherwise GitHub URL)."
        ),
    )
    parser.add_argument(
        "--mil-output",
        type=Path,
        default=Path("docs/mil_ops.md"),
        help="Path to write generated MIL ops markdown.",
    )
    parser.add_argument(
        "--mlx-output",
        type=Path,
        default=Path("docs/mlx_ops.md"),
        help="Path to write generated MLX ops markdown.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--ops-status-output",
        dest="ops_status_output",
        type=Path,
        default=Path("docs/ops_status.md"),
        help="Path to write generated ops-status markdown.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Backward-compatible alias for --stdout-ops-status.",
    )
    parser.add_argument(
        "--stdout-ops-status",
        action="store_true",
        help="Also print generated ops-status markdown to stdout.",
    )
    parser.add_argument(
        "--stdout-mil",
        action="store_true",
        help="Also print generated MIL ops markdown to stdout.",
    )
    parser.add_argument(
        "--stdout-mlx",
        action="store_true",
        help="Also print generated MLX ops markdown to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mil_rows = collect_mil_core_op_rows()
    mil_markdown = render_mil_markdown(mil_rows)
    args.mil_output.parent.mkdir(parents=True, exist_ok=True)
    args.mil_output.write_text(mil_markdown, encoding="utf-8")
    mil_op_names = {name for name, _, _ in mil_rows}
    print(f"Wrote MIL ops doc to {args.mil_output} (ops={len(mil_rows)})")

    ops_header = resolve_ops_header(args.ops_header)
    header_text = read_text(ops_header)
    mlx_ops = parse_mlx_ops(header_text)
    mlx_markdown = render_mlx_markdown(mlx_ops, ops_header, mil_op_names)
    args.mlx_output.parent.mkdir(parents=True, exist_ok=True)
    args.mlx_output.write_text(mlx_markdown, encoding="utf-8")
    print(
        f"Wrote MLX ops doc to {args.mlx_output} "
        f"(ops={len(mlx_ops)}, shared_with_mil={len(set(mlx_ops) & mil_op_names)})"
    )

    not_yet = list(NOT_YET_IMPLEMENTED)
    not_supported = list(NOT_SUPPORTED)
    ops_status_markdown = render_ops_status_markdown(not_yet, not_supported)

    args.ops_status_output.parent.mkdir(parents=True, exist_ok=True)
    args.ops_status_output.write_text(ops_status_markdown, encoding="utf-8")
    print(
        f"Wrote ops status doc to {args.ops_status_output} "
        f"(supported={len(SUPPORTED_MLX_TO_MIL_OPS)}, "
        f"not_yet={len(not_yet)}, not_supported={len(not_supported)})"
    )
    if args.stdout_mil:
        print(mil_markdown, end="")
    if args.stdout_mlx:
        print(mlx_markdown, end="")
    if args.stdout or args.stdout_ops_status:
        print(ops_status_markdown, end="")


if __name__ == "__main__":
    main()
