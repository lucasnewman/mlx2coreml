import argparse
from pathlib import Path

import coremltools as ct

# Importing ops triggers installation/registration into the Builder + registry.
from coremltools.converters.mil.mil import ops as _ops  # noqa: F401
from coremltools.converters.mil.mil.ops.registry import SSAOpRegistry


def supported_opsets(variants: dict) -> list[ct.target]:
    return [v for v in SSAOpRegistry.SUPPORTED_OPSET_VERSIONS if v in variants]


def collect_core_op_rows() -> list[tuple[str, ct.target | None, list[ct.target]]]:
    rows = []
    for op_name, variants in SSAOpRegistry.core_ops.items():
        opsets = supported_opsets(variants)
        first_supported = opsets[0] if opsets else None
        rows.append((op_name, first_supported, opsets))

    rows.sort(key=lambda row: (row[1].value if row[1] else 9999, row[0]))
    return rows


def render_markdown(rows: list[tuple[str, ct.target | None, list[ct.target]]]) -> str:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit Markdown listing MIL core ops and supported opsets."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("mil_ops.md"),
        help="Path to write the Markdown output.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the generated Markdown to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_core_op_rows()
    markdown = render_markdown(rows)

    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(rows)} ops to {args.output}")
    if args.stdout:
        print(markdown, end="")


if __name__ == "__main__":
    main()
