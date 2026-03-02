import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx2coreml.from_mlx import capture_graph_from_mlx_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture an MLX graph from a Python callable and write DOT + parsed IR artifacts. "
            "The target function is invoked as function(**inputs)."
        )
    )
    parser.add_argument(
        "--module",
        required=True,
        help="Python module path containing the capture function (for example: mypkg.my_model).",
    )
    parser.add_argument(
        "--function",
        required=True,
        help="Function name in --module. Must accept named MLX inputs (**kwargs).",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        required=True,
        help="Path to .npz inputs where each key matches a function argument name.",
    )
    parser.add_argument(
        "--dot-output",
        type=Path,
        default=Path("artifacts/captured_graph.dot"),
        help="Path to write exported MLX DOT graph.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/captured_graph.json"),
        help="Path to write parsed graph JSON.",
    )
    parser.add_argument(
        "--expected-output",
        type=Path,
        default=Path("artifacts/captured_expected_outputs.npz"),
        help="Path to write captured output tensors (.npz) keyed by parsed output name.",
    )
    parser.add_argument(
        "--inputs-output",
        type=Path,
        default=Path("artifacts/captured_inputs.npz"),
        help="Path to write normalized capture inputs (.npz).",
    )
    parser.add_argument(
        "--allow-unknown-sources",
        action="store_true",
        help="Allow parser to keep source nodes not present in input specs.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=["callback", "dot"],
        default="callback",
        help="Graph capture backend. 'callback' preserves primitive attrs; 'dot' is legacy fallback.",
    )
    return parser.parse_args()


def _load_callable(module_name: str, function_name: str):
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Callable '{function_name}' not found in module '{module_name}'.")
    return fn


def main() -> None:
    args = parse_args()

    args.dot_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.expected_output.parent.mkdir(parents=True, exist_ok=True)
    args.inputs_output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.inputs) as loaded:
        inputs = {name: np.asarray(loaded[name]) for name in loaded.files}
    function = _load_callable(args.module, args.function)

    graph, normalized_inputs, expected = capture_graph_from_mlx_function(
        dot_output_path=args.dot_output,
        inputs=inputs,
        function=function,
        allow_unknown_sources=args.allow_unknown_sources,
        capture_mode=args.capture_mode,
    )

    args.json_output.write_text(json.dumps(graph.to_dict(), indent=2) + "\n", encoding="utf-8")
    np.savez(args.inputs_output, **normalized_inputs)
    np.savez(args.expected_output, **expected)

    print(f"Wrote DOT graph: {args.dot_output}")
    print(f"Wrote parsed graph JSON: {args.json_output}")
    print(f"Wrote normalized inputs: {args.inputs_output}")
    print(f"Wrote expected outputs: {args.expected_output}")
    print(f"Captured ops ({len(graph.nodes)}): {[node.op for node in graph.nodes]}")
    print(f"Parsed outputs: {graph.outputs}")


if __name__ == "__main__":
    main()
