# mlx2coreml

`mlx2coreml` is an experimental MLX -> Core ML translation pipeline.

It captures MLX graphs, normalizes them into a small IR, lowers to MIL, and converts to Core ML artifacts (`.mlpackage` / `.mlmodelc`).

## Docs

- [docs/ops_status.md](./docs/ops_status.md): Supported op status

## Quick Start

```bash
pip install -U coremltools mlx mlx-lm numpy
```

## Examples

### Convert `mlx-community/Qwen3-0.6B-bf16` to CoreML

```bash
python scripts/convert_mlx_lm_to_coreml.py \
  --model-id mlx-community/Qwen3-0.6B-bf16 \
  --seq-len 128 \
  --deployment-target iOS18 \
  --convert-to mlprogram \
  --compute-precision fp16 \
  --compute-units all \
  --run-name qwen3_0_6b_coreml
```

Outputs are written under `artifacts/<run-name>/`

### Python API

For custom pipelines, you can use the python API directly:

```python
from pathlib import Path

import coremltools as ct
import numpy as np
from mlx_lm import load

from mlx2coreml.from_mlx import capture_graph_from_mlx_function
from mlx2coreml.lower_to_mil import (
    build_mil_program,
    compile_model_artifact,
    convert_program_to_model,
)
from mlx2coreml.passes import normalize_graph


def build_input_ids(tokenizer, prompt: str, seq_len: int) -> np.ndarray:
    # TODO: tokenize prompt and truncate/pad to `seq_len`, dtype int32.
    ...


def select_primary_output(model_output):
    # TODO: if dict/list/tuple, pick a single tensor to capture.
    ...


model_id = "mlx-community/Qwen3-0.6B-bf16"
prompt = "hello"
seq_len = 128
out_dir = Path("artifacts/my_run")
out_dir.mkdir(parents=True, exist_ok=True)

model, tokenizer = load(model_id, lazy=False)
inputs = {"input_ids": build_input_ids(tokenizer, prompt, seq_len)}

graph, normalized_inputs, expected = capture_graph_from_mlx_function(
    dot_output_path=out_dir / "capture_graph.dot",
    inputs=inputs,
    function=lambda input_ids: select_primary_output(model(input_ids)),
    allow_unknown_sources=True,
    capture_mode="callback",
)

graph = normalize_graph(graph)
program = build_mil_program(graph, deployment_target=ct.target.iOS18, normalize=False)
coreml_model = convert_program_to_model(
    program,
    deployment_target=ct.target.iOS18,
    convert_to="mlprogram",
)

model_path = out_dir / "model.mlpackage"
coreml_model.save(str(model_path))
compiled_path = compile_model_artifact(model_path, out_dir / "model.mlmodelc")

print("Compiled model:", compiled_path)
```
