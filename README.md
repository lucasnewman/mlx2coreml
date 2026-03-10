# mlx2coreml

`mlx2coreml` is an experimental MLX -> Core ML translation pipeline.

It captures MLX graphs, normalizes them into a small IR, lowers to MIL, and converts to Core ML artifacts (`.mlpackage` / `.mlmodelc`).

## Docs

- [docs/ops_status.md](./docs/ops_status.md): Supported op status

## Quick Start

```bash
pip install -U mlx2coreml
```

## Examples

### Converting [MLX LM](https://github.com/ml-explore/mlx-lm/tree/main) models to CoreML

```bash
python -m mlx2coreml.convert_mlx_lm \
  --model-id mlx-community/Qwen3-0.6B-bf16 \
  --output qwen3_0_6b_coreml \
  --seq-len 128 \
  --deployment-target iOS18 \
  --convert-to mlprogram \
  --compute-precision fp16 \
  --compute-units all
```

### Direct conversion from Python

For arbitrary models, use the conversion API directly:

```python
import numpy as np
import mlx.core as mx

from mlx_lm import load

import mlx2coreml as m2c

# Load an MLX model

model_id = "mlx-community/Qwen3-0.6B-bf16"
mlx_model, tokenizer = load(model_id, lazy=False)

# Build sample inputs for conversion

seq_len = 128
prompt = "hello"
input_ids = mx.array(tokenizer.encode(prompt), dtype=mx.int32)[None, ...]
inputs = {"input_ids": input_ids}

# Convert the model to a CoreML .mlprogram

config = m2c.ConversionConfig(
    deployment_target="iOS18",
    convert_to="mlprogram",
    compute_precision="fp16",
    flex_input_lens=[1, seq_len],
    flex_input_names={"input_ids"},
)

coreml_model = m2c.convert_mlx_to_coreml(
    mlx_model,
    inputs,
    config=config,
    capture_function=lambda input_ids: mlx_model(input_ids),
)
model_path = "model.mlpackage"
coreml_model.model.save(model_path)

# Optional: compile the model to a CoreML .mlmodelc

compiled_path = m2c.compile_mlmodel(model_path, "model.mlmodelc")
```
