from __future__ import annotations

import shutil
from pathlib import Path

import coremltools as ct


def compile_mlmodel(model_path: Path | str, compiled_path: Path | str | None = None) -> Path:
    model_path = Path(model_path).resolve()
    compiled_temp = Path(ct.models.utils.compile_model(str(model_path))).resolve()
    if compiled_path is None:
        return compiled_temp

    compiled_path = Path(compiled_path).resolve()
    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    compiled_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(compiled_temp, compiled_path)
    return compiled_path


__all__ = ["compile_mlmodel"]
