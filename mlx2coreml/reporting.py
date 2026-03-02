from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "mlx2coreml.run_report.v1"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def collect_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "coremltools": _package_version("coremltools"),
        "numpy": _package_version("numpy"),
        "mlx": _package_version("mlx"),
    }


def build_run_context(
    *,
    run_kind: str,
    deployment_target: str,
    convert_to: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_kind": run_kind,
        "deployment_target": deployment_target,
        "convert_to": convert_to,
        "seed": int(seed),
        "versions": collect_versions(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
