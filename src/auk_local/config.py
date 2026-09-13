from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


def default_home() -> Path:
    configured = os.environ.get("AUK_LOCAL_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def configure_bundled_tools(paths: LocalPaths) -> None:
    """Prefer portable helper executables shipped with the package."""
    bundled = paths.root / "runtime" / "ffmpeg"
    if not bundled.is_dir():
        return
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    if str(bundled).casefold() not in {entry.casefold() for entry in entries}:
        os.environ["PATH"] = str(bundled) + (os.pathsep + current if current else "")


@dataclass(frozen=True)
class LocalPaths:
    root: Path
    models: Path
    data: Path
    outputs: Path
    logs: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> LocalPaths:
        resolved = (root or default_home()).resolve()
        return cls(
            root=resolved,
            models=resolved / "models",
            data=resolved / "data",
            outputs=resolved / "outputs",
            logs=resolved / "logs",
        )

    def ensure_writable_dirs(self) -> None:
        for path in (self.models, self.data, self.outputs, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def load_model_manifest() -> dict[str, Any]:
    resource = files("auk_local").joinpath("model-manifest.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def model_paths(paths: LocalPaths, model_key: str) -> dict[str, Path]:
    entry = load_model_manifest()["models"][model_key]
    directory = paths.models / entry["directory"]
    result = {"directory": directory}
    if "checkpoint" in entry:
        result["checkpoint"] = directory / entry["checkpoint"]
        result["config"] = directory / "config.yaml"
    return result
