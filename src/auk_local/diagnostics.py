from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import LocalPaths, load_model_manifest


@dataclass(frozen=True)
class ModelDiagnostic:
    key: str
    display_name: str
    directory: str
    revision: str
    status: str
    missing_files: tuple[str, ...]
    invalid_files: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file_issues(directory: Path, entry: dict[str, Any], *, verify_hashes: bool = False):
    missing = tuple(name for name in entry["required"] if not (directory / name).is_file())
    invalid: list[str] = []
    for name, expected in entry.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            continue
        if path.stat().st_size != int(expected["size"]):
            invalid.append(f"{name}:size")
        elif verify_hashes and _sha256(path) != expected["sha256"]:
            invalid.append(f"{name}:sha256")
    return missing, tuple(invalid)


def inspect_models(paths: LocalPaths, *, verify_hashes: bool = False) -> list[ModelDiagnostic]:
    diagnostics: list[ModelDiagnostic] = []
    for key, entry in load_model_manifest()["models"].items():
        directory = paths.models / entry["directory"]
        missing, invalid = model_file_issues(directory, entry, verify_hashes=verify_hashes)
        if not directory.exists():
            status = "missing"
        elif missing:
            status = "incomplete"
        elif invalid:
            status = "corrupt"
        else:
            status = "ready"
        diagnostics.append(
            ModelDiagnostic(
                key=key,
                display_name=entry["display_name"],
                directory=str(directory),
                revision=entry["revision"],
                status=status,
                missing_files=missing,
                invalid_files=invalid,
            )
        )
    return diagnostics


def runtime_diagnostic(
    paths: LocalPaths,
    *,
    probe_torch: bool = True,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    paths.ensure_writable_dirs()
    disk = shutil.disk_usage(paths.root)
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "root": str(paths.root),
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "models": [asdict(item) for item in inspect_models(paths, verify_hashes=verify_hashes)],
        "model_hashes_verified": verify_hashes,
        "torch": {"installed": importlib.util.find_spec("torch") is not None},
    }
    if probe_torch and result["torch"]["installed"]:
        try:
            import torch

            cuda = torch.cuda.is_available()
            result["torch"].update(
                {
                    "version": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cuda_available": cuda,
                    "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
                    "device": torch.cuda.get_device_name(0) if cuda else None,
                    "device_memory_gib": (
                        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda else None
                    ),
                }
            )
        except Exception as exc:
            result["torch"]["load_error"] = f"{type(exc).__name__}: {exc}"
    return result


def diagnostic_json(paths: LocalPaths, *, probe_torch: bool = True, verify_hashes: bool = False) -> str:
    return json.dumps(
        runtime_diagnostic(paths, probe_torch=probe_torch, verify_hashes=verify_hashes),
        ensure_ascii=False,
        indent=2,
    )
