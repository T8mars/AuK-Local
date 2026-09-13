from __future__ import annotations

import os
import shutil
import uuid

from .config import LocalPaths, load_model_manifest
from .diagnostics import model_file_issues


def download_models(
    paths: LocalPaths,
    keys: tuple[str, ...] = ("flash", "base", "qwen"),
    *,
    verify_hashes: bool = False,
) -> None:
    from huggingface_hub import snapshot_download

    manifest = load_model_manifest()["models"]
    paths.models.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if key not in manifest:
            raise ValueError(f"未知模型：{key}")
        entry = manifest[key]
        target = paths.models / entry["directory"]
        backup = target.with_name(target.name + ".previous")
        # Recover a swap interrupted after moving the original directory aside.
        if backup.exists() and not target.exists():
            os.replace(backup, target)
        missing, invalid = model_file_issues(target, entry, verify_hashes=verify_hashes)
        if target.is_dir() and not missing and not invalid:
            print(f">> 已存在并通过必需文件检查：{target}")
            continue
        if invalid:
            print(">> 检测到损坏文件，将重新下载：" + ", ".join(invalid))
        candidates = sorted(
            paths.models.glob(f".{entry['directory']}.download-*"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        temporary = candidates[0] if candidates else paths.models / f".{entry['directory']}.download-{uuid.uuid4().hex}"
        print(f">> 下载 {entry['display_name']} @ {entry['revision']}")
        if candidates:
            print(f">> 续用暂存目录：{temporary}")
        try:
            snapshot_download(
                repo_id=entry["repo_id"],
                revision=entry["revision"],
                local_dir=temporary,
            )
            missing, invalid = model_file_issues(temporary, entry, verify_hashes=verify_hashes)
            if missing or invalid:
                raise RuntimeError("下载校验失败：" + ", ".join((*missing, *invalid)))
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except BaseException:
                if backup.exists() and not target.exists():
                    os.replace(backup, target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            print(f">> 完成：{target}")
        except BaseException:
            print(f">> 下载暂存保留用于续传：{temporary}")
            raise
