from __future__ import annotations

import ctypes
import hashlib
import os
import uuid
from pathlib import Path


def downloads_directory() -> Path:
    """Resolve the Windows Downloads known folder, including relocated folders."""
    if os.name == "nt":
        class GUID(ctypes.Structure):
            _fields_ = [("data", ctypes.c_ubyte * 16)]

        folder = GUID((ctypes.c_ubyte * 16).from_buffer_copy(
            uuid.UUID("374de290-123f-4565-9164-39c4925e467b").bytes_le,
        ))
        destination = ctypes.c_wchar_p()
        shell = ctypes.windll.shell32
        shell.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), ctypes.c_uint32,
                                               ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
        shell.SHGetKnownFolderPath.restype = ctypes.c_long
        result = shell.SHGetKnownFolderPath(ctypes.byref(folder), 0, None, ctypes.byref(destination))
        if result == 0:
            try:
                return Path(destination.value)
            finally:
                ctypes.windll.ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
                ctypes.windll.ole32.CoTaskMemFree.restype = None
                ctypes.windll.ole32.CoTaskMemFree(destination)
        raise OSError(f"无法读取 Windows 下载文件夹：{result}")
    return Path.home() / "Downloads"


def save_result_copy(result_path: str, outputs: Path, destination: Path | None = None) -> Path:
    source = Path(result_path).resolve(strict=True)
    if not source.is_relative_to(outputs.resolve()) or source.suffix.lower() != ".wav":
        raise ValueError("只能保存本整合包生成的 WAV 结果")
    target_directory = destination if destination is not None else downloads_directory()
    target_directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    target = target_directory / f"AuK_{source.parent.name}_{uuid.uuid4().hex[:8]}.wav"
    created = False
    try:
        with source.open("rb") as original, target.open("xb") as saved:
            created = True
            while chunk := original.read(1024 * 1024):
                digest.update(chunk)
                saved.write(chunk)
        saved_digest = hashlib.sha256()
        with target.open("rb") as saved:
            while chunk := saved.read(1024 * 1024):
                saved_digest.update(chunk)
        if digest.digest() != saved_digest.digest():
            raise OSError("保存后的音频校验不一致，请重试")
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


def open_outputs_directory(outputs: Path) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        raise OSError("请使用页面显示的输出目录打开文件夹")
    os.startfile(str(outputs.resolve()))
