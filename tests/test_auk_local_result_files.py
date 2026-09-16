from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from auk_local.result_files import save_result_copy


def test_save_copy_preserves_bytes_and_never_overwrites(tmp_path):
    outputs = tmp_path / "outputs"
    source = outputs / "task-123" / "result.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"RIFF" + bytes(range(256)) * 5000)
    destination = tmp_path / "downloads"
    first = save_result_copy(str(source), outputs, destination)
    second = save_result_copy(str(source), outputs, destination)
    assert first != second
    assert first.read_bytes() == second.read_bytes() == source.read_bytes()
    assert "task-123" in first.name


def test_save_copy_rejects_file_outside_outputs(tmp_path):
    source = tmp_path / "private.wav"
    source.write_bytes(b"audio")
    with pytest.raises(ValueError, match="WAV"):
        save_result_copy(str(source), tmp_path / "outputs", tmp_path / "downloads")
    assert not (tmp_path / "downloads").exists()


def test_failed_exclusive_create_preserves_existing_file(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    source = outputs / "task" / "result.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"new audio")
    destination = tmp_path / "downloads"
    destination.mkdir()
    existing = destination / "AuK_task_12345678.wav"
    existing.write_bytes(b"existing audio")
    monkeypatch.setattr("auk_local.result_files.uuid.uuid4", lambda: SimpleNamespace(hex="12345678"))
    with pytest.raises(FileExistsError):
        save_result_copy(str(source), outputs, destination)
    assert existing.read_bytes() == b"existing audio"


def test_partial_copy_is_removed_on_write_failure(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    source = outputs / "task" / "result.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"audio")
    destination = tmp_path / "downloads"
    original_open = Path.open

    class FailedWriter:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, data):
            raise OSError("disk full")

    def broken_open(path, mode="r", *args, **kwargs):
        if mode == "xb":
            path.touch()
            return FailedWriter()
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", broken_open)
    with pytest.raises(OSError, match="disk full"):
        save_result_copy(str(source), outputs, destination)
    assert list(destination.iterdir()) == []
