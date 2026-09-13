from __future__ import annotations

import errno
import hashlib
import math
from pathlib import Path

import pytest

from auk_local.audio import decode_float_audio, encode_float_audio
from auk_local.config import LocalPaths


@pytest.mark.parametrize("field,value", [
    ("sample_rate", 24000.9), ("frames", 1.9), ("channels", 1.9),
    ("frames", True), ("channels", True),
])
def test_audio_shape_does_not_silently_truncate_or_accept_booleans(field, value):
    audio = encode_float_audio([0.5], 24000)
    audio[field] = value
    with pytest.raises((ValueError, TypeError)):
        decode_float_audio(audio)


def test_audio_data_length_is_checked_before_base64_allocation(monkeypatch):
    import auk_local.audio as audio_module

    audio = encode_float_audio([0.5], 24000)
    audio["data"] *= 1000
    monkeypatch.setattr(audio_module.base64, "b64decode", lambda *_args, **_kwargs: pytest.fail("oversized data decoded"))
    with pytest.raises(ValueError, match="audio"):
        decode_float_audio(audio)


@pytest.mark.parametrize("rate", [24000.5, math.inf, True])
def test_audio_encoder_rejects_invalid_sample_rate(rate):
    with pytest.raises((ValueError, TypeError)):
        encode_float_audio([0.1], rate)


@pytest.mark.parametrize("broken", [b"bad", b"x" * 16])
def test_resume_repairs_corrupt_cached_file_without_redownloading_healthy_file(tmp_path, monkeypatch, broken):
    import huggingface_hub

    from auk_local import download

    paths = LocalPaths.from_root(tmp_path)
    staged = paths.models / ".TestModel.download-resume"
    staged.mkdir(parents=True)
    good = b"healthy-original"
    expected = b"correct-content!"
    (staged / "weights.bin").write_bytes(broken)
    (staged / "config.bin").write_bytes(good)
    entry = {
        "directory": "TestModel", "display_name": "test", "repo_id": "test/model", "revision": "fixed",
        "required": ["weights.bin", "config.bin"],
        "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in [("weights.bin", expected), ("config.bin", good)]},
    }
    monkeypatch.setattr(download, "load_model_manifest", lambda: {"models": {"flash": entry}})
    # Hugging Face trusts same-revision local metadata unless forced to download.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_kwargs: str(staged))
    repairs = []

    def repair(**kwargs):
        repairs.append(kwargs["filename"])
        assert kwargs["force_download"] is True
        assert kwargs["revision"] == "fixed"
        (kwargs["local_dir"] / kwargs["filename"]).write_bytes(expected)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", repair)
    download.download_models(paths, ("flash",), verify_hashes=True)
    assert repairs == ["weights.bin"]
    assert (paths.models / "TestModel" / "weights.bin").read_bytes() == expected
    assert (paths.models / "TestModel" / "config.bin").read_bytes() == good


def test_partial_input_write_is_removed_when_submit_fails(tmp_path, monkeypatch):
    from auk_local.manager import TaskManager

    manager = TaskManager(LocalPaths.from_root(tmp_path), start_worker=False)
    original = Path.write_bytes

    def disk_full(path, data):
        if path.name == "input.tmp":
            original(path, data[:4])
            raise OSError(errno.ENOSPC, "synthetic input disk full")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", disk_full)
    try:
        with pytest.raises(OSError, match="synthetic input disk full"):
            manager.submit({
                "task_key": "zero_shot_tts", "primary": "test", "generation_seconds": 1,
                "audio": encode_float_audio([0.1] * 2400, 24000),
            })
        assert not manager.store.list_recent()
        assert not list(manager.task_files.rglob("input.tmp"))
    finally:
        manager.close()
