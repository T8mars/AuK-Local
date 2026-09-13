from __future__ import annotations

from array import array

import pytest

from auk_local.audio import decode_float_audio, encode_float_audio, validate_duration
from auk_local.config import LocalPaths
from auk_local.diagnostics import inspect_models
from auk_local.task_templates import TASKS, build_instruction


def test_all_sixteen_tasks_build_instructions():
    assert len(TASKS) == 16
    for task in TASKS:
        instruction = build_instruction(task.key, "测试内容", "补充条件")
        assert "测试内容" in instruction


def test_float_audio_round_trip_is_lossless():
    original = array("f", [-1.0, -0.25, 0.0, 0.5, 1.0])
    decoded = decode_float_audio(encode_float_audio(original, 24_000))
    assert decoded.sample_rate == 24_000
    assert decoded.samples.tolist() == original.tolist()


def test_duration_budget_rejects_overflow():
    validate_duration(12.0, 18.0)
    with pytest.raises(ValueError, match="超过"):
        validate_duration(12.0, 18.01)


def test_missing_models_do_not_break_diagnostics(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    result = inspect_models(paths)
    assert {item.key for item in result} == {"flash", "base", "qwen"}
    assert all(item.status == "missing" for item in result)


def test_model_size_mismatch_is_reported_as_corrupt(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    flash = paths.models / "AuK-Flash"
    flash.mkdir(parents=True)
    (flash / "auk_flash.safetensors").write_bytes(b"bad")
    (flash / "vae.safetensors").write_bytes(b"bad")
    (flash / "config.yaml").write_bytes(b"bad")
    result = {item.key: item for item in inspect_models(paths)}
    assert result["flash"].status == "corrupt"
    assert result["flash"].invalid_files
