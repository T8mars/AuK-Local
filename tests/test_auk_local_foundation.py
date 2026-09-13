from __future__ import annotations

import urllib.request
from array import array
from pathlib import Path

import numpy as np
import pytest

from auk_local.audio import decode_float_audio, encode_float_audio, encode_gradio_audio, validate_duration
from auk_local.client import LocalClient, validate_loopback_url
from auk_local.config import LocalPaths, default_home
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


def test_gradio_int16_audio_is_normalized_before_submission():
    source = np.array([0, 16_384, -16_384], dtype=np.int16)
    decoded = decode_float_audio(encode_gradio_audio((24_000, source)))
    assert decoded.samples.tolist() == pytest.approx([0.0, 0.5, -0.5])


def test_default_home_is_the_package_root(monkeypatch):
    monkeypatch.delenv("AUK_LOCAL_HOME", raising=False)
    assert default_home() == Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:7860", "http://localhost:7860", "http://[::1]:7860"],
)
def test_client_accepts_only_loopback_http(url):
    assert validate_loopback_url(url) == url
    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_url("https://example.invalid")


def test_client_disables_environment_proxies(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    token = tmp_path / "token"
    token.write_text("test", encoding="utf-8")
    client = LocalClient("http://127.0.0.1:7860", token)
    proxy_handlers = [handler for handler in client._opener.handlers if isinstance(handler, urllib.request.ProxyHandler)]
    assert not any(handler.proxies for handler in proxy_handlers)


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
