from __future__ import annotations

import errno
from pathlib import Path

import numpy as np
import pytest
import torch

from auk_local.config import LocalPaths
from auk_local.inference import InferenceRuntime
from auk_local.manager import TaskManager
from auk_local.ui import build_ui


def task():
    return TaskManager.normalize_request({
        "task_key": "instruct_tts", "primary": "test", "generation_seconds": 0.2,
        "keep_loaded": True,
    })


def fake_runtime(tmp_path, monkeypatch, waveform):
    runtime = InferenceRuntime(LocalPaths.from_root(tmp_path))

    class Engine:
        def generate(self, *_args, **_kwargs):
            return waveform, 24000

    runtime.engine = Engine()
    runtime.loaded_key = "flash:1"
    monkeypatch.setattr(runtime, "_load", lambda *_args: runtime.engine)
    return runtime


@pytest.mark.parametrize("waveform", [torch.full((1, 4800), float("nan")), torch.zeros((1, 0))])
def test_invalid_generated_audio_is_not_saved_as_success(tmp_path, monkeypatch, waveform):
    runtime = fake_runtime(tmp_path, monkeypatch, waveform)
    with pytest.raises(ValueError, match="输出音频"):
        runtime.execute(task(), None, tmp_path / "output", lambda _: None)
    assert not (tmp_path / "output" / "result.wav").exists()
    assert runtime.engine is None


def test_metadata_disk_full_removes_partial_audio(tmp_path, monkeypatch):
    runtime = fake_runtime(tmp_path, monkeypatch, torch.zeros((1, 4800)))
    original = Path.write_text

    def fail_metadata(path, *args, **kwargs):
        if path.name == "metadata.json":
            raise OSError(errno.ENOSPC, "synthetic disk full")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError, match="synthetic disk full"):
        runtime.execute(task(), None, tmp_path / "output", lambda _: None)
    assert not (tmp_path / "output" / "result.wav").exists()
    assert runtime.engine is None


def test_loading_failure_runs_cleanup(tmp_path, monkeypatch):
    runtime = InferenceRuntime(LocalPaths.from_root(tmp_path))
    cleanups = []

    def fail_load(*_args):
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(runtime, "_load", fail_load)
    monkeypatch.setattr(runtime, "unload", lambda: cleanups.append(True))
    with pytest.raises(RuntimeError, match="synthetic load failure"):
        runtime.execute(task(), None, tmp_path / "output", lambda _: None)
    assert cleanups == [True]


def test_bad_ui_audio_returns_submission_error(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    try:
        demo = build_ui(manager, paths)
        run = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "run_task")
        updates = list(run(
            "参考声音克隆", "test", "", (24000, np.array([float("nan")])),
            1.0, "AuK-Flash（推荐）", 42, True, False, None,
        ))
        assert updates[0][0].startswith("提交失败：")
        assert not manager.store.list_recent()
    finally:
        manager.close()


def test_ui_stops_waiting_when_scheduler_pauses(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    try:
        demo = build_ui(manager, paths)
        run = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "run_task")
        updates = run("描述生成语音", "test", "", None, 1.0, "AuK-Flash（推荐）", 42, True, False, None)
        assert "queued" in next(updates)[0]
        manager._set_scheduler_state("paused", "synthetic storage failure")
        assert "任务已暂停" in next(updates)[0]
        with pytest.raises(StopIteration):
            next(updates)
    finally:
        manager.close()


def test_ui_preserves_completed_result_when_scheduler_pauses(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    try:
        demo = build_ui(manager, paths)
        run = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "run_task")
        updates = run("描述生成语音", "test", "", None, 1.0, "AuK-Flash（推荐）", 42, True, False, None)
        request_id = next(updates)[4]
        manager.store.claim_next()
        result = tmp_path / "result.wav"
        metadata = tmp_path / "metadata.json"
        result.write_bytes(b"synthetic-result")
        metadata.write_text("{}", encoding="utf-8")
        manager.store.complete(request_id, str(result), str(metadata))
        manager._set_scheduler_state("paused", "unrelated request failed")
        final = list(updates)[-1]
        assert final[0] == "生成完成"
        assert final[1] == str(result)
    finally:
        manager.close()
