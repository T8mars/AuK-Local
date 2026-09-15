from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from auk_local.audio import encode_float_audio
from auk_local.config import LocalPaths
from auk_local.manager import TaskManager
from auk_local.store import TaskStore


def payload(**changes):
    return {"task_key": "instruct_tts", "primary": "复查", "generation_seconds": 1, **changes}


@pytest.mark.parametrize("variant", [str.upper, lambda s: s.replace("-", ""), lambda s: "{" + s + "}"])
def test_uuid_alias_cannot_create_second_task(tmp_path, variant):
    manager = TaskManager(LocalPaths.from_root(tmp_path), start_worker=False)
    try:
        task_id = str(uuid.uuid4())
        first, _ = manager.submit(payload(request_id=task_id))
        second, created = manager.submit(payload(request_id=variant(task_id)))
        assert not created
        assert second.request_id == first.request_id
        assert len(manager.store.list_recent()) == 1
    finally:
        manager.close()


@pytest.mark.parametrize("change", ["same_size", "truncated"])
def test_retry_rejects_modified_source(tmp_path, change):
    manager = TaskManager(LocalPaths.from_root(tmp_path), start_worker=False)
    try:
        task, _ = manager.submit(payload(task_key="zero_shot_tts", audio=encode_float_audio([0.25] * 2400, 24000)))
        manager.store.fail(task.request_id, "test")
        source = Path(task.input_path)
        source.write_bytes(b"\0" * (9600 if change == "same_size" else 4800))
        with pytest.raises(ValueError, match="输入音频.*损坏|输入音频.*校验"):
            manager.retry(task.request_id)
        assert len(manager.store.list_recent()) == 1
    finally:
        manager.close()


def test_store_closes_connections_without_waiting_for_gc(tmp_path, monkeypatch):
    original = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    store = TaskStore(tmp_path / "tasks.db")
    store.submit("test", {}, None)
    store.list_recent()
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_failed_model_promotion_restores_previous_directory(tmp_path, monkeypatch):
    import huggingface_hub

    from auk_local import download

    paths = LocalPaths.from_root(tmp_path)
    target = paths.models / "TestModel"
    target.mkdir(parents=True)
    (target / "old.bin").write_bytes(b"old")
    entry = {"directory": "TestModel", "repo_id": "test/model", "revision": "fixed", "display_name": "test"}
    monkeypatch.setattr(download, "load_model_manifest", lambda: {"models": {"flash": entry}})
    monkeypatch.setattr(download, "model_file_issues", lambda path, *_args, **_kwargs: (("new.bin",), ()) if path == target else ((), ()))

    def fake_download(**kwargs):
        directory = kwargs["local_dir"]
        directory.mkdir(parents=True)
        (directory / "new.bin").write_bytes(b"new")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    replace = download.os.replace

    def fail_promotion(source, destination):
        if ".download-" in str(source):
            raise PermissionError("synthetic promotion failure")
        return replace(source, destination)

    monkeypatch.setattr(download.os, "replace", fail_promotion)
    with pytest.raises(PermissionError, match="promotion failure"):
        download.download_models(paths, ("flash",))
    assert (target / "old.bin").read_bytes() == b"old"


@pytest.mark.parametrize("missing", ["audio", "metadata", "invalid_metadata"])
def test_ui_handles_cleaned_or_corrupt_result(tmp_path, missing):
    from auk_local.ui import build_ui

    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    try:
        demo = build_ui(manager, paths)
        run = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "run_task")
        updates = run("描述生成语音", "test", "", None, 1, "AuK-Flash（推荐）", 42, True, False, None)
        task_id = next(updates)[4]
        manager.store.claim_next()
        result = paths.outputs / "result.wav"
        metadata = paths.outputs / "metadata.json"
        if missing != "audio":
            result.write_bytes(b"audio")
        if missing != "metadata":
            metadata.write_text("broken{" if missing == "invalid_metadata" else "{}", encoding="utf-8")
        manager.store.complete(task_id, str(result), str(metadata))
        final = list(updates)[-1]
        if missing == "audio":
            assert final[1] is None
            assert "不可用" in final[0]
        else:
            assert final[1] == str(result)
            assert "参数文件" in final[0]
            assert final[3] == ""
    finally:
        manager.close()


def test_inference_rejects_changed_input_before_loading_model(tmp_path, monkeypatch):
    from auk_local.inference import InferenceRuntime

    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    try:
        record, _ = manager.submit(payload(task_key="zero_shot_tts", audio=encode_float_audio([0.1] * 2400, 24000)))
        Path(record.input_path).write_bytes(b"\0" * 9600)
        runtime = InferenceRuntime(paths)
        loads = []
        monkeypatch.setattr(runtime, "_load", lambda *_args: loads.append(True))
        with pytest.raises(ValueError, match="输入音频校验失败"):
            runtime.execute(record.request, record.input_path, paths.outputs / record.request_id, lambda _: None)
        assert not loads
    finally:
        manager.close()


def test_download_recovers_interrupted_swap_without_downloading(tmp_path, monkeypatch):
    import huggingface_hub

    from auk_local import download

    paths = LocalPaths.from_root(tmp_path)
    backup = paths.models / "TestModel.previous"
    backup.mkdir(parents=True)
    (backup / "valid.bin").write_bytes(b"valid")
    entry = {"directory": "TestModel", "display_name": "test"}
    monkeypatch.setattr(download, "load_model_manifest", lambda: {"models": {"flash": entry}})
    monkeypatch.setattr(download, "model_file_issues", lambda *_args, **_kwargs: ((), ()))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_kwargs: pytest.fail("must reuse recovered model"))
    download.download_models(paths, ("flash",))
    assert (paths.models / "TestModel" / "valid.bin").read_bytes() == b"valid"
    assert not backup.exists()


def test_corrupt_metadata_api_returns_actionable_response(tmp_path):
    from fastapi.testclient import TestClient

    from auk_local.service import create_app, load_or_create_token

    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    headers = {"X-AuK-Token": load_or_create_token(paths)}
    record, _ = manager.submit(payload())
    manager.store.claim_next()
    metadata = paths.outputs / "metadata.json"
    metadata.write_bytes(b"{broken")
    manager.store.complete(record.request_id, str(paths.outputs / "result.wav"), str(metadata))
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        response = client.get(f"/api/v1/tasks/{record.request_id}/metadata", headers=headers)
        assert response.status_code == 410
        assert "损坏" in response.json()["detail"]


def test_ui_can_reopen_successful_history_without_resubmitting(tmp_path):
    from auk_local.ui import build_ui

    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    try:
        record, _ = manager.submit(payload())
        manager.store.claim_next()
        result = paths.outputs / "result.wav"
        result.write_bytes(b"test")
        metadata = paths.outputs / "metadata.json"
        metadata.write_text("{}", encoding="utf-8")
        manager.store.complete(record.request_id, str(result), str(metadata))
        demo = build_ui(manager, paths)
        view = next(fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "view_task")
        final = list(view(record.request_id, None))[-1]
        assert final[1] == str(result)
        assert len(manager.store.list_recent()) == 1
        seed = next(block for block in demo.blocks.values() if (getattr(block, "label", "") or "").startswith("固定 Seed"))
        assert seed.get_block_name() == "textbox"
    finally:
        manager.close()


def test_local_client_wait_reports_paused_but_preserves_success(tmp_path, monkeypatch):
    from auk_local.client import LocalClient

    client = LocalClient("http://127.0.0.1:7860", tmp_path / "token")
    result = {"state": "queued", "scheduler": {"state": "paused"}}
    monkeypatch.setattr(client, "status", lambda _: result)
    with pytest.raises(RuntimeError, match="调度已暂停"):
        client.wait("test", timeout=0.1, poll=0.01)
    result["state"] = "succeeded"
    assert client.wait("test")["state"] == "succeeded"


def test_manager_wait_reports_paused(tmp_path):
    manager = TaskManager(LocalPaths.from_root(tmp_path), start_worker=False)
    try:
        record, _ = manager.submit(payload())
        manager._set_scheduler_state("paused", "synthetic storage failure")
        with pytest.raises(RuntimeError, match="调度已暂停"):
            manager.wait(record.request_id, timeout=0.1, poll=0.01)
    finally:
        manager.close()
