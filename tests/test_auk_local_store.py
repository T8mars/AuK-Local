from __future__ import annotations

import threading

import pytest

from auk_local.config import LocalPaths
from auk_local.manager import TaskManager
from auk_local.store import TaskStore


def _request(request_id: str):
    return {"request_id": request_id, "instruction": "test", "generation_seconds": 1.0}


def test_submit_is_idempotent_and_fifo(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    first, created = store.submit("a", _request("a"), None)
    assert created and first.state == "queued"
    same, created = store.submit("a", _request("a"), None)
    assert not created and same.request_id == "a"
    store.submit("b", _request("b"), None)
    assert store.claim_next().request_id == "a"


def test_cancel_wins_before_success_commit(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    store.submit("a", _request("a"), None)
    store.claim_next()
    assert store.cancel("a") == "cancelling"
    assert not store.complete("a", "result.wav", "metadata.json")
    assert store.finish_cancel("a")
    assert store.get("a").state == "cancelled"


def test_success_commit_wins_before_cancel(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    store.submit("a", _request("a"), None)
    store.claim_next()
    assert store.complete("a", "result.wav", "metadata.json")
    assert store.cancel("a") == "succeeded"


def test_restart_marks_running_interrupted_but_keeps_queue(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    store.submit("running", _request("running"), None)
    store.claim_next()
    store.submit("queued", _request("queued"), None)
    assert store.recover_after_restart() == 1
    assert store.get("running").state == "interrupted"
    assert store.get("queued").state == "queued"


def test_second_manager_cannot_recover_live_instance(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    first = TaskManager(paths, start_worker=False)
    try:
        with pytest.raises(RuntimeError, match="另一个 AuK Local"):
            TaskManager(paths, start_worker=False)
    finally:
        first.close()
    replacement = TaskManager(paths, start_worker=False)
    replacement.close()


def test_cancel_after_worker_done_reaches_final_state(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)

    class FinishedThenCancelledSupervisor:
        def run(self, task, _input_path, output_dir, _on_phase):
            output_dir.mkdir(parents=True)
            result = output_dir / "result.wav"
            metadata = output_dir / "metadata.json"
            result.write_bytes(b"wav")
            metadata.write_text("{}", encoding="utf-8")
            assert manager.store.cancel(task["request_id"]) == "cancelling"
            return {"result_path": str(result), "metadata_path": str(metadata)}

        def cancel(self, _request_id):
            return False

        def close(self):
            return None

    manager.supervisor = FinishedThenCancelledSupervisor()
    manager._dispatcher = threading.Thread(target=manager._dispatch_loop, daemon=True)
    manager._dispatcher.start()
    request_id = "00000000-0000-0000-0000-000000000001"
    try:
        manager.submit(
            {
                "request_id": request_id,
                "task_key": "instruct_tts",
                "primary": "取消竞争",
                "generation_seconds": 1.0,
            }
        )
        completed = manager.wait(request_id, timeout=5)
        assert completed.state == "cancelled"
        assert not (paths.outputs / request_id / "result.wav").exists()
    finally:
        manager.close()
