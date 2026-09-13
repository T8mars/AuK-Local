from __future__ import annotations

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
