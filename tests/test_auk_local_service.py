from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from auk_local.config import LocalPaths
from auk_local.manager import TaskManager
from auk_local.service import create_app, load_or_create_token


def test_health_works_without_models_or_worker(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert not response.json()["inference_ready"]


def test_authenticated_submit_and_cancel(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    token = load_or_create_token(paths)
    headers = {"X-AuK-Token": token}
    payload = {
        "request_id": str(uuid.uuid4()),
        "task_key": "instruct_tts",
        "primary": "你好",
        "secondary": "平静女声",
        "generation_seconds": 1.0,
        "model": "flash",
        "seed": 42,
    }
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        assert client.post("/api/v1/tasks", json=payload).status_code == 401
        created = client.post("/api/v1/tasks", json=payload, headers=headers)
        assert created.status_code == 200
        assert created.json()["state"] == "queued"
        cancelled = client.post(f"/api/v1/tasks/{payload['request_id']}/cancel", json={}, headers=headers)
        assert cancelled.json()["state"] == "cancelled"


def test_idempotency_rejects_same_id_with_different_request(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    token = load_or_create_token(paths)
    headers = {"X-AuK-Token": token}
    request_id = str(uuid.uuid4())
    base = {
        "request_id": request_id,
        "task_key": "instruct_tts",
        "primary": "第一次",
        "generation_seconds": 1.0,
    }
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        assert client.post("/api/v1/tasks", json=base, headers=headers).status_code == 200
        changed = {**base, "primary": "第二次"}
        assert client.post("/api/v1/tasks", json=changed, headers=headers).status_code == 422


def test_recent_tasks_and_retry_failed_task(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    token = load_or_create_token(paths)
    headers = {"X-AuK-Token": token}
    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "task_key": "instruct_tts",
        "primary": "重试测试",
        "generation_seconds": 1.0,
    }
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        client.post("/api/v1/tasks", json=payload, headers=headers).raise_for_status()
        assert manager.store.fail(request_id, "synthetic failure")
        recent = client.get("/api/v1/tasks", headers=headers)
        assert recent.status_code == 200
        assert recent.json()[0]["request_id"] == request_id
        retried = client.post(f"/api/v1/tasks/{request_id}/retry", json={}, headers=headers)
        assert retried.status_code == 200
        assert retried.json()["request_id"] != request_id
        assert retried.json()["state"] == "queued"
