from __future__ import annotations

import uuid
import sqlite3

import numpy as np

import pytest
from fastapi.testclient import TestClient

from auk_local.config import LocalPaths
from auk_local.manager import TaskManager
from auk_local.audio import encode_float_audio
from auk_local.service import create_app, load_or_create_token


def test_health_works_without_models_or_worker(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert not response.json()["inference_ready"]


def test_local_api_defaults_to_high_quality_base_model(tmp_path):
    manager = TaskManager(LocalPaths.from_root(tmp_path), start_worker=False)
    try:
        normalized = manager.normalize_request(
            {
                "request_id": str(uuid.uuid4()),
                "task_key": "instruct_tts",
                "primary": "测试",
                "secondary": "自然清晰的女声",
                "generation_seconds": 1.0,
            }
        )
        assert normalized["model"] == "base"
    finally:
        manager.close()


def test_automatic_edit_accepts_and_ignores_stale_48_second_widget_after_crop(tmp_path):
    manager = TaskManager(LocalPaths.from_root(tmp_path), start_worker=False)
    try:
        record, created = manager.submit(
            {
                "request_id": str(uuid.uuid4()),
                "task_key": "deaccent",
                "primary": "去掉方言口音",
                "generation_seconds": 48.0,
                "duration_mode": "source_auto",
                "audio": encode_float_audio(np.zeros(24_000 * 4, dtype=np.float32), 24_000),
            }
        )
        assert created is True
        assert record.request["source_frames"] == 24_000 * 4
        assert record.request["generation_seconds"] == 48.0
    finally:
        manager.close()


def test_manual_task_still_rejects_out_of_range_duration(tmp_path):
    manager = TaskManager(LocalPaths.from_root(tmp_path), start_worker=False)
    try:
        with pytest.raises(ValueError, match="0 到 30"):
            manager.normalize_request(
                {
                    "request_id": str(uuid.uuid4()),
                    "task_key": "zero_shot_tts",
                    "primary": "测试",
                    "generation_seconds": 48.0,
                    "duration_mode": "manual",
                }
            )
    finally:
        manager.close()


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


def test_invalid_types_and_sampling_ranges_return_422(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    token = load_or_create_token(paths)
    headers = {"X-AuK-Token": token}
    base = {
        "request_id": str(uuid.uuid4()),
        "task_key": "instruct_tts",
        "primary": "校验",
        "generation_seconds": 1.0,
        "model": "base",
    }
    invalid_payloads = [
        {**base, "request_id": str(uuid.uuid4()), "generation_seconds": None},
        {**base, "request_id": str(uuid.uuid4()), "nfe_steps": -7},
        {**base, "request_id": str(uuid.uuid4()), "cfg_strength": -999},
        {
            **base,
            "request_id": str(uuid.uuid4()),
            "task_key": "zero_shot_tts",
            "audio": "not-an-object",
        },
        {
            **base,
            "request_id": str(uuid.uuid4()),
            "task_key": "zero_shot_tts",
            "audio": {"encoding": "f32le", "sample_rate": None, "channels": 1, "frames": 1, "data": "AAAAAA=="},
        },
    ]
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        for payload in invalid_payloads:
            response = client.post("/api/v1/tasks", json=payload, headers=headers)
            assert response.status_code == 422, response.text


@pytest.mark.parametrize("seed", [9_007_199_254_740_993, 9_223_372_036_854_775_807])
def test_seed_keeps_full_integer_precision(tmp_path, seed):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    try:
        normalized = manager.normalize_request(
            {
                "request_id": str(uuid.uuid4()),
                "task_key": "instruct_tts",
                "primary": "整数精度",
                "generation_seconds": 1,
                "seed": seed,
            }
        )
        assert normalized["seed"] == seed
    finally:
        manager.close()


def test_paused_scheduler_is_reported_and_rejects_new_tasks(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    headers = {"X-AuK-Token": load_or_create_token(paths)}
    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        manager._set_scheduler_state("paused", "synthetic storage failure")
        health = client.get("/api/v1/health").json()
        assert health["status"] == "degraded"
        assert not health["inference_ready"]
        assert not health["scheduler"]["accepting_tasks"]
        response = client.post("/api/v1/tasks", headers=headers, json={
            "task_key": "instruct_tts", "primary": "test", "generation_seconds": 1,
        })
        assert response.status_code == 503
        assert not manager.store.list_recent()


def test_unreadable_task_database_returns_503(tmp_path, monkeypatch):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    headers = {"X-AuK-Token": load_or_create_token(paths)}

    def unavailable(*_args):
        raise sqlite3.OperationalError("synthetic database unavailable")

    with TestClient(create_app(paths, with_ui=False, manager=manager)) as client:
        monkeypatch.setattr(manager.store, "list_recent", unavailable)
        assert client.get("/api/v1/tasks", headers=headers).status_code == 503
