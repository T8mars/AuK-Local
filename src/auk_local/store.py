from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import LocalPaths


ACTIVE_STATES = ("loading", "encoding", "sampling", "decoding", "saving", "cancelling")
FINAL_STATES = ("succeeded", "failed", "cancelled", "interrupted")


@dataclass(frozen=True)
class TaskRecord:
    request_id: str
    state: str
    phase: str
    request: dict[str, Any]
    input_path: str | None
    result_path: str | None
    metadata_path: str | None
    error: str | None
    created_at: float
    updated_at: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "state": self.state,
            "phase": self.phase,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result_available": bool(self.result_path and Path(self.result_path).is_file()),
            "metadata_available": bool(self.metadata_path and Path(self.metadata_path).is_file()),
        }


class TaskStore:
    def __init__(self, database: Path):
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    request_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    input_path TEXT,
                    result_path TEXT,
                    metadata_path TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_tasks_queue ON tasks(state, created_at)")

    @staticmethod
    def _row(row: sqlite3.Row | None) -> TaskRecord | None:
        if row is None:
            return None
        return TaskRecord(
            request_id=row["request_id"],
            state=row["state"],
            phase=row["phase"],
            request=json.loads(row["request_json"]),
            input_path=row["input_path"],
            result_path=row["result_path"],
            metadata_path=row["metadata_path"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def submit(self, request_id: str, request: dict[str, Any], input_path: str | None) -> tuple[TaskRecord, bool]:
        now = time.time()
        encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            existing = self._row(connection.execute("SELECT * FROM tasks WHERE request_id=?", (request_id,)).fetchone())
            if existing:
                if existing.request != request:
                    raise ValueError("request_id 已被不同参数使用")
                return existing, False
            connection.execute(
                "INSERT INTO tasks VALUES (?, 'queued', 'queued', ?, ?, NULL, NULL, NULL, ?, ?)",
                (request_id, encoded, input_path, now, now),
            )
        return self.get(request_id), True

    def get(self, request_id: str) -> TaskRecord:
        with self._connect() as connection:
            record = self._row(connection.execute("SELECT * FROM tasks WHERE request_id=?", (request_id,)).fetchone())
        if record is None:
            raise KeyError(request_id)
        return record

    def claim_next(self) -> TaskRecord | None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT request_id FROM tasks WHERE state='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            request_id = row["request_id"]
            now = time.time()
            connection.execute(
                "UPDATE tasks SET state='loading', phase='loading', updated_at=? WHERE request_id=? AND state='queued'",
                (now, request_id),
            )
            connection.commit()
        return self.get(request_id)

    def set_phase(self, request_id: str, phase: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE tasks SET state=?, phase=?, updated_at=? WHERE request_id=?
                AND state NOT IN ('succeeded','failed','cancelled','interrupted','cancelling')""",
                (phase, phase, time.time(), request_id),
            )
            return cursor.rowcount == 1

    def complete(self, request_id: str, result_path: str, metadata_path: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE tasks SET state='succeeded', phase='succeeded', result_path=?, metadata_path=?,
                error=NULL, updated_at=? WHERE request_id=? AND state IN ('loading','encoding','sampling','decoding','saving')""",
                (result_path, metadata_path, time.time(), request_id),
            )
            return cursor.rowcount == 1

    def fail(self, request_id: str, error: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE tasks SET state='failed', phase='failed', error=?, updated_at=?
                WHERE request_id=? AND state IN ('queued','loading','encoding','sampling','decoding','saving')""",
                (error, time.time(), request_id),
            )
            return cursor.rowcount == 1

    def rebase_managed_paths(self, paths: LocalPaths) -> int:
        """Repair task-owned absolute paths after the package is moved."""
        changed = 0
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT request_id, input_path, result_path, metadata_path FROM tasks"
            ).fetchall()
            for row in rows:
                values = (
                    str(paths.data / "tasks" / row["request_id"] / "input.f32") if row["input_path"] else None,
                    str(paths.outputs / row["request_id"] / "result.wav") if row["result_path"] else None,
                    str(paths.outputs / row["request_id"] / "metadata.json") if row["metadata_path"] else None,
                )
                if values != (row["input_path"], row["result_path"], row["metadata_path"]):
                    connection.execute(
                        "UPDATE tasks SET input_path=?, result_path=?, metadata_path=? WHERE request_id=?",
                        (*values, row["request_id"]),
                    )
                    changed += 1
        return changed

    def cancel(self, request_id: str) -> str:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state FROM tasks WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(request_id)
            state = row["state"]
            now = time.time()
            if state == "queued":
                connection.execute(
                    "UPDATE tasks SET state='cancelled', phase='cancelled', updated_at=? WHERE request_id=?",
                    (now, request_id),
                )
                result = "cancelled"
            elif state in ACTIVE_STATES and state != "cancelling":
                connection.execute(
                    "UPDATE tasks SET state='cancelling', phase='cancelling', updated_at=? WHERE request_id=?",
                    (now, request_id),
                )
                result = "cancelling"
            else:
                result = state
            connection.commit()
            return result

    def finish_cancel(self, request_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET state='cancelled', phase='cancelled', updated_at=? WHERE request_id=? AND state='cancelling'",
                (time.time(), request_id),
            )
            return cursor.rowcount == 1

    def recover_after_restart(self) -> int:
        placeholders = ",".join("?" for _ in ACTIVE_STATES)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"""UPDATE tasks SET state='interrupted', phase='interrupted',
                error='服务重启中断了运行任务', updated_at=? WHERE state IN ({placeholders})""",
                (time.time(), *ACTIVE_STATES),
            )
            return cursor.rowcount

    def list_recent(self, limit: int = 50) -> list[TaskRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]
