from __future__ import annotations

import base64
import hashlib
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .audio import decode_float_audio, read_verified_source, validate_duration
from .config import LocalPaths
from .store import TaskRecord, TaskStore
from .task_templates import TASK_BY_KEY, build_instruction
from .worker import TaskCancelled, WorkerSupervisor


logger = logging.getLogger(__name__)


class _StorageUnavailable(RuntimeError):
    pass


class _SingleInstanceLock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a+b")
        try:
            self._handle.seek(0, os.SEEK_END)
            if self._handle.tell() == 0:
                self._handle.write(b"\0")
                self._handle.flush()
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError("另一个 AuK Local 服务正在使用这个整合包") from exc

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} 必须是数字")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} 必须是有限数字")
    return result


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} 必须是整数")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} 必须是整数") from exc
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} 必须是有限整数")
        if value.is_integer():
            return int(value)
    raise ValueError(f"{field} 必须是整数")


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} 必须是布尔值")
    return value


class TaskManager:
    STORAGE_ATTEMPTS = 4
    STORAGE_RETRY_DELAY = 0.1

    def __init__(self, paths: LocalPaths, *, start_worker: bool = True):
        self.paths = paths
        paths.ensure_writable_dirs()
        self._instance_lock = _SingleInstanceLock(paths.data / "task-manager.lock")
        self._close_guard = threading.Lock()
        self._closed = False
        self._health_guard = threading.Lock()
        self._scheduler_state = "ready"
        self._scheduler_error: str | None = None
        self._pending_outcome: dict[str, Any] | None = None
        self._dispatcher = None
        self.task_files = paths.data / "tasks"
        self.task_files.mkdir(parents=True, exist_ok=True)
        try:
            self.store = TaskStore(paths.data / "tasks.db")
            self.store.rebase_managed_paths(paths)
            self.store.recover_after_restart()
        except BaseException:
            self._instance_lock.close()
            raise
        self._submit_lock = threading.Lock()
        try:
            self.supervisor = WorkerSupervisor(paths, cancel_check=self._cancel_requested) if start_worker else None
        except BaseException:
            self._instance_lock.close()
            raise
        self._stop = threading.Event()
        self._wake = threading.Event()
        if self.supervisor is not None:
            self._dispatcher = threading.Thread(target=self._dispatch_loop, name="AuK task dispatcher", daemon=True)
            self._dispatcher.start()

    @property
    def scheduler_health(self) -> dict[str, Any]:
        """In-memory status remains readable when the task database is unavailable."""
        with self._health_guard:
            alive = self._dispatcher is None or self._dispatcher.is_alive()
            return {
                "state": self._scheduler_state,
                "error": self._scheduler_error,
                "accepting_tasks": not self._closed and self._scheduler_state == "ready" and alive,
                "dispatcher_alive": self._dispatcher.is_alive() if self._dispatcher is not None else None,
                "pending_request_id": (
                    self._pending_outcome["request_id"] if self._pending_outcome is not None else None
                ),
            }

    def _set_scheduler_state(self, state: str, error: str | None = None) -> None:
        with self._health_guard:
            if not self._closed and (self._scheduler_state != "paused" or state == "paused"):
                self._scheduler_state = state
                self._scheduler_error = error

    def _storage_call(self, operation: str, callback):
        for attempt in range(self.STORAGE_ATTEMPTS):
            try:
                result = callback()
            except (sqlite3.Error, OSError) as exc:
                detail = f"{operation}: {type(exc).__name__}: {exc}"
                if attempt + 1 == self.STORAGE_ATTEMPTS:
                    self._set_scheduler_state("paused", detail)
                    raise _StorageUnavailable(detail) from exc
                self._set_scheduler_state("recovering", detail)
                time.sleep(self.STORAGE_RETRY_DELAY * (2**attempt))
            else:
                self._set_scheduler_state("ready")
                return result
        raise AssertionError("unreachable")

    def _ensure_accepting(self) -> None:
        if not self.scheduler_health["accepting_tasks"]:
            raise RuntimeError("AuK 任务调度暂不可用，请查看服务诊断；恢复存储后重新启动服务")

    def _cancel_requested(self, request_id: str) -> bool:
        health = self.scheduler_health
        if health["state"] == "paused":
            raise _StorageUnavailable(health["error"] or "任务调度已暂停")
        return self._closed or self._storage_call(
            "检查任务取消状态", lambda: self.store.get(request_id).state in {"cancelling", "cancelled"}
        )

    @staticmethod
    def normalize_request(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or uuid.uuid4()).strip()
        try:
            request_id = str(uuid.UUID(request_id))
        except ValueError as exc:
            raise ValueError("request_id 必须是 UUID") from exc
        task_key = str(payload.get("task_key") or "").strip()
        if task_key not in TASK_BY_KEY:
            raise ValueError(f"未知任务：{task_key}")
        model = str(payload.get("model") or "base").strip().lower()
        if model not in {"flash", "base"}:
            raise ValueError("model 必须是 flash 或 base")
        primary = str(payload.get("primary") or "").strip()
        secondary = str(payload.get("secondary") or "").strip()
        instruction = str(payload.get("instruction") or "").strip() or build_instruction(task_key, primary, secondary)
        duration_mode = str(payload.get("duration_mode") or "manual").strip().lower()
        if duration_mode not in {
            "auto", "manual", "source", "source_auto", "speed", "emotion", "content", "nonverbal",
        }:
            raise ValueError(
                "duration_mode 必须是 auto、manual、source、source_auto、speed、emotion、content 或 nonverbal"
            )
        generation_seconds = _finite_float(payload.get("generation_seconds", 0), "generation_seconds")
        automatic_edit_modes = {"source", "source_auto", "speed", "emotion", "content", "nonverbal"}
        if generation_seconds <= 0 or (generation_seconds > 30 and duration_mode not in automatic_edit_modes):
            raise ValueError("generation_seconds 必须在 0 到 30 秒之间；自动编辑模式会忽略旧的时长控件值")
        seed = _integer(payload.get("seed", 42), "seed")
        if seed < 0 or seed > 0x7FFFFFFFFFFFFFFF:
            raise ValueError("seed 超出范围")
        nfe = _integer(payload.get("nfe_steps", 4 if model == "flash" else 32), "nfe_steps")
        cfg = _finite_float(payload.get("cfg_strength", 0.0 if model == "flash" else 2.0), "cfg_strength")
        sway = _finite_float(payload.get("sway_sampling_coef", -1.0), "sway_sampling_coef")
        if nfe < 4 or nfe > 64:
            raise ValueError("nfe_steps 必须在 4 到 64 之间")
        if cfg < 0.0 or cfg > 5.0:
            raise ValueError("cfg_strength 必须在 0 到 5 之间")
        if sway < -1.0 or sway > 1.0:
            raise ValueError("sway_sampling_coef 必须在 -1 到 1 之间")
        if model == "flash" and (nfe != 4 or cfg != 0.0 or sway != -1.0):
            raise ValueError("AuK-Flash 固定使用 NFE=4、CFG=0、sway=-1 占位")
        cpu_offload = _boolean(payload.get("cpu_offload", True), "cpu_offload")
        keep_loaded = _boolean(payload.get("keep_loaded", True), "keep_loaded")
        seed_mode = str(payload.get("seed_mode") or "fixed").strip().lower()
        if seed_mode not in {"random", "fixed"}:
            raise ValueError("seed_mode 必须是 random 或 fixed")
        return {
            "request_id": request_id,
            "task_key": task_key,
            "model": model,
            "primary": primary,
            "secondary": secondary,
            "instruction": instruction,
            "generation_seconds": generation_seconds,
            "seed": seed,
            "nfe_steps": nfe,
            "cfg_strength": cfg,
            "sway_sampling_coef": sway,
            "cpu_offload": cpu_offload,
            "keep_loaded": keep_loaded,
            "duration_mode": duration_mode,
            "seed_mode": seed_mode,
            "client": str(payload.get("client") or "api")[:32],
        }

    def submit(self, payload: dict[str, Any]) -> tuple[TaskRecord, bool]:
        with self._submit_lock:
            self._ensure_accepting()
            request = self.normalize_request(payload)
            source_payload = payload.get("audio")
            task_template = TASK_BY_KEY[request["task_key"]]
            if task_template.needs_audio and not source_payload:
                raise ValueError(f"{task_template.label}需要源音频或参考音频")
            task_dir = self.task_files / request["request_id"]
            input_path: Path | None = None
            raw_audio: bytes | None = None
            if source_payload:
                decoded = decode_float_audio(source_payload)
                if request["duration_mode"] not in {
                    "source", "source_auto", "speed", "emotion", "content", "nonverbal",
                }:
                    validate_duration(decoded.duration_seconds, request["generation_seconds"])
                raw_audio = decoded.samples.tobytes()
                input_path = task_dir / "input.f32"
                request["source_sample_rate"] = decoded.sample_rate
                request["source_frames"] = len(decoded.samples)
                request["source_sha256"] = hashlib.sha256(raw_audio).hexdigest()
            else:
                validate_duration(0.0, request["generation_seconds"])
            try:
                existing = self.store.get(request["request_id"])
            except KeyError:
                existing = None
            if existing is not None:
                if existing.request != request:
                    raise ValueError("request_id 已被不同参数或音频使用")
                return existing, False
            if input_path is not None and raw_audio is not None:
                task_dir.mkdir(parents=True, exist_ok=True)
                temporary = input_path.with_suffix(".tmp")
                try:
                    temporary.write_bytes(raw_audio)
                    os.replace(temporary, input_path)
                finally:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("无法清理输入音频暂存文件", exc_info=True)
            record, created = self._storage_call(
                "接收任务",
                lambda: self.store.submit(request["request_id"], request, str(input_path) if input_path else None),
            )
        if created:
            self._wake.set()
        return record, created

    def get(self, request_id: str) -> TaskRecord:
        return self.store.get(request_id)

    def unload_models(self) -> bool:
        if self.supervisor is None:
            return False
        return self.supervisor.unload()

    def cancel(self, request_id: str) -> str:
        state = self._storage_call("取消任务", lambda: self.store.cancel(request_id))
        if state == "cancelling" and self.supervisor is not None:
            self.supervisor.cancel(request_id)
        return state

    def retry(self, request_id: str) -> tuple[TaskRecord, bool]:
        previous = self.get(request_id)
        if previous.state not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("只有失败、取消或被服务重启中断的任务可以重试")
        payload = dict(previous.request)
        payload["request_id"] = str(uuid.uuid4())
        if previous.input_path:
            input_path = Path(previous.input_path)
            if not input_path.is_file():
                raise FileNotFoundError("原任务输入音频已被清理，无法重试")
            raw = read_verified_source(input_path, previous.request)
            payload["audio"] = {
                "encoding": "f32le",
                "sample_rate": int(previous.request["source_sample_rate"]),
                "channels": 1,
                "frames": len(raw) // 4,
                "data": base64.b64encode(raw).decode("ascii"),
            }
        return self.submit(payload)

    def _claim_next(self):
        # A claim can commit before its subsequent read fails. Quarantine any
        # orphaned active row before retrying; there is no executing task here.
        needs_recovery = False

        def claim():
            nonlocal needs_recovery
            if needs_recovery:
                self.store.recover_after_restart()
                needs_recovery = False
            try:
                return self.store.claim_next()
            except (sqlite3.Error, OSError):
                needs_recovery = True
                raise

        return self._storage_call("领取任务", claim)

    def _persist_outcome(self, outcome: dict[str, Any]) -> str:
        request_id = outcome["request_id"]

        def commit():
            # Re-read on every attempt: a previous commit may have succeeded
            # even if its response raised. Never delete an already-successful result.
            for _ in range(3):
                state = self.store.get(request_id).state
                if state in {"succeeded", "failed", "cancelled", "interrupted"}:
                    return state
                if state == "cancelling":
                    if self.store.finish_cancel(request_id):
                        return "cancelled"
                elif outcome["kind"] == "succeeded":
                    message = outcome["message"]
                    if self.store.complete(request_id, message["result_path"], message["metadata_path"]):
                        return "succeeded"
                elif outcome["kind"] == "cancelled" and self._closed:
                    self.store.recover_after_restart()
                elif self.store.fail(request_id, outcome.get("error") or "推理任务被中断"):
                    return "failed"
            raise RuntimeError("任务终态持续变化，无法确认提交结果")

        return self._storage_call("保存任务终态", commit)

    def _dispatch_loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self.scheduler_health["state"] == "paused":
                    break
                record = self._claim_next()
                if record is None:
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue
                outcome: dict[str, Any] = {"request_id": record.request_id}
                with self._health_guard:
                    self._pending_outcome = {"request_id": record.request_id, "kind": "running"}
                try:
                    message = self.supervisor.run(
                        record.request,
                        record.input_path,
                        self.paths.outputs / record.request_id,
                        lambda phase, request_id=record.request_id: self._storage_call(
                            "保存任务阶段", lambda: self.store.set_phase(request_id, phase)
                        ),
                    )
                    outcome.update(kind="succeeded", message=message)
                except TaskCancelled:
                    outcome.update(kind="cancelled")
                except _StorageUnavailable:
                    raise
                except Exception as exc:  # noqa: BLE001 - isolate ordinary task failures
                    outcome.update(kind="failed", error=str(exc))
                with self._health_guard:
                    self._pending_outcome = outcome
                final_state = self._persist_outcome(outcome)
                with self._health_guard:
                    self._pending_outcome = None
                if final_state == "cancelled":
                    # A terminated worker may never report filenames. These two
                    # task-owned paths are fixed, and a successful terminal state
                    # never reaches this cleanup branch.
                    output_dir = self.paths.outputs / record.request_id
                    for filename in ("result.wav", "metadata.json"):
                        try:
                            (output_dir / filename).unlink(missing_ok=True)
                        except OSError:
                            logger.warning("无法清理已取消任务的未发布文件", exc_info=True)
        except Exception as exc:  # noqa: BLE001 - expose dispatcher failure to health/admission
            self._set_scheduler_state("paused", f"调度已暂停: {type(exc).__name__}: {exc}")
            logger.error("AuK task dispatcher paused", exc_info=True)

    def wait(self, request_id: str, timeout: float = 600.0, poll: float = 0.25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(request_id)
            if record.state in {"succeeded", "failed", "cancelled", "interrupted"}:
                return record
            scheduler = self.scheduler_health
            if scheduler["state"] in {"paused", "stopping", "stopped"} or scheduler["dispatcher_alive"] is False:
                raise RuntimeError("AuK 任务调度已暂停，请恢复存储并重启服务，再重试任务")
            time.sleep(poll)
        raise TimeoutError(f"任务等待超时：{request_id}")

    def close(self) -> None:
        with self._close_guard:
            if self._closed:
                return
            self._closed = True
        with self._health_guard:
            self._scheduler_state = "stopping"
        self._stop.set()
        self._wake.set()
        try:
            if self.supervisor is not None:
                self.supervisor.close()
            if self._dispatcher is not None:
                self._dispatcher.join(timeout=15)
        finally:
            self._instance_lock.close()
            with self._health_guard:
                self._scheduler_state = "stopped"
