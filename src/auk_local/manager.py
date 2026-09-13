from __future__ import annotations

import base64
import hashlib
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .audio import decode_float_audio, validate_duration
from .config import LocalPaths
from .store import TaskRecord, TaskStore
from .task_templates import TASK_BY_KEY, build_instruction
from .worker import TaskCancelled, WorkerSupervisor


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
    def __init__(self, paths: LocalPaths, *, start_worker: bool = True):
        self.paths = paths
        paths.ensure_writable_dirs()
        self._instance_lock = _SingleInstanceLock(paths.data / "task-manager.lock")
        self._close_guard = threading.Lock()
        self._closed = False
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
        self.supervisor = WorkerSupervisor(paths) if start_worker else None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._dispatcher = None
        if self.supervisor is not None:
            self._dispatcher = threading.Thread(target=self._dispatch_loop, name="AuK task dispatcher", daemon=True)
            self._dispatcher.start()

    @staticmethod
    def normalize_request(payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload.get("request_id") or uuid.uuid4()).strip()
        try:
            uuid.UUID(request_id)
        except ValueError as exc:
            raise ValueError("request_id 必须是 UUID") from exc
        task_key = str(payload.get("task_key") or "").strip()
        if task_key not in TASK_BY_KEY:
            raise ValueError(f"未知任务：{task_key}")
        model = str(payload.get("model") or "flash").strip().lower()
        if model not in {"flash", "base"}:
            raise ValueError("model 必须是 flash 或 base")
        primary = str(payload.get("primary") or "").strip()
        secondary = str(payload.get("secondary") or "").strip()
        instruction = str(payload.get("instruction") or "").strip() or build_instruction(task_key, primary, secondary)
        generation_seconds = _finite_float(payload.get("generation_seconds", 0), "generation_seconds")
        if generation_seconds <= 0 or generation_seconds > 30:
            raise ValueError("generation_seconds 必须在 0 到 30 秒之间")
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
        keep_loaded = _boolean(payload.get("keep_loaded", False), "keep_loaded")
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
            "client": str(payload.get("client") or "api")[:32],
        }

    def submit(self, payload: dict[str, Any]) -> tuple[TaskRecord, bool]:
        with self._submit_lock:
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
                temporary.write_bytes(raw_audio)
                os.replace(temporary, input_path)
            record, created = self.store.submit(
                request["request_id"], request, str(input_path) if input_path else None
            )
        if created:
            self._wake.set()
        return record, created

    def get(self, request_id: str) -> TaskRecord:
        return self.store.get(request_id)

    def cancel(self, request_id: str) -> str:
        state = self.store.cancel(request_id)
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
            raw = input_path.read_bytes()
            payload["audio"] = {
                "encoding": "f32le",
                "sample_rate": int(previous.request["source_sample_rate"]),
                "channels": 1,
                "frames": len(raw) // 4,
                "data": base64.b64encode(raw).decode("ascii"),
            }
        return self.submit(payload)

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            record = self.store.claim_next()
            if record is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            output_dir = self.paths.outputs / record.request_id
            try:
                message = self.supervisor.run(
                    record.request,
                    record.input_path,
                    output_dir,
                    lambda phase, request_id=record.request_id: self.store.set_phase(request_id, phase),
                )
                if not self.store.complete(record.request_id, message["result_path"], message["metadata_path"]):
                    for path_key in ("result_path", "metadata_path"):
                        try:
                            Path(message[path_key]).unlink(missing_ok=True)
                        except OSError:
                            pass
                    self.store.finish_cancel(record.request_id)
            except TaskCancelled:
                self.store.finish_cancel(record.request_id)
            except Exception as exc:  # noqa: BLE001 - task failures must be persisted instead of killing dispatch
                if not self.store.fail(record.request_id, str(exc)):
                    self.store.finish_cancel(record.request_id)

    def wait(self, request_id: str, timeout: float = 600.0, poll: float = 0.25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(request_id)
            if record.state in {"succeeded", "failed", "cancelled", "interrupted"}:
                return record
            time.sleep(poll)
        raise TimeoutError(f"任务等待超时：{request_id}")

    def close(self) -> None:
        with self._close_guard:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._wake.set()
        try:
            if self.supervisor is not None:
                self.supervisor.close()
            if self._dispatcher is not None:
                self._dispatcher.join(timeout=15)
        finally:
            self._instance_lock.close()
