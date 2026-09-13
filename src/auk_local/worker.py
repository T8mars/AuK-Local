from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import LocalPaths


class TaskCancelled(RuntimeError):
    pass


def _start_parent_watchdog() -> None:
    """Windows does not terminate multiprocessing children when a parent is killed."""
    parent = mp.parent_process()
    if parent is None:
        return

    def watch():
        parent.join()
        # The service is gone; do not wait for an inference/kernel or a queue
        # feeder to finish. Releasing the process also releases its CUDA context.
        os._exit(0)

    threading.Thread(target=watch, name="AuK parent watchdog", daemon=True).start()


def worker_main(input_queue, output_queue, root: str) -> None:
    _start_parent_watchdog()
    from .inference import InferenceRuntime

    runtime = InferenceRuntime(LocalPaths.from_root(Path(root)))
    while True:
        command = input_queue.get()
        if command["type"] == "shutdown":
            runtime.unload()
            return
        if command["type"] == "unload":
            runtime.unload()
            output_queue.put({"type": "unloaded"})
            continue
        if command["type"] != "run":
            continue
        request_id = command["task"]["request_id"]
        try:
            result = runtime.execute(
                command["task"],
                command.get("input_path"),
                Path(command["output_dir"]),
                lambda phase, request_id=request_id: output_queue.put(
                    {"type": "phase", "request_id": request_id, "phase": phase}
                ),
            )
            output_queue.put({"type": "done", "request_id": request_id, **result})
        except BaseException as exc:  # noqa: BLE001 - child must report interrupts and fatal task errors
            output_queue.put(
                {"type": "error", "request_id": request_id, "error": f"{type(exc).__name__}: {exc}"}
            )


class WorkerSupervisor:
    def __init__(self, paths: LocalPaths, *, cancel_check: Callable[[str], bool] | None = None):
        self.paths = paths
        self.context = mp.get_context("spawn")
        self._guard = threading.RLock()
        self._run_lock = threading.Lock()
        self._current_id: str | None = None
        self._cancel_event = threading.Event()
        self._process = None
        self._input = None
        self._output = None
        self._closed = False
        self._cancel_check = cancel_check
        self._start_worker()

    def _start_worker(self) -> None:
        if self._closed:
            raise RuntimeError("推理 worker 已关闭")
        self._input = self.context.Queue()
        self._output = self.context.Queue()
        self._process = self.context.Process(
            target=worker_main,
            args=(self._input, self._output, str(self.paths.root)),
            name="AuK inference worker",
            daemon=True,
        )
        self._process.start()

    def _restart_worker(self) -> None:
        if self._closed:
            return
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=10)
        self._close_queues()
        self._start_worker()

    def _close_queues(self) -> None:
        for channel in (self._input, self._output):
            if channel is None:
                continue
            try:
                channel.close()
                channel.cancel_join_thread()
            except (OSError, ValueError):
                pass
        self._input = None
        self._output = None

    def run(self, task: dict[str, Any], input_path: str | None, output_dir: Path, on_phase: Callable[[str], None]):
        with self._run_lock:
            request_id = task["request_id"]
            try:
                with self._guard:
                    if self._closed:
                        raise RuntimeError("推理 worker 已关闭")
                    self._current_id = request_id
                    self._cancel_event.clear()
                    # Cancellation can arrive after DB claim but before _current_id
                    # exists. Check the durable state under the activation guard.
                    if self._cancel_check is not None and self._cancel_check(request_id):
                        raise TaskCancelled(request_id)
                    if not self._process.is_alive():
                        self._restart_worker()
                    self._input.put(
                        {"type": "run", "task": task, "input_path": input_path, "output_dir": str(output_dir)}
                    )
                while True:
                    if self._cancel_event.is_set():
                        with self._guard:
                            if not self._closed:
                                self._restart_worker()
                        raise TaskCancelled(request_id)
                    if not self._process.is_alive():
                        raise RuntimeError("推理 worker 意外退出")
                    try:
                        message = self._output.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if message.get("request_id") not in {None, request_id}:
                        continue
                    if message["type"] == "phase":
                        on_phase(message["phase"])
                    elif message["type"] == "done":
                        return message
                    elif message["type"] == "error":
                        raise RuntimeError(message["error"])
            except TaskCancelled:
                raise
            except Exception:
                # A phase callback can fail while inference is still running.
                # Stop that worker before exposing the failure to the dispatcher.
                with self._guard:
                    if not self._closed:
                        self._restart_worker()
                raise
            finally:
                with self._guard:
                    self._current_id = None
                    self._cancel_event.clear()

    def cancel(self, request_id: str) -> bool:
        with self._guard:
            if self._closed or self._current_id != request_id:
                return False
            self._cancel_event.set()
            return True

    def close(self) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
            self._cancel_event.set()
            process = self._process
            if process is not None and process.is_alive():
                self._input.put({"type": "shutdown"})
                process.join(timeout=10)
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=10)
            elif process is not None:
                process.join(timeout=10)
            self._close_queues()
