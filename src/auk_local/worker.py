from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from .config import LocalPaths


class TaskCancelled(RuntimeError):
    pass


def worker_main(input_queue, output_queue, root: str) -> None:
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
                lambda phase: output_queue.put({"type": "phase", "request_id": request_id, "phase": phase}),
            )
            output_queue.put({"type": "done", "request_id": request_id, **result})
        except BaseException as exc:
            output_queue.put(
                {"type": "error", "request_id": request_id, "error": f"{type(exc).__name__}: {exc}"}
            )


class WorkerSupervisor:
    def __init__(self, paths: LocalPaths):
        self.paths = paths
        self.context = mp.get_context("spawn")
        self._guard = threading.RLock()
        self._run_lock = threading.Lock()
        self._current_id: str | None = None
        self._cancel_event = threading.Event()
        self._process = None
        self._input = None
        self._output = None
        self._start_worker()

    def _start_worker(self) -> None:
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
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10)
        self._start_worker()

    def run(self, task: dict[str, Any], input_path: str | None, output_dir: Path, on_phase: Callable[[str], None]):
        with self._run_lock:
            request_id = task["request_id"]
            with self._guard:
                self._current_id = request_id
                self._cancel_event.clear()
                if not self._process.is_alive():
                    self._restart_worker()
                self._input.put(
                    {"type": "run", "task": task, "input_path": input_path, "output_dir": str(output_dir)}
                )
            try:
                while True:
                    if self._cancel_event.is_set():
                        with self._guard:
                            self._restart_worker()
                        raise TaskCancelled(request_id)
                    if not self._process.is_alive():
                        with self._guard:
                            self._restart_worker()
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
            finally:
                with self._guard:
                    self._current_id = None
                    self._cancel_event.clear()

    def cancel(self, request_id: str) -> bool:
        with self._guard:
            if self._current_id != request_id:
                return False
            self._cancel_event.set()
            return True

    def close(self) -> None:
        with self._guard:
            if self._process is None:
                return
            if self._process.is_alive():
                self._input.put({"type": "shutdown"})
                self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10)
