from __future__ import annotations

import queue
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from auk_local.config import LocalPaths
from auk_local.manager import TaskManager
from auk_local.worker import TaskCancelled, WorkerSupervisor


class _SuccessfulSupervisor:
    def __init__(self):
        self.runs = 0

    def run(self, task, _input, output, on_phase):
        self.runs += 1
        on_phase("saving")
        output.mkdir(parents=True)
        result = output / "result.wav"
        metadata = output / "metadata.json"
        result.write_bytes(b"synthetic-result")
        metadata.write_text("{}", encoding="utf-8")
        return {"result_path": str(result), "metadata_path": str(metadata)}

    def close(self):
        return None


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="auk-lifecycle-")
        self.manager = TaskManager(LocalPaths.from_root(Path(self.temporary.name)), start_worker=False)
        self.manager.STORAGE_RETRY_DELAY = 0.001
        self.supervisor = _SuccessfulSupervisor()
        self.manager.supervisor = self.supervisor

    def tearDown(self):
        self.manager.close()
        self.temporary.cleanup()

    def submit(self):
        return self.manager.submit(
            {"task_key": "instruct_tts", "primary": "lifecycle test", "generation_seconds": 1}
        )[0]

    def start(self):
        self.manager._dispatcher = threading.Thread(target=self.manager._dispatch_loop, daemon=True)
        self.manager._dispatcher.start()

    def await_paused(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.manager.scheduler_health["state"] == "paused":
                return
            time.sleep(0.005)
        self.fail(str(self.manager.scheduler_health))

    def test_transient_claim_error_recovers_and_executes_once(self):
        record = self.submit()
        original = self.manager.store.claim_next
        calls = 0

        def claim():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise sqlite3.OperationalError("database is locked")
            return original()

        with patch.object(self.manager.store, "claim_next", side_effect=claim):
            self.start()
            self.assertEqual(self.manager.wait(record.request_id, timeout=3).state, "succeeded")
        self.assertEqual(self.supervisor.runs, 1)
        self.assertTrue(self.manager.scheduler_health["accepting_tasks"])

    def test_persistent_claim_error_pauses_and_rejects_new_tasks(self):
        record = self.submit()
        with patch.object(self.manager.store, "claim_next", side_effect=sqlite3.OperationalError("disk full")) as claim:
            with self.assertLogs("auk_local.manager", level="ERROR"):
                self.start()
                self.await_paused()
            self.assertEqual(claim.call_count, self.manager.STORAGE_ATTEMPTS)
        self.assertEqual(self.manager.get(record.request_id).state, "queued")
        with self.assertRaises(RuntimeError):
            self.submit()
        # A successful cancellation must not silently unpause a dead dispatcher.
        self.manager.cancel(record.request_id)
        self.assertEqual(self.manager.scheduler_health["state"], "paused")

    def test_committed_claim_read_failure_quarantines_orphan(self):
        first = self.submit()
        second = self.submit()
        original = self.manager.store.claim_next
        calls = 0

        def claim():
            nonlocal calls
            calls += 1
            result = original()
            if calls == 1:
                raise sqlite3.OperationalError("read failed after claim committed")
            return result

        with patch.object(self.manager.store, "claim_next", side_effect=claim):
            self.start()
            self.assertEqual(self.manager.wait(second.request_id, timeout=3).state, "succeeded")
        self.assertEqual(self.manager.get(first.request_id).state, "interrupted")
        self.assertEqual(self.supervisor.runs, 1)

    def test_completion_retry_does_not_rerun_inference(self):
        record = self.submit()
        original = self.manager.store.complete
        calls = 0

        def complete(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise sqlite3.OperationalError("temporary write failure")
            return original(*args)

        with patch.object(self.manager.store, "complete", side_effect=complete):
            self.start()
            self.assertEqual(self.manager.wait(record.request_id, timeout=3).state, "succeeded")
        self.assertEqual(self.supervisor.runs, 1)

    def test_cancelled_worker_without_result_message_cleans_partial_files(self):
        record = self.submit()

        def run(task, _input, output, _on_phase):
            output.mkdir(parents=True)
            (output / "result.wav").write_bytes(b"partial-wave")
            (output / "metadata.json").write_bytes(b"{partial-json")
            self.manager.store.cancel(task["request_id"])
            raise TaskCancelled(task["request_id"])

        self.supervisor.run = run
        self.start()
        self.assertEqual(self.manager.wait(record.request_id, timeout=3).state, "cancelled")
        self.manager.close()
        output = self.manager.paths.outputs / record.request_id
        self.assertFalse((output / "result.wav").exists())
        self.assertFalse((output / "metadata.json").exists())

    def test_ambiguous_success_commit_preserves_result(self):
        record = self.submit()
        original = self.manager.store.complete

        def complete(*args):
            original(*args)
            raise sqlite3.OperationalError("response lost after commit")

        with patch.object(self.manager.store, "complete", side_effect=complete):
            self.start()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                row = self.manager.get(record.request_id)
                if row.state == "succeeded" and self.manager.scheduler_health["pending_request_id"] is None:
                    break
                time.sleep(0.005)
        self.assertEqual(row.state, "succeeded")
        self.assertTrue(Path(row.result_path).is_file())
        self.assertEqual(self.supervisor.runs, 1)

    def test_persistent_completion_error_retains_pending_result(self):
        record = self.submit()
        with patch.object(self.manager.store, "complete", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertLogs("auk_local.manager", level="ERROR"):
                self.start()
                self.await_paused()
        health = self.manager.scheduler_health
        self.assertEqual(health["pending_request_id"], record.request_id)
        self.assertFalse(health["accepting_tasks"])
        self.assertTrue((self.manager.paths.outputs / record.request_id / "result.wav").is_file())
        self.assertEqual(self.supervisor.runs, 1)

    def test_failure_state_write_error_is_visible_not_silent_thread_death(self):
        record = self.submit()
        self.supervisor.run = Mock(side_effect=RuntimeError("inference failed"))
        with patch.object(self.manager.store, "fail", side_effect=sqlite3.OperationalError("database unavailable")):
            with self.assertLogs("auk_local.manager", level="ERROR"):
                self.start()
                self.await_paused()
        self.assertEqual(self.manager.scheduler_health["pending_request_id"], record.request_id)
        self.assertIn("inference failed", self.manager._pending_outcome["error"])

    def test_failed_concurrent_cancel_does_not_unpause_or_run_next_task(self):
        first = self.submit()
        second = self.submit()
        entered = threading.Event()
        release = threading.Event()
        original = self.supervisor.run

        def run(*args):
            entered.set()
            release.wait(2)
            return original(*args)

        self.supervisor.run = run
        self.start()
        self.assertTrue(entered.wait(1))
        try:
            with patch.object(self.manager.store, "cancel", side_effect=sqlite3.OperationalError("disk full")):
                with self.assertRaises(RuntimeError):
                    self.manager.cancel(first.request_id)
        finally:
            release.set()
        self.manager._dispatcher.join(2)
        self.assertFalse(self.manager._dispatcher.is_alive())
        self.assertEqual(self.manager.get(first.request_id).state, "succeeded")
        self.assertEqual(self.manager.get(second.request_id).state, "queued")
        self.assertEqual(self.manager.scheduler_health["state"], "paused")
        self.assertEqual(self.supervisor.runs, 1)

    def make_supervisor(self):
        supervisor = WorkerSupervisor.__new__(WorkerSupervisor)
        supervisor._guard = threading.RLock()
        supervisor._run_lock = threading.Lock()
        supervisor._current_id = None
        supervisor._cancel_event = threading.Event()
        supervisor._closed = False
        supervisor._cancel_check = self.manager._cancel_requested
        supervisor._process = SimpleNamespace(is_alive=lambda: True)
        supervisor._input = Mock()
        supervisor._output = queue.Queue()
        supervisor._restart_worker = Mock()
        supervisor.close = Mock()
        return supervisor

    def test_cancel_between_claim_and_activation_never_sends_run(self):
        record = self.submit()
        self.manager.store.claim_next()
        supervisor = self.make_supervisor()
        self.manager.supervisor = supervisor
        self.assertEqual(self.manager.cancel(record.request_id), "cancelling")
        with self.assertRaises(TaskCancelled):
            supervisor.run(record.request, None, self.manager.paths.outputs, lambda _phase: None)
        supervisor._input.put.assert_not_called()
        self.assertIsNone(supervisor._current_id)

    def test_phase_persistence_error_stops_worker_and_pauses(self):
        record = self.submit()
        supervisor = self.make_supervisor()
        supervisor._output.put({"type": "phase", "request_id": record.request_id, "phase": "sampling"})
        self.manager.supervisor = supervisor
        with patch.object(self.manager.store, "set_phase", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertLogs("auk_local.manager", level="ERROR"):
                self.start()
                self.await_paused()
        supervisor._restart_worker.assert_called_once()
        self.assertEqual(self.manager.scheduler_health["pending_request_id"], record.request_id)

    def test_manual_unload_sends_worker_command(self):
        supervisor = self.make_supervisor()
        supervisor._output.put({"type": "unloaded"})
        self.assertTrue(supervisor.unload(timeout=0.1))
        supervisor._input.put.assert_called_once_with({"type": "unload"})

    def test_manual_unload_rejects_busy_worker(self):
        supervisor = self.make_supervisor()
        supervisor._run_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "任务正在运行"):
                supervisor.unload(timeout=0.1)
        finally:
            supervisor._run_lock.release()

    @unittest.skipUnless(os.name == "nt", "Windows process handle verification")
    def test_watchdog_exits_real_child_after_parent_abrupt_exit(self):
        import ctypes
        from ctypes import wintypes

        root = Path(self.temporary.name)
        script = root / "parent_probe.py"
        ready = root / "child.pid"
        script.write_text(
            "import multiprocessing as mp, os, sys, time\n"
            "from pathlib import Path\n"
            "from auk_local.worker import _start_parent_watchdog\n"
            "def child(ready):\n"
            "    _start_parent_watchdog()\n"
            "    Path(ready).write_text(str(os.getpid()))\n"
            "    while True: time.sleep(0.05)\n"
            "if __name__ == '__main__':\n"
            "    ready = sys.argv[1]\n"
            "    process = mp.get_context('spawn').Process(target=child, args=(ready,))\n"
            "    process.start()\n"
            "    deadline = time.monotonic() + 8\n"
            "    while not Path(ready).exists() and time.monotonic() < deadline: time.sleep(0.01)\n"
            "    os._exit(0 if Path(ready).exists() else 2)\n",
            encoding="utf-8",
        )
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        process = subprocess.Popen(
            [sys.executable, "-B", str(script), str(ready)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        handle = None
        try:
            self.assertEqual(process.wait(timeout=12), 0)
            self.assertTrue(ready.is_file())
            child_pid = int(ready.read_text())
            handle = kernel.OpenProcess(0x00100000 | 0x0001, False, child_pid)
            if not handle:
                self.assertEqual(ctypes.get_last_error(), 87)  # already exited
            else:
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if handle:
                if kernel.WaitForSingleObject(handle, 0) != 0:
                    kernel.TerminateProcess(handle, 1)
                    kernel.WaitForSingleObject(handle, 5000)
                kernel.CloseHandle(handle)


if __name__ == "__main__":
    unittest.main()
