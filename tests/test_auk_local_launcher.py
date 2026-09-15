from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def test_distributable_launcher_and_batch_files_keep_status_visible():
    package = Path(__file__).resolve().parents[1]
    launcher = package / "AuK-Local.exe"
    assert launcher.is_file()
    result = subprocess.run(
        [str(launcher), "--verify-package"],
        cwd=package,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "AuK Local launcher: PASS" in result.stdout
    for name in ("Start-AuK.cmd", "Start-AuK-Service.cmd"):
        raw = (package / "scripts" / name).read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        assert b"\r\n" in raw
        batch = raw.decode("utf-8-sig")
        assert ":keep_open" in batch
        assert "pause >nul" in batch
        assert "exit /b %AUK_EXIT_CODE%" in batch


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
def test_browser_helper_supports_brackets_and_unicode_in_package_path(tmp_path):
    special = tmp_path / "整合包 [test]"
    special.mkdir()
    test_browser_helper_opens_only_this_pack(special, "correct")


@pytest.mark.skipif(os.name != "nt", reason="Windows launcher")
@pytest.mark.parametrize("state", ["correct", "degraded", "foreign", "no_ui", "wrong_protocol"])
def test_browser_helper_opens_only_this_pack(tmp_path, state):
    package = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    (tmp_path / "data").mkdir()
    token = "test-launcher-token-not-a-real-credential"
    (tmp_path / "data" / "session-token").write_text(token + "\n", encoding="utf-8")
    health = {
        "status": "degraded" if state == "degraded" else "ok", "ui_enabled": state != "no_ui",
        "protocol_version": "9.0" if state == "wrong_protocol" else "1.0",
        "instance_id": "foreign" if state == "foreign" else hashlib.sha256(token.encode()).hexdigest(),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(health).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            script = (package / "scripts" / "Open-AuK-WhenReady.ps1").read_text(encoding="utf-8")
            script = script.replace(
                "Start-Process 'http://127.0.0.1:7860'", "Write-Output 'WOULD_OPEN_AUK'",
            ).replace("127.0.0.1:7860", f"127.0.0.1:{server.server_port}")
            helper = tmp_path / "scripts" / "Open-AuK-WhenReady.ps1"
            helper.write_text(script, encoding="utf-8")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper),
                 "-TimeoutSeconds", "2"],
                capture_output=True, text=True, errors="replace", timeout=15, check=True,
            )
            assert ("WOULD_OPEN_AUK" in result.stdout) == (state in {"correct", "degraded"}), result.stderr
            health_script = (package / "scripts" / "Test-AuK-Running.ps1").read_text(encoding="utf-8")
            health_script = health_script.replace("127.0.0.1:7860", f"127.0.0.1:{server.server_port}")
            health_probe = tmp_path / "scripts" / "Test-AuK-Running.ps1"
            health_probe.write_text(health_script, encoding="utf-8")
            for name in ("Start-AuK.cmd", "Start-AuK-Service.cmd"):
                arguments = [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(health_probe),
                ]
                if name == "Start-AuK.cmd":
                    arguments.append("-RequireUi")
                result = subprocess.run(
                    arguments,
                    cwd=tmp_path, capture_output=True, text=True, errors="replace", timeout=15, check=False,
                )
                accepted = state in {"correct", "degraded"} or (state == "no_ui" and name == "Start-AuK-Service.cmd")
                assert (result.returncode == 0) == accepted, (name, state, result.stderr)
        finally:
            server.shutdown()
            thread.join(timeout=3)
