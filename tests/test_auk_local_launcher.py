from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


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
            for name in ("Start-AuK.cmd", "Start-AuK-Service.cmd"):
                batch = (package / "scripts" / name).read_text(encoding="utf-8")
                line = next(line for line in batch.splitlines() if line.startswith("powershell.exe -NoProfile -Command"))
                command = line.split('-Command "', 1)[1][:-1]
                command = command.replace("127.0.0.1:7860", f"127.0.0.1:{server.server_port}")
                result = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", command],
                    cwd=tmp_path, capture_output=True, text=True, errors="replace", timeout=15, check=False,
                )
                accepted = state in {"correct", "degraded"} or (state == "no_ui" and name == "Start-AuK-Service.cmd")
                assert (result.returncode == 0) == accepted, (name, state, result.stderr)
        finally:
            server.shutdown()
            thread.join(timeout=3)
