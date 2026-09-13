from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .version import PROTOCOL_VERSION


class LocalClient:
    def __init__(self, base_url: str, token_file: str | Path):
        self.base_url = base_url.rstrip("/")
        self.token_file = Path(token_file)

    def _token(self) -> str:
        token = self.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError(f"本机服务令牌为空：{self.token_file}")
        return token

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "X-AuK-Token": self._token()},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AuK 本机服务返回 {exc.code}：{body}") from exc

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + "/api/v1/health", timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
        server_protocol = str(result.get("protocol_version", ""))
        if server_protocol.split(".")[0] != PROTOCOL_VERSION.split(".")[0]:
            raise RuntimeError(f"协议版本不兼容：节点 {PROTOCOL_VERSION}，服务 {server_protocol}")
        return result

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.health()
        return self.request("POST", "/api/v1/tasks", payload, timeout=30)

    def status(self, request_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/v1/tasks/{request_id}", timeout=10)

    def cancel(self, request_id: str) -> dict[str, Any]:
        return self.request("POST", f"/api/v1/tasks/{request_id}/cancel", {}, timeout=10)

    def wait(self, request_id: str, timeout: float = 900, poll: float = 0.5) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.status(request_id)
            if result["state"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                return result
            time.sleep(poll)
        raise TimeoutError(f"任务等待超时：{request_id}")

    def download_audio(self, request_id: str, timeout: float = 60) -> bytes:
        request = urllib.request.Request(
            self.base_url + f"/api/v1/tasks/{request_id}/audio",
            headers={"X-AuK-Token": self._token()},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def metadata(self, request_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/v1/tasks/{request_id}/metadata", timeout=10)
