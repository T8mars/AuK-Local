from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .version import PROTOCOL_VERSION


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "AuK Local 不接受 HTTP 重定向", headers, fp)


def validate_loopback_url(base_url: str) -> str:
    candidate = str(base_url or "").rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(candidate)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("AuK Local 服务地址无效") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("AuK Local 服务地址只能是本机 loopback HTTP 地址")
    return candidate


class LocalClient:
    def __init__(self, base_url: str, token_file: str | Path):
        self.base_url = validate_loopback_url(base_url)
        self.token_file = Path(token_file)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

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
            with self._opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AuK 本机服务返回 {exc.code}：{body}") from exc

    def health(self) -> dict[str, Any]:
        with self._opener.open(self.base_url + "/api/v1/health", timeout=5) as response:
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
        with self._opener.open(request, timeout=timeout) as response:
            return response.read()

    def metadata(self, request_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/v1/tasks/{request_id}/metadata", timeout=10)
