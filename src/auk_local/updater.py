from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .config import LocalPaths
from .version import PROTOCOL_VERSION, VERSION


RELEASE_API = "https://api.github.com/repos/T8mars/AuK-Local/releases/latest"
REPOSITORY = "T8mars/AuK-Local"
MANIFEST_ASSET = "auk-local-release.json"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_FILE_COUNT = 10_000
MAX_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PROTECTED_ROOTS = {"models", "runtime", "ckpts", "data", "outputs", "logs", ".git", ".secrets"}
ALLOWED_ROOT_DIRS = {"src", "scripts", "docs", "assets"}
ALLOWED_ROOT_FILES = {
    "AuK-Local.exe",
    "BUILD-INFO.json",
    "LICENSE",
    "README.md",
    "README_zh.md",
    "requirements-windows-cu128.lock",
    "pyproject.toml",
    "使用说明.md",
    "启动AuK.cmd",
    "启动AuK服务.cmd",
    "环境诊断.cmd",
}


class UpdateError(RuntimeError):
    pass


def _version_tuple(value: str) -> tuple[int, int, int, int]:
    if not VERSION_RE.fullmatch(value):
        raise UpdateError(f"版本号格式无效：{value!r}")
    parts = [int(item) for item in value.split(".")]
    return tuple((parts + [0, 0, 0, 0])[:4])  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "//" in value:
        raise UpdateError(f"更新包路径无效：{value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UpdateError(f"更新包路径越界：{value!r}")
    top = path.parts[0]
    if top in PROTECTED_ROOTS:
        raise UpdateError(f"更新包禁止修改受保护目录：{top}")
    if len(path.parts) == 1:
        if value not in ALLOWED_ROOT_FILES:
            raise UpdateError(f"更新包包含未授权的根目录文件：{value}")
    elif top not in ALLOWED_ROOT_DIRS:
        raise UpdateError(f"更新包包含未授权目录：{top}")
    if top == "assets" and value != "assets/AuK-Local.ico":
        raise UpdateError(f"更新包禁止覆盖用户或示例素材：{value}")
    return path


class UpdateManager:
    def __init__(
        self,
        paths: LocalPaths,
        *,
        current_version: str = VERSION,
        api_url: str = RELEASE_API,
        signature_verifier: Callable[[bytes, str], bool] | None = None,
    ) -> None:
        self.paths = paths
        self.current_version = current_version
        self.api_url = api_url
        self.signature_verifier = signature_verifier or self._verify_with_launcher
        self._candidate: dict[str, Any] | None = None

    @property
    def update_root(self) -> Path:
        return self.paths.data / "updates"

    def check(self) -> dict[str, Any]:
        import httpx

        headers = {"Accept": "application/vnd.github+json", "User-Agent": f"AuK-Local/{self.current_version}"}
        try:
            with httpx.Client(follow_redirects=True, timeout=20.0, headers=headers) as client:
                release_response = client.get(self.api_url)
                release_response.raise_for_status()
                release = release_response.json()
                assets = release.get("assets") if isinstance(release, dict) else None
                if not isinstance(assets, list):
                    raise UpdateError("GitHub Release 返回内容缺少 assets")
                manifest_asset = next(
                    (item for item in assets if isinstance(item, dict) and item.get("name") == MANIFEST_ASSET), None
                )
                if not manifest_asset:
                    raise UpdateError(f"最新版 Release 缺少 {MANIFEST_ASSET}")
                manifest_size = int(manifest_asset.get("size") or 0)
                if manifest_size <= 0 or manifest_size > MAX_MANIFEST_BYTES:
                    raise UpdateError("更新清单大小异常")
                response = client.get(str(manifest_asset.get("browser_download_url") or ""))
                response.raise_for_status()
                if len(response.content) > MAX_MANIFEST_BYTES:
                    raise UpdateError("更新清单超过大小限制")
                if len(response.content) != manifest_size:
                    raise UpdateError("更新清单下载大小与 GitHub 资产信息不一致")
                envelope = response.json()
        except UpdateError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise UpdateError(f"检查更新失败：{exc}") from exc

        payload = self._verify_envelope(envelope)
        self._validate_payload(payload)
        package = payload["package"]
        package_asset = next(
            (item for item in assets if isinstance(item, dict) and item.get("name") == package["name"]), None
        )
        if not package_asset:
            raise UpdateError("Release 中找不到清单指定的代码更新包")
        if int(package_asset.get("size") or -1) != int(package["size"]):
            raise UpdateError("Release 资产大小与签名清单不一致")
        candidate = {
            "available": _version_tuple(payload["version"]) > _version_tuple(self.current_version),
            "current_version": self.current_version,
            "version": payload["version"],
            "notes": str(payload.get("notes") or ""),
            "release_url": str(release.get("html_url") or ""),
            "asset_url": str(package_asset.get("browser_download_url") or ""),
            "payload": payload,
        }
        self._candidate = candidate if candidate["available"] else None
        return candidate

    def stage_latest(self) -> dict[str, Any]:
        candidate = self.check()
        if not candidate["available"]:
            raise UpdateError(f"当前已是最新版 {self.current_version}")
        payload = candidate["payload"]
        package = payload["package"]
        downloads = self.update_root / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        archive = downloads / str(package["name"])
        self._download(candidate["asset_url"], archive, int(package["size"]))
        if _sha256(archive) != package["sha256"]:
            archive.unlink(missing_ok=True)
            raise UpdateError("更新包 SHA-256 校验失败，已删除损坏下载")
        staging_parent = self.update_root / "staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = staging_parent / f"{payload['version']}-{uuid.uuid4().hex}"
        try:
            self._extract_and_verify(archive, staging, payload["files"])
            pending = {
                "schema_version": 1,
                "from_version": self.current_version,
                "to_version": payload["version"],
                "staging_directory": str(staging.relative_to(self.paths.root)).replace("\\", "/"),
                "files": payload["files"],
            }
            pending_path = self.update_root / "pending-update.json"
            temporary = pending_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, pending_path)
            return {"staged": True, "version": payload["version"], "pending_path": str(pending_path)}
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _verify_envelope(self, envelope: Any) -> dict[str, Any]:
        if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
            raise UpdateError("更新清单封装格式无效")
        encoded = envelope.get("signed_payload")
        signature = envelope.get("signature")
        if not isinstance(encoded, str) or not isinstance(signature, str):
            raise UpdateError("更新清单缺少签名数据")
        try:
            payload_bytes = base64.b64decode(encoded, validate=True)
            base64.b64decode(signature, validate=True)
        except (ValueError, TypeError) as exc:
            raise UpdateError("更新清单签名编码无效") from exc
        if len(payload_bytes) > MAX_MANIFEST_BYTES:
            raise UpdateError("签名清单内容超过大小限制")
        if not self.signature_verifier(payload_bytes, signature):
            raise UpdateError("更新清单数字签名无效，已拒绝更新")
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise UpdateError("签名清单 JSON 无效") from exc
        if not isinstance(payload, dict):
            raise UpdateError("签名清单必须是 JSON 对象")
        return payload

    def _validate_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != 1 or payload.get("repository") != REPOSITORY:
            raise UpdateError("更新清单来源或版本不受信任")
        if payload.get("channel") != "stable":
            raise UpdateError("只允许安装 stable 更新")
        version = str(payload.get("version") or "")
        _version_tuple(version)
        compatibility = payload.get("compatibility")
        if not isinstance(compatibility, dict) or compatibility.get("service_protocol") != PROTOCOL_VERSION:
            raise UpdateError("更新包与当前服务协议不兼容")
        package = payload.get("package")
        if not isinstance(package, dict):
            raise UpdateError("更新清单缺少 package")
        name = str(package.get("name") or "")
        if name != f"AuK-Local-v{version}-code-only.zip":
            raise UpdateError("更新包文件名与版本不一致")
        size = int(package.get("size") or 0)
        if size <= 0 or size > MAX_ARCHIVE_BYTES or not HASH_RE.fullmatch(str(package.get("sha256") or "")):
            raise UpdateError("更新包大小或 SHA-256 无效")
        files = payload.get("files")
        if not isinstance(files, list) or not files or len(files) > MAX_FILE_COUNT:
            raise UpdateError("更新包文件清单为空或数量异常")
        seen: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise UpdateError("更新包文件清单格式无效")
            name = str(item.get("path") or "")
            _safe_relative_path(name)
            if name.casefold() in seen:
                raise UpdateError(f"更新包文件路径重复：{name}")
            seen.add(name.casefold())
            size = int(item.get("size") or 0)
            if size < 0 or size > MAX_UNCOMPRESSED_BYTES or not HASH_RE.fullmatch(str(item.get("sha256") or "")):
                raise UpdateError(f"更新包文件校验信息无效：{name}")
        required = {"AuK-Local.exe", "src/auk_local/version.py", "scripts/Apply-AuK-Update.ps1"}
        if not required.issubset({str(item["path"]) for item in files}):
            raise UpdateError("更新包缺少启动器、版本文件或更新执行脚本")

    def _verify_with_launcher(self, payload: bytes, signature: str) -> bool:
        launcher = self.paths.root / "AuK-Local.exe"
        if not launcher.is_file():
            raise UpdateError("缺少 AuK-Local.exe，无法验证更新签名")
        self.update_root.mkdir(parents=True, exist_ok=True)
        payload_path = self.update_root / f"verify-{uuid.uuid4().hex}.json"
        try:
            payload_path.write_bytes(payload)
            result = subprocess.run(
                [str(launcher), "--verify-update-signature", str(payload_path), signature],
                cwd=self.paths.root,
                capture_output=True,
                timeout=15,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError) as exc:
            raise UpdateError(f"无法调用启动器验证更新签名：{exc}") from exc
        finally:
            payload_path.unlink(missing_ok=True)

    def _download(self, url: str, destination: Path, expected_size: int) -> None:
        import httpx

        if not url.startswith("https://github.com/"):
            raise UpdateError("更新包下载地址不是受信任的 GitHub 地址")
        partial = destination.with_suffix(destination.suffix + ".part")
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > expected_size:
            partial.unlink()
            offset = 0
        headers = {"User-Agent": f"AuK-Local/{self.current_version}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with (
                httpx.Client(follow_redirects=True, timeout=60.0) as client,
                client.stream("GET", url, headers=headers) as response,
            ):
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset and not append:
                    offset = 0
                with partial.open("ab" if append else "wb") as handle:
                    total = offset
                    for block in response.iter_bytes(1024 * 1024):
                        total += len(block)
                        if total > expected_size or total > MAX_ARCHIVE_BYTES:
                            raise UpdateError("更新包下载大小超过签名清单")
                        handle.write(block)
            if partial.stat().st_size != expected_size:
                raise UpdateError("更新包下载不完整，可再次点击继续下载")
            os.replace(partial, destination)
        except UpdateError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise UpdateError(f"下载更新包失败：{exc}") from exc

    def _extract_and_verify(self, archive: Path, staging: Path, manifest_files: list[dict[str, Any]]) -> None:
        expected = {str(item["path"]): item for item in manifest_files}
        staging.mkdir(parents=True)
        total = 0
        found: set[str] = set()
        try:
            with zipfile.ZipFile(archive) as package:
                members = [item for item in package.infolist() if not item.is_dir()]
                if len(members) > MAX_FILE_COUNT:
                    raise UpdateError("更新包文件数量超过限制")
                for info in members:
                    relative = _safe_relative_path(info.filename).as_posix()
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == 0o120000 or info.flag_bits & 0x1:
                        raise UpdateError(f"更新包包含链接或加密文件：{relative}")
                    if relative not in expected or relative in found:
                        raise UpdateError(f"更新包包含清单之外或重复的文件：{relative}")
                    total += int(info.file_size)
                    if total > MAX_UNCOMPRESSED_BYTES:
                        raise UpdateError("更新包解压后大小超过限制")
                    destination = staging.joinpath(*PurePosixPath(relative).parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    written = 0
                    with package.open(info) as source, destination.open("wb") as output:
                        while block := source.read(1024 * 1024):
                            digest.update(block)
                            written += len(block)
                            output.write(block)
                    item = expected[relative]
                    if written != int(item["size"]) or digest.hexdigest() != item["sha256"]:
                        raise UpdateError(f"更新包内文件校验失败：{relative}")
                    found.add(relative)
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            raise UpdateError(f"更新包无法安全解压：{exc}") from exc
        missing = set(expected) - found
        if missing:
            raise UpdateError(f"更新包缺少签名清单文件：{min(missing)}")
