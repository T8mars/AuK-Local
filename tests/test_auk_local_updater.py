from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

from auk_local.config import LocalPaths
from auk_local.updater import UpdateError, UpdateManager, _safe_relative_path, _version_tuple


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_entry(name: str, data: bytes) -> dict[str, object]:
    return {"path": name, "size": len(data), "sha256": sha(data)}


def make_payload(files: list[dict[str, object]], version: str = "9.9.9") -> dict[str, object]:
    return {
        "schema_version": 1,
        "repository": "T8mars/AuK-Local",
        "channel": "stable",
        "version": version,
        "compatibility": {"service_protocol": "1.0"},
        "package": {
            "name": f"AuK-Local-v{version}-code-only.zip",
            "size": 1,
            "sha256": "0" * 64,
        },
        "files": files,
    }


def required_files() -> dict[str, bytes]:
    return {
        "AuK-Local.exe": b"launcher",
        "src/auk_local/version.py": b'VERSION = "9.9.9"\n',
        "scripts/Apply-AuK-Update.ps1": b"# updater\n",
    }


def test_version_order_and_path_policy():
    assert _version_tuple("1.2") < _version_tuple("1.2.1") < _version_tuple("2.0.0")
    assert _safe_relative_path("src/auk_local/ui.py").as_posix() == "src/auk_local/ui.py"
    for unsafe in ("../escape", "runtime/python.exe", "models/model.pt", "assets/demo.wav", "unknown/file"):
        with pytest.raises(UpdateError):
            _safe_relative_path(unsafe)


def test_signed_envelope_rejects_tampering(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    expected = json.dumps({"ok": True}).encode()
    manager = UpdateManager(paths, signature_verifier=lambda payload, signature: payload == expected and signature == "AAAA")
    envelope = {
        "schema_version": 1,
        "signed_payload": base64.b64encode(expected).decode(),
        "signature": "AAAA",
    }
    assert manager._verify_envelope(envelope) == {"ok": True}
    envelope["signed_payload"] = base64.b64encode(b'{"ok":false}').decode()
    with pytest.raises(UpdateError, match="数字签名"):
        manager._verify_envelope(envelope)


def test_extract_requires_exact_manifest_and_rejects_traversal(tmp_path):
    manager = UpdateManager(LocalPaths.from_root(tmp_path), signature_verifier=lambda *_: True)
    contents = required_files()
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, data in contents.items():
            output.writestr(name, data)
    entries = [file_entry(name, data) for name, data in contents.items()]
    destination = tmp_path / "staged"
    manager._extract_and_verify(archive, destination, entries)
    assert (destination / "src/auk_local/version.py").read_bytes() == contents["src/auk_local/version.py"]

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as output:
        output.writestr("../escape.txt", b"bad")
    with pytest.raises(UpdateError, match="越界"):
        manager._extract_and_verify(bad, tmp_path / "bad-stage", [file_entry("AuK-Local.exe", b"bad")])
    assert not (tmp_path / "escape.txt").exists()


def test_stage_latest_verifies_archive_and_writes_atomic_pending(tmp_path, monkeypatch):
    paths = LocalPaths.from_root(tmp_path)
    contents = required_files()
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as output:
        for name, data in contents.items():
            output.writestr(name, data)
    entries = [file_entry(name, data) for name, data in contents.items()]
    payload = make_payload(entries)
    payload["package"]["size"] = source.stat().st_size
    payload["package"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate = {"available": True, "asset_url": "https://github.com/example", "payload": payload}
    manager = UpdateManager(paths, current_version="0.1.0", signature_verifier=lambda *_: True)
    monkeypatch.setattr(manager, "check", lambda: candidate)
    monkeypatch.setattr(manager, "_download", lambda _url, target, _size: shutil.copy2(source, target))
    result = manager.stage_latest()
    pending = json.loads(Path(result["pending_path"]).read_text(encoding="utf-8"))
    assert pending["from_version"] == "0.1.0"
    assert pending["to_version"] == "9.9.9"
    assert not list((paths.data / "updates").glob("*.tmp"))


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows PowerShell required")
def test_external_applier_preserves_protected_data_and_rolls_back(tmp_path):
    package = Path(__file__).resolve().parents[1]
    script = package / "scripts" / "Apply-AuK-Update.ps1"
    root = tmp_path / "package"
    staging = root / "data" / "updates" / "staging" / "test"
    staging.mkdir(parents=True)
    (root / "src/auk_local").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "runtime").mkdir()
    (root / "outputs").mkdir()
    (root / "README.md").write_bytes(b"old readme")
    (root / "src/auk_local/version.py").write_bytes(b"old version")
    protected = {
        root / "models/model.bin": b"model-sentinel",
        root / "runtime/python.exe": b"runtime-sentinel",
        root / "outputs/result.wav": b"output-sentinel",
    }
    for path, data in protected.items():
        path.write_bytes(data)
    new_files = {
        "README.md": b"new readme",
        "src/auk_local/version.py": b"new version",
        "环境诊断.cmd": b"@echo off\r\n",
    }
    for name, data in new_files.items():
        path = staging.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_pending():
        pending = {
            "schema_version": 1,
            "from_version": "0.1.0",
            "to_version": "0.2.0",
            "staging_directory": "data/updates/staging/test",
            "files": [file_entry(name, data) for name, data in new_files.items()],
        }
        path = root / "data/updates/pending-update.json"
        path.write_text(json.dumps(pending), encoding="utf-8")

    write_pending()
    failed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-PackageRoot", str(root), "-NoRestart", "-FailAfterFirstCopy"],
        capture_output=True, text=True, errors="replace", timeout=30, check=False,
    )
    assert failed.returncode != 0
    assert (root / "README.md").read_bytes() == b"old readme"
    write_pending()
    succeeded = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-PackageRoot", str(root), "-NoRestart"],
        capture_output=True, text=True, errors="replace", timeout=30, check=False,
    )
    assert succeeded.returncode == 0, succeeded.stderr
    assert (root / "README.md").read_bytes() == b"new readme"
    assert (root / "环境诊断.cmd").read_bytes() == b"@echo off\r\n"
    for path, data in protected.items():
        assert path.read_bytes() == data


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="Windows PowerShell required")
def test_launcher_hands_exit_42_to_external_updater(tmp_path):
    package = Path(__file__).resolve().parents[1]
    root = tmp_path / "launcher-test"
    (root / "runtime").mkdir(parents=True)
    (root / "src/auk_local").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(package / "AuK-Local.exe", root / "AuK-Local.exe")
    (root / "runtime/python.exe").write_bytes(b"exists")
    (root / "scripts/Start-AuK.cmd").write_text("@echo off\r\nexit /b 42\r\n", encoding="utf-8-sig")
    marker = root / "updater-started.txt"
    (root / "scripts/Apply-AuK-Update.ps1").write_text(
        "param([string]$PackageRoot,[int]$ParentProcessId)\n"
        f"[IO.File]::WriteAllText('{str(marker).replace(chr(39), chr(39) * 2)}','started')\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run([str(root / "AuK-Local.exe")], cwd=root, timeout=20, check=False)
    assert result.returncode == 0
    deadline = time.time() + 10
    while time.time() < deadline and not marker.exists():
        time.sleep(0.1)
    assert marker.read_text() == "started"


def test_windows_update_script_has_utf8_bom_for_chinese_allowlist():
    package = Path(__file__).resolve().parents[1]
    assert (package / "scripts/Apply-AuK-Update.ps1").read_bytes().startswith(b"\xef\xbb\xbf")


def test_windows_batch_launchers_use_consistent_crlf():
    package = Path(__file__).resolve().parents[1]
    scripts = [*package.glob("*.cmd"), *package.joinpath("scripts").glob("*.cmd")]
    assert scripts
    for script in scripts:
        payload = script.read_bytes()
        assert payload.startswith(b"\xef\xbb\xbf"), script
        assert b"\n" not in payload.replace(b"\r\n", b""), script
