from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from auk_local.config import LocalPaths  # noqa: E402
from auk_local.updater import UpdateManager  # noqa: E402


def main() -> int:
    manager = UpdateManager(LocalPaths.from_root(ROOT))
    envelope_path = ROOT / "dist" / "auk-local-release.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    payload = manager._verify_envelope(envelope)
    manager._validate_payload(payload)
    archive = ROOT / "dist" / payload["package"]["name"]
    if archive.stat().st_size != payload["package"]["size"]:
        raise RuntimeError("Archive size differs from signed manifest")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != payload["package"]["sha256"]:
        raise RuntimeError("Archive hash differs from signed manifest")
    with tempfile.TemporaryDirectory(prefix="auk-release-verify-") as folder:
        manager._extract_and_verify(archive, Path(folder) / "extracted", payload["files"])
    print(
        f"AuK Local v{payload['version']} release verification: PASS "
        f"({len(payload['files'])} files, signed and hash-verified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
