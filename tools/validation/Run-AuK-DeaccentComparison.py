from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from auk_local.audio import encode_float_audio  # noqa: E402
from auk_local.client import LocalClient  # noqa: E402


VARIANTS = (
    (
        "official_cfg2",
        "请去掉这段语音里的方言口音，保持说话人音色一致。",
        2.0,
    ),
    (
        "explicit_standard_cfg2",
        "请把这段语音转换成标准普通话发音，去除方言口音和方言腔调，保持原文字内容与说话人音色一致。",
        2.0,
    ),
    (
        "explicit_standard_cfg35",
        "请把这段语音转换成标准普通话发音，去除方言口音和方言腔调，保持原文字内容与说话人音色一致。",
        3.5,
    ),
    (
        "strict_standard_cfg35",
        "请将整段语音重新发成字正腔圆的标准普通话，彻底去除原有方言腔调；原文字内容和说话人音色必须保持一致。",
        3.5,
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare de-accent instructions on one user-supplied recording")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    samples, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    mono = np.ascontiguousarray(samples.mean(axis=1, dtype=np.float32))
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    client = LocalClient("http://127.0.0.1:7860", ROOT / "data" / "session-token")
    report_path = ROOT / "planning" / "deaccent-comparison-v0.2.3.json"
    report = {
        "started_at": time.time(),
        "input_filename": args.input.name,
        "input_sha256": input_sha256,
        "input_seconds": len(mono) / sample_rate,
        "seed": 72073881,
        "variants": [],
    }
    for index, (name, instruction, cfg) in enumerate(VARIANTS):
        payload = {
            "request_id": str(uuid.uuid4()),
            "task_key": "deaccent",
            "primary": "去掉方言口音，转换成标准普通话",
            "secondary": "",
            "instruction": instruction,
            "generation_seconds": 48.0,
            "duration_mode": "source_auto",
            "model": "base",
            "seed": 72073881,
            "seed_mode": "fixed",
            "nfe_steps": 32,
            "cfg_strength": cfg,
            "sway_sampling_coef": -1.0,
            "cpu_offload": True,
            "keep_loaded": index + 1 < len(VARIANTS),
            "client": "deaccent-comparison-v0.2.3",
            "audio": encode_float_audio(mono, int(sample_rate)),
        }
        print(f"START {name}", flush=True)
        started = time.time()
        submitted = client.submit(payload)
        request_id = submitted["request_id"]
        status = client.wait(request_id, timeout=1_200)
        entry = {
            "name": name,
            "request_id": request_id,
            "state": status["state"],
            "instruction": instruction,
            "cfg_strength": cfg,
            "wall_seconds": round(time.time() - started, 3),
        }
        if status["state"] == "succeeded":
            output = client.download_audio(request_id)
            entry["metadata"] = client.metadata(request_id)
            entry["result_path"] = f"outputs/{request_id}/result.wav"
            entry["result_sha256"] = hashlib.sha256(output).hexdigest()
        else:
            entry["error"] = status.get("error")
        report["variants"].append(entry)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DONE {name} {entry['state']}", flush=True)
        if entry["state"] != "succeeded":
            break
    report["finished_at"] = time.time()
    report["all_succeeded"] = len(report["variants"]) == len(VARIANTS) and all(
        item["state"] == "succeeded" for item in report["variants"]
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
