from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from auk_local.audio import encode_float_audio  # noqa: E402
from auk_local.client import LocalClient  # noqa: E402
from auk_local.task_templates import (  # noqa: E402
    build_instruction,
    emotion_duration_multiplier,
    nonverbal_duration_delta,
)


def load_audio(path: Path, crop_seconds: float | None = None, *, telephone: bool = False):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1, dtype=np.float32)
    if crop_seconds is not None:
        mono = mono[: round(crop_seconds * sample_rate)]
    if telephone:
        waveform = torch.from_numpy(mono).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sample_rate, 8_000)
        waveform = torchaudio.functional.lowpass_biquad(waveform, 8_000, 3_400)
        waveform = torchaudio.functional.highpass_biquad(waveform, 8_000, 300)
        waveform = torchaudio.functional.resample(waveform, 8_000, sample_rate)
        mono = waveform.squeeze(0).numpy()
    return int(sample_rate), np.ascontiguousarray(mono, dtype=np.float32)


def main() -> int:
    assets = ROOT / "assets" / "demo-input-audio"
    report_path = ROOT / "planning" / "real-regression-v0.2.2.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        ("emotion_sad", "emotion", "悲伤", assets / "emotion-edit" / "en-1-input.wav", None, False),
        ("emotion_happy", "emotion", "开心", assets / "emotion-edit" / "en-1-input.wav", None, False),
        ("deaccent_anhui", "deaccent", "去掉方言口音", assets / "accent" / "accent-anhui-input.wav", None, False),
        ("nonverbal_sigh", "nonverbal", "在“这个月”前增加叹气声", assets / "nv" / "zh-a-input.wav", None, False),
        ("whisper", "whisper", "转换成耳语", assets / "pitch" / "pitch-1-input.wav", None, False),
        ("timbre", "timbre", "低沉浑厚、沉稳清晰的男声", assets / "vc" / "vc-1-input.wav", None, False),
        ("enhance", "enhance", "去噪并去除房间混响", assets / "se" / "se-zh-1-input.wav", None, False),
        ("quality_telephone", "quality", "去掉电话感", assets / "pitch" / "pitch-1-input.wav", None, True),
        ("quality_bandwidth", "quality", "补充高频并提升清晰度", assets / "pitch" / "pitch-1-input.wav", None, True),
        ("speaker_by_order", "speech_separate", "第二个开始说话的人", assets / "ss" / "zh-1-input.wav", 14.0, False),
        ("speaker_by_content", "target_speaker", "警队规矩", assets / "ss" / "zh-1-input.wav", 14.0, False),
        ("music_vocals", "music_separate", "只保留歌声", assets / "vocal-extraction" / "vocal-1-input.wav", None, False),
    ]
    client = LocalClient("http://127.0.0.1:7860", ROOT / "data" / "session-token")
    report = {"started_at": time.time(), "model": "base", "cases": []}
    for index, (name, task_key, primary, path, crop_seconds, telephone) in enumerate(cases):
        sample_rate, samples = load_audio(path, crop_seconds, telephone=telephone)
        source_seconds = len(samples) / sample_rate
        if task_key == "emotion":
            target_seconds = source_seconds * emotion_duration_multiplier(primary)
            duration_mode = "emotion"
        elif task_key == "nonverbal":
            target_seconds = max(0.1, source_seconds + nonverbal_duration_delta(primary))
            duration_mode = "nonverbal"
        else:
            target_seconds = source_seconds
            duration_mode = "source_auto"
        payload = {
            "request_id": str(uuid.uuid4()),
            "task_key": task_key,
            "primary": primary,
            "secondary": "",
            "instruction": build_instruction(task_key, primary),
            "generation_seconds": target_seconds,
            "duration_mode": duration_mode,
            "model": "base",
            "seed": 20260916,
            "seed_mode": "fixed",
            "nfe_steps": 32,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0,
            "cpu_offload": True,
            "keep_loaded": index + 1 < len(cases),
            "client": "real-regression-v0.2.2",
            "audio": encode_float_audio(samples, sample_rate),
        }
        print(f"START {name} source={source_seconds:.3f}s target={target_seconds:.3f}s", flush=True)
        submitted = client.submit(payload)
        request_id = submitted["request_id"]
        last_phase = None
        while True:
            status = client.status(request_id)
            if status.get("phase") != last_phase:
                last_phase = status.get("phase")
                print(f"  {request_id} {status.get('state')}/{last_phase}", flush=True)
            if status["state"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                break
            time.sleep(0.5)
        entry = {
            "name": name,
            "request_id": request_id,
            "state": status["state"],
            "source_path": str(path),
            "source_seconds_submitted": source_seconds,
            "instruction": payload["instruction"],
        }
        if status["state"] == "succeeded":
            entry["metadata"] = client.metadata(request_id)
        else:
            entry["error"] = status.get("error")
        report["cases"].append(entry)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"DONE {name} state={status['state']}", flush=True)
    report["finished_at"] = time.time()
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return 0 if all(case["state"] == "succeeded" for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
