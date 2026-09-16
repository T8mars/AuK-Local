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
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from auk_local.audio import encode_float_audio  # noqa: E402
from auk_local.client import LocalClient  # noqa: E402
from auk_local.duration import estimate_tts_seconds  # noqa: E402
from auk_local.task_templates import build_instruction  # noqa: E402


CASES = (
    {"name": "instruct_tts", "key": "instruct_tts", "primary": "一只小猫在叫啊", "secondary": "自然、清晰、温暖"},
    {
        "name": "zero_shot_tts",
        "key": "zero_shot_tts",
        "primary": "Ladies and gentlemen, it is an honor to welcome you today.",
        "source": "zero-shot-tts/ref.wav",
    },
    {
        "name": "content_edit",
        "key": "content_edit",
        "primary": "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
        "source": "content-edit/content.wav",
    },
    {
        "name": "lyric_edit",
        "key": "lyric_edit",
        "primary": "Replace 'rear view' with 'like you' in the lyrics",
        "source": "vocal-edit/vocaledit-en-1-input.wav",
    },
    {"name": "pitch_up_3", "key": "pitch", "primary": "+3", "source": "pitch/pitch-1-input.wav"},
    {"name": "speed_1_5", "key": "speed", "primary": "1.5", "source": "speed/speed-edit-1-input.wav"},
    {"name": "volume_up_10", "key": "volume", "primary": "+10", "source": "energy/energy-edit-1-input.wav"},
    {"name": "emotion_sad", "key": "emotion", "primary": "sad", "source": "emotion-edit/en-1-input.wav"},
    {"name": "emotion_fear", "key": "emotion", "primary": "fearful", "source": "emotion-edit/en-1-input.wav"},
    {"name": "emotion_happy", "key": "emotion", "primary": "happy", "source": "emotion-edit/en-1-input.wav"},
    {
        "name": "timbre",
        "key": "timbre",
        "primary": "低沉而浑厚，语速平稳，吐字清晰，沉稳而富有思考",
        "source": "vc/vc-1-input.wav",
    },
    {"name": "deaccent_anhui", "key": "deaccent", "primary": "去掉方言口音", "source": "accent/accent-anhui-input.wav"},
    {"name": "deaccent_sichuan", "key": "deaccent", "primary": "去掉方言口音", "source": "accent/accent-sichuan-input.wav"},
    {
        "name": "nonverbal_add_sigh",
        "key": "nonverbal",
        "primary": "在“这个月”前增加叹气声",
        "source": "nv/zh-a-input.wav",
    },
    {
        "name": "nonverbal_remove_laughter",
        "key": "nonverbal",
        "primary": "删除音频中所有笑声",
        "source": "nv/zh-b-input.wav",
    },
    {"name": "whisper", "key": "whisper", "primary": "转换成耳语", "source": "whisper/wh-w2n-zh-input.wav"},
    {"name": "whisper_normal_speech", "key": "whisper", "primary": "转换成耳语", "source": "pitch/pitch-1-input.wav"},
    {
        "name": "enhance",
        "key": "enhance",
        "primary": "去噪并去除房间混响",
        "source": "se/se-zh-1-input.wav",
    },
    {
        "name": "quality_telephone",
        "key": "quality",
        "primary": "去掉电话感",
        "source": "pitch/pitch-1-input.wav",
        "telephone": True,
    },
    {
        "name": "speech_separate_second",
        "key": "speech_separate",
        "primary": "第二个开始说话的人",
        "source": "ss/zh-1-input.wav",
        "crop": 14.0,
    },
    {
        "name": "music_vocals",
        "key": "music_separate",
        "primary": "只保留歌声",
        "source": "vocal-extraction/vocal-1-input.wav",
    },
    {
        "name": "target_speaker",
        "key": "target_speaker",
        "primary": "警队规矩",
        "source": "ss/zh-1-input.wav",
        "crop": 14.0,
    },
)

AUTOMATIC_EDIT_MODES = {
    "pitch": "source_auto",
    "speed": "speed",
    "volume": "source_auto",
    "emotion": "emotion",
    "timbre": "source_auto",
    "deaccent": "source_auto",
    "nonverbal": "nonverbal",
    "whisper": "source_auto",
    "enhance": "source_auto",
    "quality": "source_auto",
    "speech_separate": "source_auto",
    "music_separate": "source_auto",
    "target_speaker": "source_auto",
    "content_edit": "content",
    "lyric_edit": "content",
}


def load_source(case: dict):
    relative = case.get("source")
    if relative is None:
        return None, None, 0.0
    path = ROOT / "assets" / "demo-input-audio" / relative
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1, dtype=np.float32)
    crop = case.get("crop")
    if crop is not None:
        mono = mono[: round(float(crop) * sample_rate)]
    if case.get("telephone"):
        waveform = torch.from_numpy(mono).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), 8_000)
        waveform = torchaudio.functional.lowpass_biquad(waveform, 8_000, 3_400)
        waveform = torchaudio.functional.highpass_biquad(waveform, 8_000, 300)
        waveform = torchaudio.functional.resample(waveform, 8_000, int(sample_rate))
        mono = waveform.squeeze(0).numpy()
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    return (int(sample_rate), mono), path, len(mono) / sample_rate


def request_duration(case: dict) -> tuple[float, str]:
    key = case["key"]
    if key == "instruct_tts":
        return estimate_tts_seconds(case["primary"]), "auto"
    if key == "zero_shot_tts":
        return estimate_tts_seconds(case["primary"]), "auto"
    return 48.0, AUTOMATIC_EDIT_MODES[key]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the verified AuK Base task matrix")
    parser.add_argument("--only", action="append", default=[], help="Run only the named case; repeat as needed")
    parser.add_argument("--append", action="store_true", help="Replace matching cases in the existing report")
    args = parser.parse_args(argv)
    report_path = ROOT / "planning" / "real-regression-v0.2.3.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    client = LocalClient("http://127.0.0.1:7860", ROOT / "data" / "session-token")
    selected = [case for case in CASES if not args.only or case["name"] in set(args.only)]
    if len(selected) != (len(set(args.only)) if args.only else len(CASES)):
        raise ValueError(f"Unknown or duplicate --only values: {args.only}")
    if args.append and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_names = {case["name"] for case in selected}
        report["cases"] = [case for case in report.get("cases", []) if case.get("name") not in selected_names]
    else:
        report = {
            "schema": 2,
            "started_at": time.time(),
            "service_health": client.health(),
            "model": "base",
            "fixed_seed": 20260916,
            "cases": [],
        }
    for index, case in enumerate(selected):
        audio, source_path, source_seconds = load_source(case)
        requested_seconds, duration_mode = request_duration(case)
        instruction = build_instruction(case["key"], case["primary"], case.get("secondary", ""))
        payload = {
            "request_id": str(uuid.uuid4()),
            "task_key": case["key"],
            "primary": case["primary"],
            "secondary": case.get("secondary", ""),
            "instruction": instruction,
            "generation_seconds": requested_seconds,
            "duration_mode": duration_mode,
            "model": "base",
            "seed": 20260916,
            "seed_mode": "fixed",
            "nfe_steps": 32,
            "cfg_strength": 2.0,
            "sway_sampling_coef": -1.0,
            "cpu_offload": True,
            "keep_loaded": index + 1 < len(selected),
            "client": "full-real-regression-v0.2.3",
        }
        if audio is not None:
            payload["audio"] = encode_float_audio(audio[1], audio[0])
        print(
            f"START {case['name']} source={source_seconds:.3f}s requested={requested_seconds:.3f}s instruction={instruction}",
            flush=True,
        )
        started = time.time()
        status = None
        request_id = payload["request_id"]
        try:
            submitted = client.submit(payload)
            request_id = submitted["request_id"]
            status = client.wait(request_id, timeout=1_200, poll=0.5)
            entry = {
                "name": case["name"],
                "task_key": case["key"],
                "request_id": request_id,
                "state": status["state"],
                "source_path": source_path.relative_to(ROOT).as_posix() if source_path else None,
                "source_seconds_submitted": source_seconds,
                "source_transform": "telephone_300_3400_hz" if case.get("telephone") else None,
                "instruction": instruction,
                "duration_mode": duration_mode,
                "requested_generation_seconds": requested_seconds,
                "wall_seconds": round(time.time() - started, 3),
            }
            if status["state"] == "succeeded":
                metadata = client.metadata(request_id)
                audio_bytes = client.download_audio(request_id)
                result_path = ROOT / "outputs" / request_id / "result.wav"
                entry.update(
                    {
                        "metadata": metadata,
                        "result_path": result_path.relative_to(ROOT).as_posix(),
                        "result_sha256": hashlib.sha256(audio_bytes).hexdigest(),
                        "stale_duration_ignored": (
                            requested_seconds <= 30
                            or float(metadata["applied_generation_seconds"]) != requested_seconds
                        ),
                    }
                )
            else:
                entry["error"] = status.get("error")
        except Exception as exc:
            entry = {
                "name": case["name"],
                "task_key": case["key"],
                "request_id": request_id,
                "state": "client_error",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(time.time() - started, 3),
            }
        report["cases"].append(entry)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DONE {case['name']} state={entry['state']}", flush=True)
        if entry["state"] != "succeeded":
            break
    report["finished_at"] = time.time()
    expected_names = {case["name"] for case in CASES}
    report["all_succeeded"] = {case.get("name") for case in report["cases"]} == expected_names and all(
        case["state"] == "succeeded" and case.get("stale_duration_ignored", True) for case in report["cases"]
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
