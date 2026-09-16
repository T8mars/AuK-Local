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


ASSETS = ROOT / "assets" / "demo-input-audio"
CASES = (
    ("deaccent_official_anhui_seed42", "deaccent", "standard Mandarin", "请去掉这段语音里的方言口音，保持说话人音色一致。", "accent/accent-anhui-input.wav", None, 4.00, 2.0),
    ("deaccent_official_sichuan_seed42", "deaccent", "standard Mandarin", "请去掉这段语音里的方言口音，保持说话人音色一致。", "accent/accent-sichuan-input.wav", None, 6.74, 2.0),
    ("emotion_sad_seed42", "emotion", "sad", "Say this in a sad tone", "emotion-edit/en-1-input.wav", None, 7.60, 2.0),
    ("emotion_fear_seed42", "emotion", "fearful", "Say this in a afraid tone", "emotion-edit/en-1-input.wav", None, 7.22, 2.0),
    ("emotion_happy_seed42", "emotion", "happy", "Say this in a happy tone", "emotion-edit/en-1-input.wav", None, 6.60, 2.0),
    ("emotion_sad_cfg35", "emotion", "sad", "Say this in a sad tone", "emotion-edit/en-1-input.wav", None, 7.60, 3.5),
    ("enhance_controlled_en", "enhance", "controlled", "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.", "pitch/pitch-1-input.wav", "noise_reverb", 5.50, 2.0),
    ("enhance_controlled_zh", "enhance", "controlled", "请保留所有说话人，去除噪声和混响，输出等长的纯净语音。", "pitch/pitch-1-input.wav", "noise_reverb", 5.50, 2.0),
    ("quality_telephone_en", "quality", "telephone", "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech.", "pitch/pitch-1-input.wav", "telephone", 5.50, 2.0),
    ("quality_telephone_cfg35", "quality", "telephone", "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech.", "pitch/pitch-1-input.wav", "telephone", 5.50, 3.5),
    ("speech_second_full", "speech_separate", "second", "Keep only the second speaker", "ss/zh-1-input.wav", None, 18.24, 2.0),
    ("speech_first_full", "speech_separate", "first", "Keep only the first speaker", "ss/zh-1-input.wav", None, 19.28, 2.0),
    ("target_full", "target_speaker", "警队规矩", "Keep only the speaker who says “警队规矩”", "ss/zh-1-input.wav", None, 19.28, 2.0),
    ("speech_first_english_official", "speech_separate", "first", "Keep only the first speaker", "ss/en-1-input.wav", None, 28.00, 2.0),
    ("target_english_official", "target_speaker", "get what", "Keep only the speaker who says “get what”", "ss/en-1-input.wav", None, 28.00, 2.0),
    ("nonverbal_remove_laughter_official", "nonverbal", "remove laughter", "Remove the laughter", "nv/zh-b-input.wav", None, 12.58, 2.0),
    ("nonverbal_remove_laughter_controlled", "nonverbal", "remove laughter", "Remove the laughter", "pitch/pitch-1-input.wav", "laughter_insert", 5.50, 2.0),
    ("deaccent_user_seed42", "deaccent", "standard Mandarin", "请去掉这段语音里的方言口音，保持说话人音色一致。", None, "user_deaccent", 13.92, 2.0),
)


def load_source(
    relative: str | None,
    transform: str | None,
    user_deaccent_audio: Path | None,
) -> tuple[int, np.ndarray, str]:
    if transform == "user_deaccent":
        if user_deaccent_audio is None:
            raise ValueError("The user_deaccent case requires --user-deaccent-audio PATH")
        path = user_deaccent_audio.resolve()
    else:
        path = ASSETS / str(relative)
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1, dtype=np.float32)
    if transform == "telephone":
        wave = torch.from_numpy(mono).unsqueeze(0)
        wave = torchaudio.functional.resample(wave, int(rate), 8_000)
        wave = torchaudio.functional.lowpass_biquad(wave, 8_000, 3_400)
        wave = torchaudio.functional.highpass_biquad(wave, 8_000, 300)
        wave = torchaudio.functional.resample(wave, 8_000, int(rate))
        mono = wave.squeeze(0).numpy()
    elif transform == "noise_reverb":
        rng = np.random.default_rng(20260916)
        noise = rng.standard_normal(len(mono)).astype(np.float32)
        signal_rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
        noise *= signal_rms / (10 ** (8.0 / 20.0)) / max(float(np.sqrt(np.mean(np.square(noise)))), 1e-8)
        # Deterministic room impulse: direct sound plus decaying reflections.
        ir = np.zeros(round(0.32 * rate), dtype=np.float32)
        ir[0] = 1.0
        for delay, gain in ((0.045, 0.42), (0.091, 0.28), (0.147, 0.18), (0.235, 0.10)):
            ir[min(len(ir) - 1, round(delay * rate))] = gain
        mono = np.convolve(mono, ir, mode="full")[: len(mono)].astype(np.float32) + noise
        peak = float(np.max(np.abs(mono)))
        if peak > 0.95:
            mono *= 0.95 / peak
    elif transform == "laughter_insert":
        laughter_path = ROOT.parent / "test-materials" / "public-domain" / "wikimedia-laughter-public-domain.wav"
        laughter, laughter_rate = sf.read(laughter_path, dtype="float32", always_2d=True)
        laughter_mono = laughter.mean(axis=1, dtype=np.float32)
        if int(laughter_rate) != int(rate):
            laughter_wave = torchaudio.functional.resample(
                torch.from_numpy(laughter_mono).unsqueeze(0), int(laughter_rate), int(rate),
            )
            laughter_mono = laughter_wave.squeeze(0).numpy()
        event = laughter_mono[round(1.0 * rate):round(2.3 * rate)]
        anchor = round(2.75 * rate)
        mono = np.concatenate([mono[:anchor], event, mono[anchor:]]).astype(np.float32)
    return int(rate), np.ascontiguousarray(mono, dtype=np.float32), str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--user-deaccent-audio", type=Path)
    args = parser.parse_args(argv)
    selected = [case for case in CASES if not args.only or case[0] in set(args.only)]
    if len(selected) != (len(set(args.only)) if args.only else len(CASES)):
        raise ValueError(f"Unknown or duplicate --only values: {args.only}")
    client = LocalClient("http://127.0.0.1:7860", ROOT / "data" / "session-token")
    report_path = ROOT / "planning" / "effect-recovery-v0.2.3.json"
    if args.append and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_names = {case[0] for case in selected}
        report["cases"] = [case for case in report.get("cases", []) if case.get("name") not in selected_names]
    else:
        report = {"schema": 1, "started_at": time.time(), "seed": 42, "cases": []}
    for index, (name, key, primary, instruction, relative, transform, seconds, cfg) in enumerate(selected):
        rate, samples, source_path = load_source(relative, transform, args.user_deaccent_audio)
        payload = {
            "request_id": str(uuid.uuid4()),
            "task_key": key,
            "primary": primary,
            "secondary": "",
            "instruction": instruction,
            "generation_seconds": seconds,
            "duration_mode": "manual",
            "model": "base",
            "seed": 42,
            "seed_mode": "fixed",
            "nfe_steps": 32,
            "cfg_strength": cfg,
            "sway_sampling_coef": -1.0,
            "cpu_offload": True,
            "keep_loaded": index + 1 < len(selected),
            "client": "effect-recovery-v0.2.3",
            "audio": encode_float_audio(samples, rate),
        }
        print(f"START {name}", flush=True)
        started = time.time()
        try:
            submitted = client.submit(payload)
            request_id = submitted["request_id"]
            status = client.wait(request_id, timeout=1_200, poll=0.5)
            entry = {
                "name": name,
                "task_key": key,
                "request_id": request_id,
                "state": status["state"],
                "source_path": source_path,
                "source_transform": transform,
                "source_sample_rate": rate,
                "source_seconds": len(samples) / rate,
                "instruction": instruction,
                "generation_seconds": seconds,
                "cfg_strength": cfg,
                "wall_seconds": round(time.time() - started, 3),
            }
            if status["state"] == "succeeded":
                result_path = ROOT / "outputs" / request_id / "result.wav"
                entry.update(
                    metadata=client.metadata(request_id),
                    result_path=result_path.relative_to(ROOT).as_posix(),
                    result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                )
            else:
                entry["error"] = status.get("error")
        except Exception as exc:
            entry = {"name": name, "task_key": key, "state": "client_error", "error": f"{type(exc).__name__}: {exc}"}
        report["cases"].append(entry)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"DONE {name} {entry['state']}", flush=True)
        if entry["state"] != "succeeded":
            break
    report["finished_at"] = time.time()
    report["all_succeeded"] = all(c["state"] == "succeeded" for c in report["cases"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["all_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
