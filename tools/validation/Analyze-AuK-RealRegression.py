from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools" / "validation" / "Run-AuK-RealRegression.py"
spec = importlib.util.spec_from_file_location("auk_real_regression_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


def metrics(samples: np.ndarray, sample_rate: int) -> dict[str, float | str]:
    signal = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(signal * signal) + 1e-20))
    n_fft = 2048
    hop = 512
    padded = np.pad(signal, (0, max(0, n_fft - len(signal))))
    spectra = np.stack(
        [
            np.abs(np.fft.rfft(padded[index : index + n_fft] * np.hanning(n_fft))) ** 2
            for index in range(0, len(padded) - n_fft + 1, hop)
        ]
    )
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    mean_power = spectra.mean(axis=0)
    total_power = float(mean_power.sum() + 1e-20)
    high_power = float(mean_power[frequencies >= 4_000].sum())
    flatness = float(
        np.exp(np.mean(np.log(spectra + 1e-16), axis=1)).mean()
        / (np.mean(spectra, axis=1).mean() + 1e-20)
    )
    return {
        "sha256": hashlib.sha256(signal.tobytes()).hexdigest(),
        "seconds": len(signal) / sample_rate,
        "rms_db": 20 * math.log10(rms),
        "peak": float(np.max(np.abs(signal))),
        "zero_crossing_rate": float(np.mean(signal[1:] * signal[:-1] < 0)),
        "spectral_centroid_hz": float((mean_power * frequencies).sum() / total_power),
        "high_band_ratio": high_power / total_power,
        "high_band_power": high_power,
        "spectral_flatness": flatness,
    }


def f0_metrics(path: Path) -> dict[str, float]:
    signal, sample_rate = librosa.load(path, sr=16_000, mono=True)
    f0 = librosa.yin(signal, fmin=70, fmax=500, sr=sample_rate, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=signal, frame_length=1024, hop_length=256)[0]
    voiced = f0[(rms > np.quantile(rms, 0.35)) & (f0 < 490)]
    return {
        "f0_median_hz": float(np.median(voiced)),
        "f0_p10_hz": float(np.quantile(voiced, 0.10)),
        "f0_p90_hz": float(np.quantile(voiced, 0.90)),
    }


def load_source(case: dict) -> tuple[np.ndarray, int]:
    crop = 14.0 if case["name"].startswith("speaker_") else None
    telephone = case["name"].startswith("quality_")
    sample_rate, samples = runner.load_audio(Path(case["source_path"]), crop, telephone=telephone)
    return samples, sample_rate


def main() -> int:
    report = json.loads((ROOT / "planning" / "real-regression-v0.2.2.json").read_text(encoding="utf-8"))
    analyzed: dict[str, dict] = {}
    for case in report["cases"]:
        assert case["state"] == "succeeded", f"{case['name']} did not succeed"
        source, source_rate = load_source(case)
        output_path = ROOT / "outputs" / case["request_id"] / "result.wav"
        output, output_rate = sf.read(output_path, dtype="float32", always_2d=True)
        output = output.mean(axis=1, dtype=np.float32)
        source_result = metrics(source, source_rate)
        output_result = metrics(output, output_rate)
        applied = float(case["metadata"]["applied_generation_seconds"])
        assert abs(float(output_result["seconds"]) - applied) <= 0.03
        assert source_result["sha256"] != output_result["sha256"]
        analyzed[case["name"]] = {
            "request_id": case["request_id"],
            "source": source_result,
            "output": output_result,
            "metadata": case["metadata"],
        }

    sad_path = ROOT / "outputs" / analyzed["emotion_sad"]["request_id"] / "result.wav"
    happy_path = ROOT / "outputs" / analyzed["emotion_happy"]["request_id"] / "result.wav"
    sad_f0 = f0_metrics(sad_path)
    happy_f0 = f0_metrics(happy_path)
    analyzed["emotion_sad"]["f0"] = sad_f0
    analyzed["emotion_happy"]["f0"] = happy_f0

    checks = {
        "all_real_tasks_succeeded": all(case["state"] == "succeeded" for case in report["cases"]),
        "emotion_same_seed_has_large_pitch_separation": (
            happy_f0["f0_median_hz"] - sad_f0["f0_median_hz"] > 50
        ),
        "nonverbal_add_reserves_official_time": (
            analyzed["nonverbal_sigh"]["output"]["seconds"]
            - analyzed["nonverbal_sigh"]["source"]["seconds"]
            > 0.70
        ),
        "whisper_is_quiet_and_noise_like": (
            analyzed["whisper"]["output"]["rms_db"] < analyzed["whisper"]["source"]["rms_db"] - 15
            and analyzed["whisper"]["output"]["high_band_ratio"]
            > analyzed["whisper"]["source"]["high_band_ratio"] * 5
        ),
        "deaccent_used_speech_edge_trim": (
            analyzed["deaccent_anhui"]["metadata"]["input_preprocessing"]["vad_trim_bounds_sec"] is not None
        ),
        "enhance_reduces_noise_flatness": (
            analyzed["enhance"]["output"]["spectral_flatness"]
            < analyzed["enhance"]["source"]["spectral_flatness"] * 0.10
        ),
        "bandwidth_extension_restores_high_band_energy": (
            analyzed["quality_bandwidth"]["output"]["high_band_power"]
            > analyzed["quality_bandwidth"]["source"]["high_band_power"] * 2
        ),
        "speaker_prompts_produce_different_outputs": (
            analyzed["speaker_by_order"]["output"]["sha256"]
            != analyzed["speaker_by_content"]["output"]["sha256"]
        ),
        "vocal_extraction_peak_is_limited": (
            analyzed["music_vocals"]["output"]["peak"] <= 0.9501
            and analyzed["music_vocals"]["metadata"]["vocal_output_peak_limited"] is True
        ),
    }
    assert all(checks.values()), json.dumps(checks, ensure_ascii=False, indent=2)
    output = {"checks": checks, "cases": analyzed}
    (ROOT / "planning" / "real-regression-v0.2.2-analysis.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
