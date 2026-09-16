from __future__ import annotations

import pytest

from auk_local.config import LocalPaths
from auk_local.inference import InferenceRuntime
from auk_local.preprocess import limit_vocal_output, prepare_model_audio


@pytest.mark.parametrize("keep_loaded", [False, True])
def test_failed_inference_always_unloads_engine(tmp_path, monkeypatch, keep_loaded):
    runtime = InferenceRuntime(LocalPaths.from_root(tmp_path))

    class BrokenEngine:
        def generate(self, *_args, **_kwargs):
            raise RuntimeError("synthetic OOM")

    def fake_load(_model_key, _cpu_offload, _progress):
        runtime.engine = BrokenEngine()
        runtime.loaded_key = "flash:1"
        return runtime.engine

    monkeypatch.setattr(runtime, "_load", fake_load)
    task = {
        "request_id": "00000000-0000-0000-0000-000000000002",
        "instruction": "测试",
        "task_key": "instruct_tts",
        "model": "flash",
        "generation_seconds": 0.2,
        "seed": 42,
        "nfe_steps": 4,
        "cfg_strength": 0.0,
        "sway_sampling_coef": -1.0,
        "cpu_offload": True,
        "keep_loaded": keep_loaded,
    }
    with pytest.raises(RuntimeError, match="synthetic OOM"):
        runtime.execute(task, None, tmp_path / "output", lambda _phase: None)
    assert runtime.engine is None
    assert runtime.loaded_key is None


def test_task_aware_preprocessing_trims_edges_and_normalizes_whisper():
    import torch

    sample_rate = 24_000
    time_axis = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    speech = (0.1 * torch.sin(2 * torch.pi * 220 * time_axis)).unsqueeze(0)
    source = torch.cat(
        [torch.zeros(1, sample_rate), speech, torch.zeros(1, sample_rate)], dim=1,
    )
    emotion, metadata = prepare_model_audio(source, sample_rate, "emotion", "悲伤")
    assert emotion.shape[-1] < source.shape[-1]
    assert metadata["vad_trim_bounds_sec"] is not None

    whisper, metadata = prepare_model_audio(source, sample_rate, "whisper", "转换成耳语")
    assert metadata["whisper_target_rms"] == pytest.approx(0.0064)
    assert metadata["whisper_level_method"] == "lufs"
    import pyloudnorm as pyln

    output_lufs = pyln.Meter(sample_rate).integrated_loudness(whisper.squeeze(0).numpy())
    assert output_lufs == pytest.approx(-44.47, abs=0.02)

    nonverbal, metadata = prepare_model_audio(source, sample_rate, "nonverbal", "增加笑声")
    assert nonverbal.shape[-1] == source.shape[-1]
    assert metadata["vad_trim_bounds_sec"] is None


def test_vocal_output_peak_is_limited_for_lyric_and_music_tasks():
    import torch

    source = torch.tensor([[0.0, 1.5, -2.0]])
    limited, applied = limit_vocal_output(source, "music_separate")
    assert applied is True
    assert limited.abs().max().item() == pytest.approx(0.95)
    unchanged, applied = limit_vocal_output(source, "emotion")
    assert applied is False
    assert torch.equal(unchanged, source)


def test_official_demo_assets_match_prompt_enhancer_unpadded_durations():
    from pathlib import Path

    import soundfile as sf
    import torch

    assets = Path(__file__).resolve().parents[1] / "assets" / "demo-input-audio"
    expected = (
        ("speed/speed-edit-1-input.wav", "speed", "1.5", 10.300),
        ("accent/accent-anhui-input.wav", "deaccent", "去掉方言口音", 3.996),
        ("whisper/wh-w2n-zh-input.wav", "whisper", "转换成耳语", 8.354),
    )
    for relative, task_key, primary, expected_seconds in expected:
        samples, sample_rate = sf.read(assets / relative, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
        _prepared, metadata = prepare_model_audio(waveform, int(sample_rate), task_key, primary)
        assert metadata["vad_engine"] == "silero"
        assert metadata["speech_seconds_unpadded"] == pytest.approx(expected_seconds, abs=0.001)


def test_vocal_output_uses_official_minus_14_lufs_downward_limit():
    import pyloudnorm as pyln
    import torch

    sample_rate = 24_000
    time_axis = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
    source = (0.8 * torch.sin(2 * torch.pi * 440 * time_axis)).unsqueeze(0)
    limited, applied = limit_vocal_output(source, "music_separate", sample_rate)
    measured = pyln.Meter(sample_rate).integrated_loudness(limited.squeeze(0).numpy())
    assert applied is True
    assert measured == pytest.approx(-14.0, abs=0.02)
