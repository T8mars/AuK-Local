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
    source = torch.cat(
        [torch.zeros(1, sample_rate), torch.full((1, sample_rate), 0.1), torch.zeros(1, sample_rate)], dim=1,
    )
    emotion, metadata = prepare_model_audio(source, sample_rate, "emotion", "悲伤")
    assert emotion.shape[-1] < source.shape[-1]
    assert metadata["vad_trim_bounds_sec"] is not None

    whisper, metadata = prepare_model_audio(source, sample_rate, "whisper", "转换成耳语")
    assert metadata["whisper_target_rms"] == pytest.approx(0.0064)
    assert torch.sqrt(torch.mean(whisper.to(torch.float64) ** 2)).item() == pytest.approx(0.0064)

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
