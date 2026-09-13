from __future__ import annotations

import pytest

from auk_local.config import LocalPaths
from auk_local.inference import InferenceRuntime


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
