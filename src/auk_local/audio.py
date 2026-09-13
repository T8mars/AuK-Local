from __future__ import annotations

import base64
import math
from array import array
from dataclasses import dataclass


@dataclass(frozen=True)
class FloatAudio:
    samples: array
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def encode_float_audio(samples, sample_rate: int) -> dict[str, object]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = array("f", (float(value) for value in samples))
    if not values:
        raise ValueError("audio is empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("audio contains NaN or Inf")
    return {
        "encoding": "f32le",
        "sample_rate": int(sample_rate),
        "channels": 1,
        "frames": len(values),
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def decode_float_audio(payload: dict[str, object]) -> FloatAudio:
    if payload.get("encoding") != "f32le":
        raise ValueError("only f32le audio is supported")
    sample_rate = int(payload.get("sample_rate", 0))
    channels = int(payload.get("channels", 0))
    frames = int(payload.get("frames", 0))
    if sample_rate <= 0 or channels != 1 or frames <= 0:
        raise ValueError("invalid audio shape or sample rate")
    raw = base64.b64decode(str(payload.get("data", "")), validate=True)
    if len(raw) != frames * 4:
        raise ValueError("audio byte count does not match frames")
    samples = array("f")
    samples.frombytes(raw)
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("audio contains NaN or Inf")
    return FloatAudio(samples=samples, sample_rate=sample_rate)


def validate_duration(source_seconds: float, target_seconds: float, max_seconds: float = 30.0) -> None:
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("目标时长必须大于 0 秒")
    if source_seconds < 0 or not math.isfinite(source_seconds):
        raise ValueError("输入音频时长无效")
    if source_seconds + target_seconds > max_seconds + 1e-9:
        raise ValueError(
            f"输入 {source_seconds:.2f}s + 输出 {target_seconds:.2f}s = "
            f"{source_seconds + target_seconds:.2f}s，超过 {max_seconds:.0f}s 限制"
        )
