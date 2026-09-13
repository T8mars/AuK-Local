from __future__ import annotations

import base64
import hashlib
import math
from array import array
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FloatAudio:
    samples: array
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def read_verified_source(path: Path, request: dict) -> bytes:
    raw = path.read_bytes()
    if (
        not raw
        or len(raw) != int(request["source_frames"]) * 4
        or hashlib.sha256(raw).hexdigest() != request["source_sha256"]
    ):
        raise ValueError("输入音频校验失败，文件已损坏或被修改，请重新上传")
    return raw


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


def encode_gradio_audio(audio_value) -> dict[str, object]:
    """Convert Gradio's numpy audio tuple to normalized mono float PCM."""
    if not isinstance(audio_value, (tuple, list)) or len(audio_value) != 2:
        raise ValueError("Gradio 音频格式无效")
    sample_rate, samples = audio_value

    import numpy as np

    values = np.asarray(samples)
    if values.ndim not in {1, 2} or values.size == 0:
        raise ValueError("输入音频为空或声道格式无效")
    if np.issubdtype(values.dtype, np.signedinteger):
        info = np.iinfo(values.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        values = values.astype(np.float32) / scale
    elif np.issubdtype(values.dtype, np.unsignedinteger):
        info = np.iinfo(values.dtype)
        midpoint = float(info.max + 1) / 2.0
        values = (values.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32, copy=False)
    else:
        raise ValueError(f"不支持的音频数据类型：{values.dtype}")
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    return encode_float_audio(values, int(sample_rate))


def decode_float_audio(payload: dict[str, object]) -> FloatAudio:
    if not isinstance(payload, dict):
        raise TypeError("audio 必须是对象")
    if payload.get("encoding") != "f32le":
        raise ValueError("only f32le audio is supported")
    try:
        sample_rate = int(payload.get("sample_rate", 0))
        channels = int(payload.get("channels", 0))
        frames = int(payload.get("frames", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid audio shape or sample rate") from exc
    if sample_rate <= 0 or channels != 1 or frames <= 0:
        raise ValueError("invalid audio shape or sample rate")
    if sample_rate < 8_000 or sample_rate > 192_000:
        raise ValueError("audio sample_rate 必须在 8000 到 192000 之间")
    if frames > sample_rate * 30:
        raise ValueError("输入音频不能超过 30 秒")
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
