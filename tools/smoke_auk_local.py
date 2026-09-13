from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import torch
import torchaudio

from auk_local.audio import encode_float_audio
from auk_local.config import LocalPaths
from auk_local.manager import TaskManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one end-to-end AuK Local smoke task")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=["flash", "base"], default="flash")
    parser.add_argument("--task-key", default="instruct_tts")
    parser.add_argument("--primary", default="你好，这是 AuK 本地整合包测试。")
    parser.add_argument("--secondary", default="一位声音自然、清晰、平静的年轻女性")
    parser.add_argument("--instruction", default="", help="Override the built-in Chinese task template")
    parser.add_argument("--audio", type=Path, help="Reference or source WAV")
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-loaded", action="store_true")
    args = parser.parse_args()
    payload = {
        "request_id": str(uuid.uuid4()),
        "task_key": args.task_key,
        "primary": args.primary,
        "secondary": args.secondary,
        "instruction": args.instruction,
        "generation_seconds": args.seconds,
        "model": args.model,
        "seed": args.seed,
        "cpu_offload": True,
        "keep_loaded": args.keep_loaded,
        "client": "smoke",
    }
    if args.audio:
        waveform, sample_rate = torchaudio.load(str(args.audio))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        payload["audio"] = encode_float_audio(
            waveform.squeeze(0).to(torch.float32).contiguous().numpy(), sample_rate
        )
    manager = TaskManager(LocalPaths.from_root(args.root))
    try:
        request_id = payload["request_id"]
        record, _ = manager.submit(payload)
        print(json.dumps(record.public_dict(), ensure_ascii=False), flush=True)
        completed = manager.wait(request_id, timeout=1800)
        print(json.dumps(completed.public_dict(), ensure_ascii=False, indent=2), flush=True)
        if completed.state != "succeeded":
            raise SystemExit(1)
        print(Path(completed.metadata_path).read_text(encoding="utf-8"), flush=True)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
