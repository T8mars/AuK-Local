from __future__ import annotations

import gc
import json
import os
import time
from array import array
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audio import validate_duration
from .config import LocalPaths, load_model_manifest, model_paths
from .diagnostics import model_file_issues
from .version import VERSION


Progress = Callable[[str], None]


class InferenceRuntime:
    def __init__(self, paths: LocalPaths):
        self.paths = paths
        self.engine = None
        self.loaded_key: str | None = None

    def _load(self, model_key: str, cpu_offload: bool, progress: Progress):
        if model_key not in {"flash", "base"}:
            raise ValueError(f"不支持的模型：{model_key}")
        cache_key = f"{model_key}:{int(cpu_offload)}"
        if self.engine is not None and self.loaded_key == cache_key:
            return self.engine
        self.unload()
        progress("loading")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from auk.infer.infer_auk import AukInfer

        selected = model_paths(self.paths, model_key)
        qwen = model_paths(self.paths, "qwen")["directory"]
        manifest = load_model_manifest()["models"]
        issues: list[str] = []
        for key, directory in ((model_key, selected["directory"]), ("qwen", qwen)):
            missing, invalid = model_file_issues(directory, manifest[key])
            issues.extend(str(directory / name) for name in missing)
            issues.extend(f"{directory / item.rsplit(':', 1)[0]} ({item.rsplit(':', 1)[1]})" for item in invalid)
        if issues:
            raise FileNotFoundError("模型文件缺失或损坏：" + ", ".join(issues))
        self.engine = AukInfer(
            config_path=str(selected["config"]),
            ckpt_path=str(selected["checkpoint"]),
            device="cuda:0",
            dtype="bf16",
            qwen_path=str(qwen),
            cpu_offload=cpu_offload,
        )
        self.loaded_key = cache_key
        return self.engine

    def unload(self) -> None:
        if self.engine is not None:
            del self.engine
            self.engine = None
            self.loaded_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001, S110 - cleanup errors must not hide the original inference result
            pass

    def execute(self, task: dict[str, Any], input_path: str | None, output_dir: Path, progress: Progress) -> dict[str, str]:
        import torch
        import torchaudio

        started = time.time()
        model_key = str(task["model"])
        cpu_offload = bool(task.get("cpu_offload", True))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        keep_loaded = bool(task.get("keep_loaded", False))
        engine = None
        result_path = None
        metadata_path = None
        succeeded = False
        try:
            engine = self._load(model_key, cpu_offload, progress)
            audio = None
            source_seconds = 0.0
            qwen_audio = None
            if input_path:
                source_samples = array("f")
                with open(input_path, "rb") as source_file:
                    source_samples.fromfile(source_file, os.path.getsize(input_path) // 4)
                source_rate = int(task["source_sample_rate"])
                waveform = torch.tensor(source_samples, dtype=torch.float32).unsqueeze(0)
                source_seconds = waveform.shape[-1] / source_rate
                audio = (waveform, source_rate)
                qwen_waveform = waveform
                if source_rate != 16_000:
                    qwen_waveform = torchaudio.functional.resample(waveform, source_rate, 16_000)
                qwen_audio = qwen_waveform.squeeze(0).contiguous().numpy()
            target_seconds = float(task["generation_seconds"])
            validate_duration(source_seconds, target_seconds)
            content: list[dict[str, Any]] = [{"type": "text", "text": str(task["instruction"])}]
            if qwen_audio is not None:
                content.append({"type": "audio", "audio": qwen_audio})
            messages = [{"role": "user", "content": content}]
            progress("encoding")
            torch.manual_seed(int(task["seed"]))
            progress("sampling")
            generated, sample_rate = engine.generate(
                messages,
                audio=audio,
                gen_seconds=target_seconds,
                nfe=int(task.get("nfe_steps", 32)),
                cfg_strength=float(task.get("cfg_strength", 2.0)),
                sway_sampling_coef=float(task.get("sway_sampling_coef", -1.0)),
                seed=int(task["seed"]),
            )
            progress("decoding")
            if (
                not torch.is_tensor(generated)
                or generated.ndim != 2
                or generated.numel() == 0
                or not torch.isfinite(generated).all()
                or not isinstance(sample_rate, int)
                or sample_rate <= 0
            ):
                raise ValueError("模型输出音频为空、包含 NaN/Inf 或格式无效，请重试")
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / "result.wav"
            metadata_path = output_dir / "metadata.json"
            progress("saving")
            torchaudio.save(
                str(result_path),
                generated.to(torch.float32).cpu(),
                int(sample_rate),
                encoding="PCM_F",
                bits_per_sample=32,
            )
            peak_vram = 0
            if torch.cuda.is_available():
                peak_vram = int(torch.cuda.max_memory_allocated())
            manifests = load_model_manifest()["models"]
            model_manifest = manifests[model_key]
            qwen_manifest = manifests["qwen"]
            metadata = {
                "request_id": task["request_id"],
                "instruction": task["instruction"],
                "task_key": task.get("task_key"),
                "model": model_key,
                "model_revision": model_manifest["revision"],
                "model_files_sha256": {
                    name: details["sha256"] for name, details in model_manifest.get("files", {}).items()
                },
                "qwen_revision": qwen_manifest["revision"],
                "qwen_files_sha256": {
                    name: details["sha256"] for name, details in qwen_manifest.get("files", {}).items()
                },
                "auk_local_version": VERSION,
                "model_identity_check": "required files and byte sizes verified at load",
                "seed": int(task["seed"]),
                "requested_generation_seconds": target_seconds,
                "source_seconds": source_seconds,
                "actual_output_seconds": generated.shape[-1] / int(sample_rate),
                "sample_rate": int(sample_rate),
                "nfe_steps": 4 if model_key == "flash" else int(task.get("nfe_steps", 32)),
                "cfg_strength": 0.0 if model_key == "flash" else float(task.get("cfg_strength", 2.0)),
                "sway_sampling_coef": None if model_key == "flash" else float(task.get("sway_sampling_coef", -1.0)),
                "dtype": "bf16_autocast",
                "cpu_offload": cpu_offload,
                "elapsed_seconds": round(time.time() - started, 3),
                "peak_vram_bytes": peak_vram,
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            succeeded = True
            return {"result_path": str(result_path), "metadata_path": str(metadata_path)}
        finally:
            if not succeeded:
                for partial in (result_path, metadata_path):
                    if partial is not None:
                        try:
                            partial.unlink(missing_ok=True)
                        except OSError:
                            pass
            if not keep_loaded or not succeeded:
                del engine
                self.unload()
