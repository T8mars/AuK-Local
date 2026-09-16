from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


LOCAL = Path(__file__).resolve().parents[2]
WORKSPACE = LOCAL.parent
COMFY = WORKSPACE / "Comfyui-Auk-T8"
DEST = WORKSPACE / "AuK-0.2.3-真实验证结果"
ASSETS = LOCAL / "assets" / "demo-input-audio"

CATEGORY = {
    "instruct_tts": "描述生成语音",
    "zero_shot_tts": "参考声音克隆",
    "content_edit": "语音文字编辑",
    "lyric_edit": "歌词编辑",
    "pitch": "音高编辑",
    "speed": "速度编辑",
    "volume": "音量编辑",
    "emotion": "情绪编辑",
    "timbre": "音色编辑",
    "deaccent": "去口音",
    "nonverbal": "非语言声音编辑",
    "whisper": "耳语转换",
    "enhance": "语音增强",
    "quality": "音质修复",
    "speech_separate": "说话人分离",
    "music_separate": "音乐人声提取",
    "target_speaker": "指定说话人提取",
}


def safe_name(value: str) -> str:
    for char in '<>:"/\\|?*':
        value = value.replace(char, "_")
    return value.strip(" .")


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def task_rows() -> dict[str, dict]:
    connection = sqlite3.connect(LOCAL / "data" / "tasks.db")
    connection.row_factory = sqlite3.Row
    try:
        return {row["request_id"]: dict(row) for row in connection.execute("SELECT * FROM tasks")}
    finally:
        connection.close()


def actual_input(row: dict, destination: Path) -> None:
    request = json.loads(row["request_json"])
    samples = np.fromfile(row["input_path"], dtype="<f4")
    sf.write(destination, samples, int(request["source_sample_rate"]), subtype="FLOAT")


def resolve_local_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else LOCAL / path


def build_local_report(report_path: Path, output_root: Path, rows: dict[str, dict]) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for index, case in enumerate(report.get("cases", []), 1):
        name = safe_name(str(case.get("name") or case.get("task_key") or index))
        label = CATEGORY.get(str(case.get("task_key")), str(case.get("task_key") or "未知"))
        case_dir = output_root / f"{index:02d}-{label}-{name}"
        case_dir.mkdir(parents=True, exist_ok=True)
        row = rows.get(str(case.get("request_id")))
        if row and row.get("input_path") and Path(row["input_path"]).is_file():
            actual_input(row, case_dir / "01-实际提交输入.wav")
        original = resolve_local_path(case.get("source_path"))
        if original and original.is_file():
            copy(original, case_dir / f"00-原始素材{original.suffix.lower()}")
        result = resolve_local_path(case.get("result_path"))
        if result and result.is_file():
            copy(result, case_dir / "02-生成结果.wav")
        metadata = case.get("metadata")
        if metadata is None and row and row.get("metadata_path") and Path(row["metadata_path"]).is_file():
            metadata = json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
        (case_dir / "03-参数与元数据.json").write_text(
            json.dumps({"case": case, "metadata": metadata}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def comfy_submitted_source(case: dict, destination: Path) -> None:
    relative = case.get("source_asset")
    if not relative:
        return
    source = ASSETS / relative
    data, rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.mean(axis=1, dtype=np.float32)).unsqueeze(0)
    transform = case.get("source_transform")
    if isinstance(transform, (int, float)):
        waveform = waveform[:, : round(float(transform) * rate)]
    elif transform == "telephone":
        waveform = torchaudio.functional.resample(waveform, int(rate), 8_000)
        waveform = torchaudio.functional.lowpass_biquad(waveform, 8_000, 3_400)
        waveform = torchaudio.functional.highpass_biquad(waveform, 8_000, 300)
        waveform = torchaudio.functional.resample(waveform, 8_000, int(rate))
    sf.write(destination, waveform.squeeze(0).numpy(), int(rate), subtype="FLOAT")


def build_comfy() -> None:
    report_path = COMFY / "planning" / "comfy-real-v2.0.6.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_root = DEST / "30-ComfyUI原生节点-17任务"
    for index, case in enumerate(report["cases"], 1):
        key = str(case["task_key"])
        case_dir = output_root / f"{index:02d}-{CATEGORY.get(key, key)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        if case.get("source_asset"):
            source = ASSETS / case["source_asset"]
            copy(source, case_dir / f"00-原始素材{source.suffix.lower()}")
            comfy_submitted_source(case, case_dir / "01-实际节点输入.wav")
        result = COMFY / case["result_path"]
        copy(result, case_dir / "02-节点生成结果.wav")
        (case_dir / "03-节点参数与元数据.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )

    effect_dir = DEST / "31-ComfyUI重点效果-受控对照"
    source_dir = COMFY / "planning" / "comfy-effect-v2.0.6"
    for source in source_dir.glob("*.wav"):
        copy(source, effect_dir / source.name)
    copy(COMFY / "planning" / "comfy-effect-v2.0.6.json", effect_dir / "运行参数.json")
    copy(COMFY / "planning" / "comfy-semantic-v2.0.6.json", effect_dir / "独立验证报告.json")


def copy_reports() -> None:
    output = DEST / "90-机器可读验证报告"
    output.mkdir(parents=True, exist_ok=True)
    candidates = (
        LOCAL / "planning" / "real-regression-v0.2.3.json",
        LOCAL / "planning" / "effect-recovery-v0.2.3.json",
        LOCAL / "planning" / "deaccent-comparison-v0.2.3.json",
        WORKSPACE / "semantic-eval" / "reports" / "semantic-v0.2.3.json",
        WORKSPACE / "semantic-eval" / "reports" / "effect-recovery-semantic-v0.2.3.json",
        WORKSPACE / "semantic-eval" / "reports" / "nonverbal-controlled-semantic-v0.2.3.json",
        WORKSPACE / "semantic-eval" / "reports" / "speaker-anchor-v0.2.3.json",
        WORKSPACE / "semantic-eval" / "reports" / "speaker-english-official-semantic-v0.2.3.json",
        WORKSPACE / "semantic-eval" / "reports" / "deaccent-comparison-semantic-v0.2.3.json",
        COMFY / "planning" / "comfy-real-v2.0.6.json",
        COMFY / "planning" / "comfy-semantic-v2.0.6.json",
    )
    for source in candidates:
        if source.is_file():
            copy(source, output / source.name)


def copy_public_test_materials() -> None:
    output = DEST / "00-公共领域测试素材"
    source_root = WORKSPACE / "test-materials" / "public-domain"
    for name in ("wikimedia-laughter-public-domain.ogg", "wikimedia-laughter-public-domain.wav"):
        source = source_root / name
        if source.is_file():
            copy(source, output / name)
    (output / "来源说明.txt").write_text(
        "Wikimedia Commons: Laughter.ogg\n"
        "作者：ezwa\n"
        "许可：Public Domain\n"
        "https://commons.wikimedia.org/wiki/File:Laughter.ogg\n",
        encoding="utf-8",
    )


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    rows = task_rows()
    build_local_report(
        LOCAL / "planning" / "real-regression-v0.2.3.json",
        DEST / "10-本地整合包-完整22组真实推理",
        rows,
    )
    build_local_report(
        LOCAL / "planning" / "effect-recovery-v0.2.3.json",
        DEST / "20-本地整合包-修复对照与重点验证",
        rows,
    )
    build_comfy()
    copy_reports()
    copy_public_test_materials()
    readme = """# AuK 0.2.3 真实验证结果

本目录保存原素材、模型实际收到的输入、生成结果、参数与独立验证报告。结果不是只看“任务成功”或波形不同。

## 已有独立证据支持

- 语音增强：受控噪声/混响输入相对干净参考的频谱距离改善约 88%。
- 音质修复：受控电话带宽输入相对干净参考改善约 15%，内容保留。
- 第二位说话人分离：完整官方中文素材上，输出更接近第二个说话人锚点；本地与 ComfyUI 都实际运行。
- 音乐人声提取：输出对独立 HDemucs 伪人声参考的相似度显著高于伴奏参考。
- 非语言声音：增加叹气通过 AudioSet 验证；公共领域笑声受控素材中，笑声概率从约 0.316 降至 0.0023，文字保留。
- 正常说话转耳语：Whispering 概率显著上升且文字保留。
- 情绪“开心”：官方英文素材、Seed 42 获得独立情绪分类器支持。

## 模型能力限制 / 未强行判通过

- 去口音：没有可靠的本地普通话口音分类器。官方安徽、四川示例和用户素材候选均保留供试听；用户素材会出现文字变化，不能宣称修复成功。
- 情绪“悲伤、恐惧”：官方提示词、Seed 42/20260916、CFG 2.0/3.5 多组结果均未获独立分类器支持，属于当前模型不稳定项。
- 指定说话人提取：中文与英文官方素材都出现说话人锚点证据冲突，标为不确定；结果保留供试听。
- 第一位说话人分离的英文官方例证不稳定；第二位中文官方示例有支持证据。

## 已修的确定代码问题

- 所有任务现在是“输入最多 30 秒、输出最多 30 秒”，不再错误计算输入加输出必须小于 30 秒。
- 质量修复、说话人分离和指定说话人使用经过真实对照验证的官方英文提示模板。
- 本地与 ComfyUI 都使用完整官方说话人素材复测；17 个 ComfyUI 原生节点均完成真实模型推理。

外部笑声素材：Wikimedia Commons `Laughter.ogg`，作者 ezwa，Public Domain：
https://commons.wikimedia.org/wiki/File:Laughter.ogg
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    print(DEST)


if __name__ == "__main__":
    main()
