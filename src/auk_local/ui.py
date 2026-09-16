from __future__ import annotations

import html
import json
import math
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .audio import encode_gradio_audio, read_verified_source
from .duration import TTS_TASK_KEYS, estimate_tts_seconds
from .manager import TaskManager
from .result_files import open_outputs_directory, save_result_copy
from .task_templates import (
    TASK_BY_KEY,
    TASK_BY_LABEL,
    TASK_GUIDES,
    TASKS,
    build_instruction,
    content_scaled_seconds,
    emotion_duration_multiplier,
    nonverbal_duration_delta,
    parse_speed_multiplier,
)
from .updater import UpdateError, UpdateManager
from .version import VERSION


CSS = """
:root { color-scheme: light !important; }
body, .gradio-container { background: #f8fafc !important; color: #0f172a !important; }
.gradio-container { padding-bottom: 92px !important; }
.auk-header { padding: 22px 28px; border: 1px solid rgba(251,114,153,.28); border-radius: 18px;
  background: linear-gradient(135deg, rgba(251,114,153,.13), rgba(89,125,255,.08)); margin-bottom: 14px; }
.auk-header h1 { margin: 6px 0 !important; color: #0f172a; }
.auk-eyebrow { color: #fb7299; font-size: 12px; font-weight: 750; letter-spacing: .11em; }
.auk-subtitle { color: #475569; }
.auk-byline { margin-top: 8px; color: #334155; font-size: 14px; font-weight: 650; }
.auk-byline a { color: #d94f91 !important; text-decoration: none !important; }
.auk-social-links { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.auk-social-link { display: inline-flex; align-items: center; min-height: 32px; padding: 5px 11px;
  border: 1px solid rgba(217,79,145,.24); border-radius: 999px; background: rgba(255,255,255,.78);
  color: #334155 !important; font-size: 13px; font-weight: 650; text-decoration: none !important;
  transition: border-color .16s ease, color .16s ease, transform .16s ease; }
.auk-social-link:hover { border-color: #d94f91; color: #d94f91 !important; transform: translateY(-1px); }
.auk-task-guide { margin: 2px 0 10px; padding: 12px 14px; border: 1px solid rgba(89,125,255,.20);
  border-left: 4px solid #597dff; border-radius: 12px; background: #f8faff; color: #334155; line-height: 1.65; }
.auk-task-guide-title { color: #3156c8; font-weight: 800; }
.auk-task-guide-example { margin-top: 4px; color: #0f172a; }
.auk-task-guide-note { margin-top: 3px; color: #64748b; font-size: 13px; }
.auk-budget { padding: 10px 14px; border: 1px solid rgba(89,125,255,.22); border-radius: 12px;
  background: rgba(89,125,255,.055); color: #3156c8; }
.auk-status { min-height: 46px; padding: 12px 14px; border-radius: 12px; background: #fff; border: 1px solid #e2e8f0; }
.primary { background: #d94f91 !important; border-color: #d94f91 !important; }
.auk-model-status { margin: 0 0 14px; padding: 11px 14px; border: 1px solid #e2e8f0; border-radius: 12px;
  background: rgba(255,255,255,.86); color: #475569; }
.auk-model-status strong { color: #0f172a; }
.auk-updater { margin: 0 0 14px; padding: 14px !important; border: 1px solid rgba(251,114,153,.32);
  border-radius: 14px; background: #fff7fb; box-shadow: 0 4px 14px rgba(217,79,145,.08); }
.auk-updater button { min-height: 46px !important; font-weight: 750 !important; }
.auk-actions { position: fixed !important; left: 50%; bottom: 12px; transform: translateX(-50%);
  width: min(1160px, calc(100% - 32px));
  z-index: 20; padding: 10px !important; border: 1px solid rgba(251,114,153,.25); border-radius: 14px;
  background: rgba(255,255,255,.94); box-shadow: 0 8px 28px rgba(15,23,42,.12); backdrop-filter: blur(12px); }
"""


def build_ui(
    manager: TaskManager,
    paths,
    *,
    update_manager: UpdateManager | None = None,
    request_update_restart=None,
    ensure_no_active_tasks=None,
):
    import gradio as gr

    from .diagnostics import inspect_models

    task_labels = [task.label for task in TASKS]
    updater = update_manager or UpdateManager(paths)

    def task_guide_html(label):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        if task is None:
            return "<aside class='auk-task-guide'>请选择任务类型后查看官方用法。</aside>"
        guide = TASK_GUIDES[task.key]
        audio_requirement = "必须上传音频" if task.needs_audio else "无需上传音频"
        duration_requirement = {
            "tts": "时长：默认按目标文本自动估算，也可关闭后手动设置。",
            "source": "时长：按原音频精确长度自动锁定。",
            "speed": "时长：按“原音频时长 ÷ 速度倍率”自动锁定。",
            "emotion": "时长：按官方情绪系数自动锁定（悲伤 ×1.22、恐惧 ×1.16、其余 ×1.06）。",
            "content": "时长：按原音频与本次文字增删比例自动估算；填写完整原文/歌词会更准确。",
            "nonverbal": "时长：按官方非语言事件增删系数，在原音频时长上自动调整。",
            "manual": "时长：可手动设置，也可勾选“按原音频时长”。",
        }[task.duration_strategy]
        return (
            "<aside class='auk-task-guide'>"
            f"<div class='auk-task-guide-title'>📘 官方用法 · {html.escape(task.label)} · {audio_requirement}</div>"
            f"<div>{html.escape(guide.requirement)}</div>"
            f"<div class='auk-task-guide-example'><strong>示例：</strong>{html.escape(guide.example)}</div>"
            f"<div class='auk-task-guide-note'>{html.escape(duration_requirement)}</div>"
            f"<div class='auk-task-guide-note'>{html.escape(guide.note)}</div>"
            "</aside>"
        )

    def check_program_update():
        try:
            result = updater.check()
            if result["available"]:
                note = f" · {result['notes']}" if result.get("notes") else ""
                return (
                    f"发现新版 {result['version']}（当前 {VERSION}）{note}。签名已验证，可以安装。",
                    gr.update(interactive=True),
                )
            return f"当前版本 {VERSION} 已是最新版。", gr.update(interactive=False)
        except UpdateError as exc:
            return f"检查更新失败：{exc}", gr.update(interactive=False)

    def install_program_update():
        try:
            if ensure_no_active_tasks is not None:
                ensure_no_active_tasks()
            result = updater.stage_latest()
            if request_update_restart is None:
                raise RuntimeError("当前启动方式不支持自动重启，请使用根目录 AuK-Local.exe")
            request_update_restart()
            return (
                f"版本 {result['version']} 已下载并逐文件校验。程序将自动关闭、安装并重新启动；模型和用户数据不会被修改。",
                gr.update(interactive=False),
            )
        except (UpdateError, RuntimeError, sqlite3.Error, OSError) as exc:
            return f"安装更新失败：{exc}", gr.update(interactive=True)

    def source_audio_seconds(audio_value):
        if audio_value is None:
            raise ValueError("请先上传音频")
        sample_rate, samples = audio_value
        sample_rate = float(sample_rate)
        try:
            frame_count = int(samples.shape[0])
        except (AttributeError, IndexError, TypeError):
            frame_count = len(samples)
        if not math.isfinite(sample_rate) or sample_rate <= 0 or frame_count <= 0:
            raise ValueError("音频数据无效")
        return frame_count / sample_rate

    def source_audio_info_html(audio_value):
        if audio_value is None:
            return (
                "<div class='auk-budget'>尚未上传音频。波形中拖选只是在选择；"
                "请点击右下角剪刀，再点 Trim（确认），或使用下面的明确截取按钮，实际时长才会改变。</div>"
            )
        try:
            seconds = source_audio_seconds(audio_value)
        except (TypeError, ValueError, OverflowError):
            return "<div class='auk-budget'>音频数据无效，请重新上传。</div>"
        return (
            f"<div class='auk-budget'><strong>实际提交输入：{seconds:.2f} 秒</strong>。"
            "这个数值才是模型收到的长度；波形选区必须点击剪刀并再点 Trim（确认）后才生效；"
            "点击剪刀左侧的 ↶ 可恢复最初上传音频。</div>"
        )

    def apply_source_trim(
        label, primary, secondary, duration, auto_duration, source_duration, audio_value, trim_start, trim_end,
    ):
        if audio_value is None:
            raise gr.Error("请先上传音频")
        import numpy as np

        sample_rate, samples = audio_value
        values = np.asarray(samples)
        total_seconds = source_audio_seconds(audio_value)
        try:
            start = float(trim_start or 0.0)
            end = float(trim_end or 0.0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise gr.Error("截取开始和结束必须是秒数") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise gr.Error("截取开始和结束必须是有限秒数")
        if end <= 0:
            end = total_seconds
        if start < 0 or end <= start or end > total_seconds + 1e-6:
            raise gr.Error(f"截取范围必须在 0 到 {total_seconds:.2f} 秒内，且结束要大于开始")
        start_frame = min(values.shape[0] - 1, max(0, int(round(start * float(sample_rate)))))
        end_frame = min(values.shape[0], max(start_frame + 1, int(round(end * float(sample_rate)))))
        clipped = (int(sample_rate), values[start_frame:end_frame].copy())
        duration_update, budget_update = update_duration_control(
            label, primary, duration, auto_duration, source_duration, clipped, secondary,
        )
        return clipped, duration_update, budget_update, source_audio_info_html(clipped)

    def effective_ui_duration(
        label, primary, duration, auto_duration, source_duration=False, audio_value=None, secondary="",
    ):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        if task is not None and task.duration_strategy == "speed":
            return source_audio_seconds(audio_value) / parse_speed_multiplier(primary), "speed"
        if task is not None and task.duration_strategy == "source":
            return source_audio_seconds(audio_value), "source_auto"
        if task is not None and task.duration_strategy == "emotion":
            return source_audio_seconds(audio_value) * emotion_duration_multiplier(primary), "emotion"
        if task is not None and task.needs_audio and bool(source_duration):
            return source_audio_seconds(audio_value), "source"
        if task is not None and task.duration_strategy == "content":
            return content_scaled_seconds(
                task.key, primary, source_audio_seconds(audio_value), str(secondary or "").strip(),
            ), "content"
        if task is not None and task.duration_strategy == "nonverbal":
            return max(0.1, source_audio_seconds(audio_value) + nonverbal_duration_delta(primary)), "nonverbal"
        if task is not None and task.key in TTS_TASK_KEYS and bool(auto_duration):
            return estimate_tts_seconds(primary), "auto"
        return float(duration), "manual"

    def budget_html(
        label, audio_value, duration, primary="", auto_duration=False, source_duration=False, secondary="",
    ):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        if task is None:
            return "<div class='auk-budget'>请选择任务类型。</div>"
        source_seconds = 0.0
        try:
            if task.needs_audio and audio_value is not None:
                source_seconds = source_audio_seconds(audio_value)
            target_seconds, duration_mode = effective_ui_duration(
                label, primary, duration, auto_duration, source_duration, audio_value, secondary,
            )
            if not math.isfinite(target_seconds) or target_seconds <= 0:
                raise ValueError("invalid duration")
        except (TypeError, ValueError, OverflowError):
            if task.duration_strategy == "speed" and audio_value is None:
                return "<div class='auk-budget'>速度编辑：请先上传音频，输出时长将按“输入时长 ÷ 速度倍率”自动计算。</div>"
            if task.duration_strategy == "source" and audio_value is None:
                return "<div class='auk-budget'>此任务按原音频时长生成：请先上传音频。</div>"
            if task.duration_strategy == "emotion" and audio_value is None:
                return "<div class='auk-budget'>情绪编辑：请先上传音频，输出时长将按官方情绪系数自动计算。</div>"
            if task.duration_strategy == "content" and audio_value is None:
                return "<div class='auk-budget'>文字/歌词编辑：请先上传音频，输出时长将按文字变化自动估算。</div>"
            if task.duration_strategy == "nonverbal" and audio_value is None:
                return "<div class='auk-budget'>非语言声音编辑：请先上传音频，输出时长将按事件增删自动估算。</div>"
            if task.needs_audio and bool(source_duration) and audio_value is None:
                return "<div class='auk-budget'>按原音频时长：请先上传音频。</div>"
            return "<div class='auk-budget'>输入数据无效，请检查音频和目标时长。</div>"
        state = "可提交" if source_seconds <= 30.0 + 1e-9 and target_seconds <= 30.0 + 1e-9 else "已超出限制"
        mode = {
            "auto": "自动估算 TTS",
            "source": "按原音频时长",
            "source_auto": "官方等长任务",
            "speed": "速度倍率自动计算",
            "emotion": "官方情绪系数自动计算",
            "content": "官方文字变化自动估算",
            "nonverbal": "官方声音增删自动估算",
            "manual": "手动设置",
        }[duration_mode]
        return (
            f"<div class='auk-budget'>{mode} · 单项 30 秒限制："
            f"输入 {source_seconds:.2f}s · 输出 {target_seconds:.2f}s · {state}</div>"
        )

    def update_duration_control(
        label, primary, duration, auto_duration, source_duration, audio_value, secondary="",
    ):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        locked = bool(
            task is not None
            and (
                task.key == "speed"
                or task.duration_strategy == "source"
                or task.duration_strategy == "emotion"
                or task.duration_strategy in {"content", "nonverbal"}
                or (task.key in TTS_TASK_KEYS and auto_duration)
                or (task.needs_audio and source_duration)
            )
        )
        try:
            target_seconds, duration_mode = effective_ui_duration(
                label, primary, duration, auto_duration, source_duration, audio_value, secondary,
            )
            # Automatic strategies may legitimately estimate a value outside the
            # slider's display range.  Keep the real estimate in budget_html() and
            # recompute it again at submission time, but never feed an invalid
            # value to Gradio's 0.2–30 second slider: Gradio rejects the whole
            # event and marks every output component as an error.
            display_seconds = min(30.0, max(0.2, target_seconds))
            update = gr.update(value=display_seconds, interactive=duration_mode == "manual")
        except (TypeError, ValueError, OverflowError):
            update = gr.update(interactive=not locked)
        return update, budget_html(
            label, audio_value, duration, primary, auto_duration, source_duration, secondary,
        )

    def refresh_source_audio(
        label, primary, duration, auto_duration, source_duration, audio_value, secondary="",
    ):
        """Synchronize server state after upload, clear, recording, or waveform trimming."""
        duration_update, budget_update = update_duration_control(
            label, primary, duration, auto_duration, source_duration, audio_value, secondary,
        )
        return (
            duration_update,
            budget_update,
            source_audio_info_html(audio_value),
            gr.update(value=0.0),
            gr.update(value=0.0),
        )

    def choose_auto_duration(label, primary, duration, auto_duration, source_duration, audio_value, secondary=""):
        resolved_source_duration = False if bool(auto_duration) else bool(source_duration)
        duration_update, budget_update = update_duration_control(
            label, primary, duration, auto_duration, resolved_source_duration, audio_value, secondary,
        )
        return gr.update(value=resolved_source_duration), duration_update, budget_update

    def choose_source_duration(label, primary, duration, auto_duration, source_duration, audio_value, secondary=""):
        resolved_auto_duration = False if bool(source_duration) else bool(auto_duration)
        duration_update, budget_update = update_duration_control(
            label, primary, duration, resolved_auto_duration, source_duration, audio_value, secondary,
        )
        return gr.update(value=resolved_auto_duration), duration_update, budget_update

    def update_seed_control(random_seed):
        return gr.update(interactive=not bool(random_seed))

    def update_task(
        label, audio_value, duration, primary="", auto_duration=False, source_duration=False, secondary="",
    ):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        if task is None:
            return (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                budget_html(
                    label, audio_value, duration, primary, auto_duration, source_duration, secondary,
                ), task_guide_html(label),
            )
        # AuK's official editor prompts are single-operation templates. Appending
        # free-form text changes the trained prompt and can make output copy the source.
        hides_secondary = task.key not in {"instruct_tts", "content_edit", "lyric_edit"}
        secondary_update = gr.update(label=task.secondary_label, visible=not hides_secondary)
        secondary_update["value"] = ""
        resolved_auto_duration = bool(auto_duration) if task.key in TTS_TASK_KEYS else False
        optional_source_duration = task.needs_audio and task.duration_strategy in {
            "tts", "manual", "content", "nonverbal",
        }
        resolved_source_duration = bool(source_duration) if optional_source_duration else False
        if resolved_auto_duration and resolved_source_duration:
            resolved_source_duration = False
        duration_update, budget_update = update_duration_control(
            label, primary, duration, resolved_auto_duration, resolved_source_duration, audio_value, secondary,
        )
        duration_update["label"] = {
            "speed": "自动输出时长（原音频时长 ÷ 速度倍率）",
            "source": "自动输出时长（按原音频）",
            "emotion": "自动输出时长（按官方情绪系数）",
            "content": "自动输出时长（按文字变化估算）",
            "nonverbal": "自动输出时长（按声音增删估算）",
        }.get(task.duration_strategy, "目标时长（秒）")
        source_duration_label = {
            "source": "按原音频时长（本任务自动控制）",
            "speed": "按原音频时长（速度倍率自动控制）",
            "emotion": "按原音频时长（官方情绪系数自动控制）",
        }.get(task.duration_strategy, "按原音频时长")
        return (
            gr.update(label=task.primary_label, placeholder=f"示例：{TASK_GUIDES[task.key].example}", value=""),
            secondary_update,
            gr.update(label="参考声音" if task.key == "zero_shot_tts" else "待处理音频", visible=task.needs_audio),
            gr.update(value=resolved_auto_duration, visible=task.key in TTS_TASK_KEYS),
            gr.update(
                value=resolved_source_duration,
                label=source_duration_label,
                visible=True,
                interactive=optional_source_duration,
            ),
            duration_update,
            budget_update,
            task_guide_html(label),
        )

    def preview_instruction(label, primary, secondary):
        if label not in TASK_BY_LABEL:
            return "请选择任务类型。"
        if not str(primary or "").strip():
            return "请先填写主要内容；最终模型指令会在这里预览。"
        try:
            return build_instruction(TASK_BY_LABEL[label].key, primary, secondary)
        except ValueError as exc:
            return f"输入有误：{exc}"

    def recent_rows():
        state_labels = {
            "queued": "排队中",
            "loading": "加载中",
            "encoding": "编码中",
            "sampling": "采样中",
            "decoding": "解码中",
            "saving": "保存中",
            "succeeded": "成功",
            "failed": "失败",
            "cancelled": "已取消",
            "interrupted": "已中断",
            "cancelling": "取消中",
        }
        rows = []
        try:
            records = manager.store.list_recent(50)
        except (sqlite3.Error, OSError):
            return []
        for record in records:
            task = TASK_BY_KEY.get(str(record.request.get("task_key")))
            rows.append(
                [
                    time.strftime("%m-%d %H:%M:%S", time.localtime(record.created_at)),
                    record.request_id,
                    task.label if task else record.request.get("task_key", ""),
                    str(record.request.get("primary") or "")[:60],
                    record.request.get("model", ""),
                    record.request.get("seed", ""),
                    "可播放" if record.input_path and Path(record.input_path).is_file() else (
                        "已丢失" if record.input_path else "无"
                    ),
                    state_labels.get(record.state, record.state),
                    record.error or "",
                ]
            )
        return rows

    def history_detail(request_id):
        request_id = str(request_id or "").strip()
        if not request_id:
            return (
                "", "请点击历史记录中的任意一行。", None, None,
                gr.update(value="", label="当时填写 · 主要内容"),
                gr.update(value="", label="当时填写 · 附加内容", visible=False),
                "", "",
            )
        try:
            record = manager.get(request_id)
        except (KeyError, sqlite3.Error, OSError) as exc:
            return (
                request_id, f"读取历史任务失败：{exc}", None, None,
                gr.update(value="", label="当时填写 · 主要内容"),
                gr.update(value="", label="当时填写 · 附加内容", visible=False),
                "", "",
            )

        task = TASK_BY_KEY.get(str(record.request.get("task_key") or ""))
        state_labels = {
            "queued": "排队中", "loading": "加载中", "encoding": "编码中", "sampling": "采样中",
            "decoding": "解码中", "saving": "保存中", "succeeded": "成功", "failed": "失败",
            "cancelled": "已取消", "interrupted": "已中断", "cancelling": "取消中",
        }
        result_value = record.result_path if record.result_path and Path(record.result_path).is_file() else None
        result_note = "生成结果尚不可用或已丢失"
        if result_value:
            try:
                import soundfile as sf

                result_info = sf.info(result_value)
                if result_info.frames <= 0 or result_info.samplerate <= 0:
                    raise ValueError("生成结果为空")
                result_note = "生成结果可播放"
            except (OSError, RuntimeError, ValueError):
                result_value = None
                result_note = "生成结果损坏，无法播放"
        input_value = None
        input_note = "该任务不需要参考/源音频"
        if record.input_path:
            input_note = "参考/源音频已丢失或损坏"
            try:
                import numpy as np

                raw = read_verified_source(Path(record.input_path), record.request)
                input_value = (
                    int(record.request["source_sample_rate"]),
                    np.frombuffer(raw, dtype=np.float32).copy(),
                )
                input_note = "参考/源音频可播放"
            except (OSError, KeyError, TypeError, ValueError):
                input_value = None

        metadata_value: dict = {}
        metadata_note = "运行参数文件不可用"
        if record.metadata_path and Path(record.metadata_path).is_file():
            try:
                metadata_value = json.loads(Path(record.metadata_path).read_text(encoding="utf-8"))
                metadata_note = "运行参数可用"
            except (OSError, ValueError, TypeError):
                metadata_value = {}
        status = (
            f"{state_labels.get(record.state, record.state)} · "
            f"{task.label if task else record.request.get('task_key', '')} · "
            f"{result_note} · {input_note} · {metadata_note}"
        )
        if record.error:
            status += f" · {record.error}"
        primary_label = task.primary_label if task else "主要内容"
        secondary_label = task.secondary_label if task else "附加内容"
        secondary_value = str(record.request.get("secondary") or "")
        detail_json = json.dumps(
            {
                "request": record.request,
                "state": record.state,
                "phase": record.phase,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "result_available": result_value is not None,
                "input_available": input_value is not None,
                "metadata": metadata_value,
            },
            ensure_ascii=False,
            indent=2,
        )
        return (
            request_id,
            status,
            result_value,
            input_value,
            gr.update(value=str(record.request.get("primary") or ""), label=f"当时填写 · {primary_label}"),
            gr.update(
                value=secondary_value,
                label=f"当时填写 · {secondary_label}",
                visible=bool(secondary_value),
            ),
            str(record.request.get("instruction") or ""),
            detail_json,
        )

    def select_history(evt):
        row = evt.row_value if evt is not None and isinstance(evt.row_value, (list, tuple)) else None
        request_id = row[1] if row and len(row) > 1 else ""
        return history_detail(request_id)

    # gr is imported inside build_ui; deferred annotations cannot resolve it.
    # An actual EventData class is required for Gradio to inject row selection.
    select_history.__annotations__["evt"] = gr.SelectData

    def save_selected_result(request_id):
        request_id = str(request_id or "").strip()
        if not request_id:
            return "请先生成成功，或在历史记录中选择一项成功任务。"
        try:
            record = manager.get(request_id)
            if record.state != "succeeded" or not record.result_path:
                return "请先生成成功，或在历史记录中选择一项成功任务。"
            saved = save_result_copy(record.result_path, paths.outputs)
            return f"已保存并通过 SHA-256 校验：{saved}"
        except (KeyError, sqlite3.Error, OSError, ValueError) as exc:
            return f"保存失败：{exc}。原始结果仍可从输出目录取得。"

    def open_result_folder():
        try:
            open_outputs_directory(paths.outputs)
            return f"已打开输出目录：{paths.outputs}"
        except OSError as exc:
            return f"打开失败：{exc}。输出目录：{paths.outputs}"

    def release_loaded_models():
        try:
            released = manager.unload_models()
            if released:
                return "已释放 AuK 与 Qwen 模型显存；下次生成会重新加载。"
            return "当前没有运行中的模型进程，无需释放。"
        except (RuntimeError, TimeoutError, OSError) as exc:
            return f"释放失败：{exc}"

    def task_updates(request_id, last_audio, is_current):
        if last_audio and not Path(last_audio).is_file():
            last_audio = None
        last_phase = None
        last_emit = 0.0
        phase_labels = {
            "queued": "等待执行",
            "loading": "阶段 1/5 · 正在加载模型（首次通常需要 30–110 秒）",
            "encoding": "阶段 2/5 · 正在编码文本和参考音频",
            "sampling": "阶段 3/5 · 正在生成音频",
            "decoding": "阶段 4/5 · 正在检查并解码结果",
            "saving": "阶段 5/5 · 正在保存 WAV 和运行参数",
            "cancelling": "正在取消",
        }
        while True:
            if not is_current():
                return
            try:
                record = manager.get(request_id)
            except (sqlite3.Error, OSError) as exc:
                yield f"读取任务失败：{exc}。请检查磁盘或服务日志。", None, last_audio, "", request_id, last_audio, recent_rows()
                return
            scheduler = manager.scheduler_health
            if record.state not in {"succeeded", "failed", "cancelled", "interrupted"} and (
                scheduler["state"] in {"paused", "stopping", "stopped"} or scheduler["dispatcher_alive"] is False
            ):
                detail = scheduler["error"] or "调度不可用"
                yield (
                    f"任务已暂停：{detail}。请恢复存储并重启服务，然后从历史记录重试。",
                    None, last_audio, "", request_id, last_audio, recent_rows(),
                )
                return
            if record.state in {"succeeded", "failed", "cancelled", "interrupted"}:
                break
            now = time.monotonic()
            if record.phase != last_phase or now - last_emit >= 1.0:
                last_phase = record.phase
                last_emit = now
                elapsed = max(0, int(time.time() - record.created_at))
                seed_value = record.request.get("seed", "")
                phase_text = phase_labels.get(record.phase, record.phase)
                yield (
                    f"任务 {request_id[:8]} · Seed {seed_value} · {phase_text} · 已用时 {elapsed} 秒",
                    None, last_audio, "", request_id, last_audio, recent_rows(),
                )
            time.sleep(0.25)
        if record.state == "succeeded":
            if not record.result_path or not Path(record.result_path).is_file():
                yield "结果音频不可用，可能已被清理或移动。", None, last_audio, "", request_id, last_audio, recent_rows()
                return
            message = "生成完成"
            metadata = ""
            try:
                if not record.metadata_path:
                    raise FileNotFoundError("参数路径缺失")
                metadata = Path(record.metadata_path).read_text(encoding="utf-8")
                details = json.loads(metadata)
                if "seed" in details:
                    message += f" · Seed {details['seed']}"
                if "elapsed_seconds" in details:
                    message += f" · 用时 {float(details['elapsed_seconds']):.1f} 秒"
            except (OSError, ValueError):
                message = "音频已生成，但参数文件缺失、损坏或不可读。"
                metadata = ""
            yield message, record.result_path, last_audio, metadata, request_id, record.result_path, recent_rows()
        else:
            yield f"任务{record.state}：{record.error or ''}", None, last_audio, "", request_id, last_audio, recent_rows()

    def begin_action(view_state):
        if view_state is None:
            return None
        lock = view_state.setdefault("_lock", threading.Lock())
        with lock:
            ticket = view_state.get("next_ticket", 0) + 1
            view_state["next_ticket"] = ticket
            return ticket

    def can_report(view_state, ticket):
        if view_state is None:
            return True
        with view_state["_lock"]:
            return ticket >= view_state.get("displayed_ticket", 0)

    def unchanged_outputs():
        return tuple(gr.skip() for _ in range(7))

    def wait_for_result(request_id, last_audio, view_state=None, ticket=None):
        owner = uuid.uuid4().hex
        if view_state is not None:
            with view_state["_lock"]:
                obsolete = ticket < view_state.get("displayed_ticket", 0)
                if not obsolete:
                    view_state["displayed_ticket"] = ticket
                    view_state["owner"] = owner
            if obsolete:
                yield unchanged_outputs()
                return

        def is_current():
            return view_state is None or view_state.get("owner") == owner

        for update in task_updates(request_id, last_audio, is_current):
            if not is_current():
                yield unchanged_outputs()
                return
            yield update
        if not is_current():
            # Gradio replays the event's last streamed value in its completion
            # packet. Replace that cache with skips before a stale event ends.
            yield unchanged_outputs()

    def action_error(message):
        # A rejected action must not erase the task/result already displayed.
        return message, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), recent_rows()

    def run_task(label, primary, secondary, audio_value, duration, model_label, seed, cpu_offload, keep_loaded,
                 last_audio, view_state=None, auto_duration=False, random_seed=False, source_duration=False):
        ticket = begin_action(view_state)
        try:
            task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
            if task is None:
                raise ValueError("请选择任务类型")
            if model_label not in {"AuK Base（高质量推荐）", "AuK-Flash（极速）", "AuK-Flash（推荐）", "AuK Base"}:
                raise ValueError("请选择模型")
            is_flash = model_label in {"AuK-Flash（极速）", "AuK-Flash（推荐）"}
            resolved_duration, duration_mode = effective_ui_duration(
                label, primary, duration, auto_duration, source_duration, audio_value, secondary,
            )
            resolved_seed = secrets.randbelow(2**31) if bool(random_seed) else seed
            payload = {
                "request_id": str(uuid.uuid4()),
                "task_key": task.key,
                "primary": primary,
                "secondary": secondary,
                "generation_seconds": resolved_duration,
                "duration_mode": duration_mode,
                "model": "flash" if is_flash else "base",
                "seed": resolved_seed,
                "seed_mode": "random" if bool(random_seed) else "fixed",
                "cpu_offload": cpu_offload,
                "keep_loaded": keep_loaded == "keep" if isinstance(keep_loaded, str) else bool(keep_loaded),
                "nfe_steps": 4 if is_flash else 32,
                "cfg_strength": 0.0 if is_flash else 2.0,
                "client": "ui",
            }
            if task.needs_audio and audio_value is not None:
                payload["audio"] = encode_gradio_audio(audio_value)
            record, _ = manager.submit(payload)
        except Exception as exc:  # noqa: BLE001 - UI boundary reports task submission errors to the user
            if can_report(view_state, ticket):
                yield action_error(f"提交失败：{exc}")
            else:
                yield unchanged_outputs()
            return
        yield from wait_for_result(record.request_id, last_audio, view_state, ticket)

    def retry_task(request_id, last_audio, view_state=None):
        ticket = begin_action(view_state)
        try:
            record, _ = manager.retry(str(request_id or "").strip())
        except Exception as exc:  # noqa: BLE001 - UI boundary reports retry errors to the user
            if can_report(view_state, ticket):
                yield action_error(f"重试失败：{exc}")
            else:
                yield unchanged_outputs()
            return
        yield from wait_for_result(record.request_id, last_audio, view_state, ticket)

    def view_task(request_id, last_audio, view_state=None):
        ticket = begin_action(view_state)
        request_id = str(request_id or "").strip()
        try:
            manager.get(request_id)
        except (KeyError, sqlite3.Error, OSError) as exc:
            if can_report(view_state, ticket):
                yield action_error(f"读取历史任务失败：{exc}")
            else:
                yield unchanged_outputs()
            return
        yield from wait_for_result(request_id, last_audio, view_state, ticket)

    def cancel_task(request_id):
        if not request_id:
            return "当前没有可取消任务"
        try:
            state = manager.cancel(request_id)
            return f"取消请求：{state}"
        except Exception as exc:  # noqa: BLE001 - UI boundary reports cancellation errors to the user
            return f"取消失败：{exc}"

    model_diagnostics = inspect_models(paths)
    model_status = " · ".join(
        f"<strong>{item.display_name}</strong>：{'就绪' if item.status == 'ready' else '缺失/不完整'}"
        for item in model_diagnostics
    )
    with gr.Blocks(title="AuK 本地音频工作台", analytics_enabled=False) as demo:
        gr.HTML(
            "<section class='auk-header'><div class='auk-eyebrow'>AUK · LOCAL AUDIO WORKSPACE</div>"
            "<h1>AuK 本地音频工作台</h1>"
            "<div class='auk-subtitle'>语音生成、编辑、增强与分离 · 本地运行</div>"
            "<div class='auk-byline'>By <a href='https://space.bilibili.com/385085361' target='_blank' "
            "rel='noopener noreferrer'>B站 T8star-Aix</a></div>"
            "<nav class='auk-social-links' aria-label='T8star-Aix 社媒与项目链接'>"
            "<a class='auk-social-link' href='https://space.bilibili.com/385085361' target='_blank' "
            "rel='noopener noreferrer'>B站</a>"
            "<a class='auk-social-link' href='https://www.youtube.com/@T8star-Aix/' target='_blank' "
            "rel='noopener noreferrer'>YouTube</a>"
            "<a class='auk-social-link' href='https://api.seedance.nz/sign-up?aff=5f4w' target='_blank' "
            "rel='noopener noreferrer'>API</a>"
            "<a class='auk-social-link' href='https://huggingface.co/t8star/Auk-Comfy' target='_blank' "
            "rel='noopener noreferrer'>Hugging Face</a>"
            "<a class='auk-social-link' href='https://github.com/T8mars/AuK-Local' target='_blank' "
            "rel='noopener noreferrer'>GitHub</a>"
            "</nav></section>"
        )
        gr.HTML(f"<div class='auk-model-status'>{model_status}</div>")
        with gr.Row(elem_classes=["auk-updater"]):
            update_status = gr.Textbox(
                label="程序自动更新",
                value=f"当前版本 {VERSION} · 更新只替换程序文件，不修改模型、Python、任务记录和输出。",
                interactive=False,
                scale=6,
            )
            check_update_button = gr.Button("🔄 检查更新", scale=2)
            install_update_button = gr.Button("⬇ 立即更新并重启", variant="primary", interactive=False, scale=2)
        current_request = gr.State("")
        last_audio = gr.State(None)
        view_state = gr.State({"owner": None})
        with gr.Row():
            with gr.Column(scale=6):
                task_choice = gr.Dropdown(task_labels, value=task_labels[0], label="任务类型")
                task_guide = gr.HTML(task_guide_html(task_labels[0]))
                primary = gr.Textbox(
                    label=TASKS[0].primary_label,
                    lines=4,
                    value="你好，欢迎使用 AuK。",
                    placeholder=f"示例：{TASK_GUIDES[TASKS[0].key].example}",
                )
                secondary = gr.Textbox(label=TASKS[0].secondary_label, lines=3, value="自然、清晰、温暖")
                source_audio = gr.Audio(label="待处理音频", type="numpy", visible=False)
                source_audio_info = gr.HTML(source_audio_info_html(None))
                with gr.Row():
                    trim_start = gr.Number(value=0.0, minimum=0.0, label="截取开始（秒）")
                    trim_end = gr.Number(value=0.0, minimum=0.0, label="截取结束（秒；0 表示到结尾）")
                    trim_button = gr.Button("✂ 应用截取", variant="secondary")
                initial_duration = estimate_tts_seconds("你好，欢迎使用 AuK。")
                budget = gr.HTML(budget_html(task_labels[0], None, initial_duration, "你好，欢迎使用 AuK。", True))
                with gr.Row():
                    duration = gr.Slider(
                        0.2, 30.0, value=initial_duration, step=0.1,
                        label="目标时长（秒）", interactive=False,
                    )
                    model = gr.Dropdown(
                        ["AuK Base（高质量推荐）", "AuK-Flash（极速）"],
                        value="AuK Base（高质量推荐）",
                        label="模型",
                    )
                with gr.Row():
                    auto_duration = gr.Checkbox(value=True, label="自动估算 TTS 时长（避免结尾多读）")
                    source_duration = gr.Checkbox(
                        value=False,
                        label="按原音频时长",
                        visible=True,
                        interactive=False,
                    )
                with gr.Row():
                    random_seed = gr.Checkbox(value=True, label="🎲 每次使用随机 Seed（抽卡）")
                    seed = gr.Textbox(value="42", label="固定 Seed（关闭随机后生效）", max_lines=1, interactive=False)
                with gr.Row(elem_classes=["auk-actions"]):
                    run_button = gr.Button("开始生成", variant="primary")
                    cancel_button = gr.Button("取消正在查看的任务")
                with gr.Accordion("高级参数", open=False):
                    cpu_offload = gr.Checkbox(value=True, label="CPU Offload（24GB显存推荐）")
                    keep_loaded = gr.Radio(
                        [
                            ("默认常驻（后续生成无需重新加载）", "keep"),
                            ("每次生成后释放（节省显存）", "release"),
                        ],
                        value="keep",
                        label="模型驻留策略",
                    )
                    with gr.Row():
                        release_models_button = gr.Button("🧹 立即释放已加载模型", variant="secondary")
                        model_memory_status = gr.Textbox(
                            value="默认常驻；切换任务模型或 CPU Offload 设置时会按需重新加载。",
                            label="模型显存状态",
                            interactive=False,
                        )
                    instruction_preview = gr.Textbox(
                        label="最终模型指令",
                        lines=3,
                        interactive=False,
                        value=build_instruction(TASKS[0].key, "你好，欢迎使用 AuK。", "自然、清晰、温暖"),
                    )
            with gr.Column(scale=5):
                status = gr.Textbox(label="任务状态", value="服务已启动，等待任务", elem_classes=["auk-status"])
                with gr.Row():
                    result_audio = gr.Audio(label="本次结果 A", type="filepath")
                    previous_audio = gr.Audio(label="上一次结果 B", type="filepath")
                metadata = gr.Code(label="运行参数", language="json")
                gr.Markdown(f"输出目录：`{paths.outputs}`")
                with gr.Row():
                    save_result_button = gr.Button("💾 保存本次结果到下载文件夹")
                    open_folder_button = gr.Button("📂 打开输出文件夹")
                save_status = gr.Textbox(label="本机保存状态", interactive=False)
                gr.Markdown("浏览器下载被安全策略拦截时，可使用本机保存按钮；历史结果可在下方选择后保存。")
        with gr.Accordion("历史记录与失败重试", open=False):
            refresh_history = gr.Button("刷新历史", size="sm")
            gr.Markdown("单击任意历史行即可查看并播放当时的生成结果与参考/待处理音频。")
            history = gr.Dataframe(
                headers=["时间", "任务 ID", "类型", "内容摘要", "模型", "Seed", "参考/源音频", "状态", "错误"],
                value=recent_rows(),
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                retry_request = gr.Textbox(label="历史任务 ID", placeholder="从历史记录复制完整任务 ID")
                view_button = gr.Button("查看完整历史详情")
                retry_button = gr.Button("重试失败 / 取消 / 中断任务")
                cancel_history_button = gr.Button("取消选中的历史任务")
                legacy_view_button = gr.Button(visible=False)
            history_status = gr.Textbox(label="历史任务状态", interactive=False)
            with gr.Row():
                history_result_audio = gr.Audio(label="历史生成结果", type="filepath")
                history_input_audio = gr.Audio(label="当时的参考 / 待处理音频", type="numpy")
            save_history_button = gr.Button("💾 保存选中的历史结果到下载文件夹")
            history_save_status = gr.Textbox(label="历史结果保存状态", interactive=False)
            with gr.Row():
                history_primary = gr.Textbox(label="当时填写 · 主要内容", lines=3, interactive=False)
                history_secondary = gr.Textbox(
                    label="当时填写 · 附加内容", lines=3, interactive=False, visible=False,
                )
            history_instruction = gr.Textbox(label="当时发送给模型的最终指令", lines=3, interactive=False)
            history_metadata = gr.Code(label="完整任务与运行参数", language="json")
        task_choice.change(
            update_task,
            [task_choice, source_audio, duration, primary, auto_duration, source_duration, secondary],
            [primary, secondary, source_audio, auto_duration, source_duration, duration, budget, task_guide],
        )
        # Gradio's built-in waveform scissors emit `change`, not `input`.
        # Listening to change also keeps clear, replace, upload, microphone and
        # programmatic explicit trims synchronized with the value sent to AuK.
        source_audio.change(
            refresh_source_audio,
            [task_choice, primary, duration, auto_duration, source_duration, source_audio, secondary],
            [duration, budget, source_audio_info, trim_start, trim_end],
            queue=False,
        )
        trim_button.click(
            apply_source_trim,
            [
                task_choice, primary, secondary, duration, auto_duration, source_duration,
                source_audio, trim_start, trim_end,
            ],
            [source_audio, duration, budget, source_audio_info],
            queue=False,
        )
        duration.change(
            budget_html,
            [task_choice, source_audio, duration, primary, auto_duration, source_duration, secondary],
            budget,
        )
        primary.change(
            update_duration_control,
            [task_choice, primary, duration, auto_duration, source_duration, source_audio, secondary],
            [duration, budget],
        )
        secondary.change(
            update_duration_control,
            [task_choice, primary, duration, auto_duration, source_duration, source_audio, secondary],
            [duration, budget],
        )
        auto_duration.input(
            choose_auto_duration,
            [task_choice, primary, duration, auto_duration, source_duration, source_audio, secondary],
            [source_duration, duration, budget],
        )
        source_duration.input(
            choose_source_duration,
            [task_choice, primary, duration, auto_duration, source_duration, source_audio, secondary],
            [auto_duration, duration, budget],
        )
        random_seed.change(update_seed_control, random_seed, seed, queue=False)
        for component in (task_choice, primary, secondary):
            component.change(preview_instruction, [task_choice, primary, secondary], instruction_preview)
        run_button.click(
            run_task,
            [task_choice, primary, secondary, source_audio, duration, model, seed, cpu_offload, keep_loaded,
             last_audio, view_state, auto_duration, random_seed, source_duration],
            [status, result_audio, previous_audio, metadata, current_request, last_audio, history],
        )
        cancel_button.click(cancel_task, current_request, status, queue=False)
        refresh_history.click(recent_rows, outputs=history, queue=False)
        demo.load(recent_rows, outputs=history, queue=False)
        history_detail_outputs = [
            retry_request, history_status, history_result_audio, history_input_audio,
            history_primary, history_secondary, history_instruction, history_metadata,
        ]
        history.select(select_history, outputs=history_detail_outputs, queue=False)
        retry_button.click(
            retry_task,
            [retry_request, last_audio, view_state],
            [status, result_audio, previous_audio, metadata, current_request, last_audio, history],
        )
        view_button.click(history_detail, retry_request, history_detail_outputs, queue=False)
        # Preserve the old result-view callback as an internal compatibility
        # endpoint for stale browser sessions and concurrency regression tests.
        legacy_view_button.click(
            view_task,
            [retry_request, last_audio, view_state],
            [status, result_audio, previous_audio, metadata, current_request, last_audio, history],
        )
        release_models_button.click(release_loaded_models, outputs=model_memory_status, queue=False)
        save_result_button.click(save_selected_result, current_request, save_status, queue=False)
        save_history_button.click(save_selected_result, retry_request, history_save_status, queue=False)
        cancel_history_button.click(cancel_task, retry_request, history_status, queue=False)
        open_folder_button.click(open_result_folder, outputs=save_status, queue=False)
        check_update_button.click(
            check_program_update,
            outputs=[update_status, install_update_button],
            queue=False,
        )
        install_update_button.click(
            install_program_update,
            outputs=[update_status, install_update_button],
            queue=False,
        )
    return demo
