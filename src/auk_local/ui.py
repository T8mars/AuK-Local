from __future__ import annotations

import json
import math
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .audio import encode_gradio_audio
from .duration import TTS_TASK_KEYS, estimate_tts_seconds
from .manager import TaskManager
from .task_templates import TASK_BY_KEY, TASK_BY_LABEL, TASKS, build_instruction
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

    def effective_ui_duration(label, primary, duration, auto_duration):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        if task is not None and task.key in TTS_TASK_KEYS and bool(auto_duration):
            return estimate_tts_seconds(primary), True
        return float(duration), False

    def budget_html(label, audio_value, duration, primary="", auto_duration=False):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        if task is None:
            return "<div class='auk-budget'>请选择任务类型。</div>"
        source_seconds = 0.0
        try:
            if task.needs_audio and audio_value is not None:
                sample_rate, samples = audio_value
                sample_rate = float(sample_rate)
                if not math.isfinite(sample_rate) or sample_rate <= 0 or len(samples) == 0:
                    raise ValueError("invalid audio")
                source_seconds = len(samples) / sample_rate
            target_seconds, estimated = effective_ui_duration(label, primary, duration, auto_duration)
            if not math.isfinite(target_seconds) or target_seconds <= 0:
                raise ValueError("invalid duration")
        except (TypeError, ValueError, OverflowError):
            return "<div class='auk-budget'>输入数据无效，请检查音频和目标时长。</div>"
        total = source_seconds + target_seconds
        remaining = max(0.0, 30.0 - total)
        state = "可提交" if total <= 30.0 + 1e-9 else "已超出限制"
        mode = "自动估算" if estimated else "当前设置"
        return (
            f"<div class='auk-budget'>{mode} · 30 秒预算："
            f"输入 {source_seconds:.2f}s + 输出 {target_seconds:.2f}s = {total:.2f}s"
            f" · 剩余 {remaining:.2f}s · {state}</div>"
        )

    def update_duration_control(label, primary, duration, auto_duration, audio_value):
        try:
            target_seconds, estimated = effective_ui_duration(label, primary, duration, auto_duration)
            update = gr.update(value=target_seconds, interactive=not estimated)
        except (TypeError, ValueError, OverflowError):
            update = gr.update(interactive=not bool(auto_duration))
        return update, budget_html(label, audio_value, duration, primary, auto_duration)

    def update_seed_control(random_seed):
        return gr.update(interactive=not bool(random_seed))

    def update_task(label, audio_value, duration, primary="", auto_duration=False):
        task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
        if task is None:
            return gr.skip(), gr.skip(), gr.skip(), budget_html(label, audio_value, duration, primary, auto_duration)
        return (
            gr.update(label=task.primary_label),
            gr.update(label=task.secondary_label),
            gr.update(label="参考声音" if task.key == "zero_shot_tts" else "待处理音频", visible=task.needs_audio),
            budget_html(label, audio_value, duration, primary, auto_duration),
        )

    def preview_instruction(label, primary, secondary):
        if label not in TASK_BY_LABEL:
            return "请选择任务类型。"
        if not str(primary or "").strip():
            return "请先填写主要内容；最终模型指令会在这里预览。"
        return build_instruction(TASK_BY_LABEL[label].key, primary, secondary)

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
            records = manager.store.list_recent(20)
        except (sqlite3.Error, OSError):
            return []
        for record in records:
            task = TASK_BY_KEY.get(str(record.request.get("task_key")))
            rows.append(
                [
                    time.strftime("%m-%d %H:%M:%S", time.localtime(record.created_at)),
                    record.request_id,
                    task.label if task else record.request.get("task_key", ""),
                    record.request.get("model", ""),
                    record.request.get("seed", ""),
                    state_labels.get(record.state, record.state),
                    record.error or "",
                ]
            )
        return rows

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
                 last_audio, view_state=None, auto_duration=False, random_seed=False):
        ticket = begin_action(view_state)
        try:
            task = TASK_BY_LABEL.get(label) if isinstance(label, str) else None
            if task is None:
                raise ValueError("请选择任务类型")
            if model_label not in {"AuK-Flash（推荐）", "AuK Base"}:
                raise ValueError("请选择模型")
            is_flash = model_label == "AuK-Flash（推荐）"
            resolved_duration, duration_is_auto = effective_ui_duration(label, primary, duration, auto_duration)
            resolved_seed = secrets.randbelow(2**31) if bool(random_seed) else seed
            payload = {
                "request_id": str(uuid.uuid4()),
                "task_key": task.key,
                "primary": primary,
                "secondary": secondary,
                "generation_seconds": resolved_duration,
                "duration_mode": "auto" if duration_is_auto else "manual",
                "model": "flash" if is_flash else "base",
                "seed": resolved_seed,
                "seed_mode": "random" if bool(random_seed) else "fixed",
                "cpu_offload": cpu_offload,
                "keep_loaded": keep_loaded,
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
            "<h1>AuK 本地音频工作台</h1><div class='auk-subtitle'>语音生成、编辑、增强与分离 · 本地运行</div></section>"
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
                primary = gr.Textbox(label=TASKS[0].primary_label, lines=4, value="你好，欢迎使用 AuK。")
                secondary = gr.Textbox(label=TASKS[0].secondary_label, lines=3, value="自然、清晰、温暖")
                source_audio = gr.Audio(label="待处理音频", type="numpy", visible=False)
                initial_duration = estimate_tts_seconds("你好，欢迎使用 AuK。")
                budget = gr.HTML(budget_html(task_labels[0], None, initial_duration, "你好，欢迎使用 AuK。", True))
                with gr.Row():
                    duration = gr.Slider(
                        0.2, 30.0, value=initial_duration, step=0.1,
                        label="目标时长（秒）", interactive=False,
                    )
                    model = gr.Dropdown(["AuK-Flash（推荐）", "AuK Base"], value="AuK-Flash（推荐）", label="模型")
                with gr.Row():
                    auto_duration = gr.Checkbox(value=True, label="自动估算 TTS 时长（避免结尾多读）")
                    random_seed = gr.Checkbox(value=True, label="🎲 每次使用随机 Seed（抽卡）")
                    seed = gr.Textbox(value="42", label="固定 Seed（关闭随机后生效）", max_lines=1, interactive=False)
                with gr.Row(elem_classes=["auk-actions"]):
                    run_button = gr.Button("开始生成", variant="primary")
                    cancel_button = gr.Button("取消正在查看的任务")
                with gr.Accordion("高级参数", open=False):
                    cpu_offload = gr.Checkbox(value=True, label="CPU Offload（24GB显存推荐）")
                    keep_loaded = gr.Checkbox(value=False, label="生成后保持模型驻留")
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
        with gr.Accordion("历史记录与失败重试", open=False):
            refresh_history = gr.Button("刷新历史", size="sm")
            history = gr.Dataframe(
                headers=["时间", "任务 ID", "类型", "模型", "Seed", "状态", "错误"],
                value=recent_rows(),
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                retry_request = gr.Textbox(label="历史任务 ID", placeholder="从历史记录复制完整任务 ID")
                view_button = gr.Button("查看结果 / 状态")
                retry_button = gr.Button("重试失败 / 取消 / 中断任务")
        task_choice.change(
            update_task,
            [task_choice, source_audio, duration, primary, auto_duration],
            [primary, secondary, source_audio, budget],
        )
        source_audio.change(budget_html, [task_choice, source_audio, duration, primary, auto_duration], budget)
        duration.change(budget_html, [task_choice, source_audio, duration, primary, auto_duration], budget)
        task_choice.change(
            update_duration_control,
            [task_choice, primary, duration, auto_duration, source_audio],
            [duration, budget],
        )
        primary.change(
            update_duration_control,
            [task_choice, primary, duration, auto_duration, source_audio],
            [duration, budget],
        )
        auto_duration.change(
            update_duration_control,
            [task_choice, primary, duration, auto_duration, source_audio],
            [duration, budget],
        )
        random_seed.change(update_seed_control, random_seed, seed, queue=False)
        for component in (task_choice, primary, secondary):
            component.change(preview_instruction, [task_choice, primary, secondary], instruction_preview)
        run_button.click(
            run_task,
            [task_choice, primary, secondary, source_audio, duration, model, seed, cpu_offload, keep_loaded,
             last_audio, view_state, auto_duration, random_seed],
            [status, result_audio, previous_audio, metadata, current_request, last_audio, history],
        )
        cancel_button.click(cancel_task, current_request, status, queue=False)
        refresh_history.click(recent_rows, outputs=history, queue=False)
        demo.load(recent_rows, outputs=history, queue=False)
        retry_button.click(
            retry_task,
            [retry_request, last_audio, view_state],
            [status, result_audio, previous_audio, metadata, current_request, last_audio, history],
        )
        view_button.click(
            view_task,
            [retry_request, last_audio, view_state],
            [status, result_audio, previous_audio, metadata, current_request, last_audio, history],
        )
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
