from __future__ import annotations

import asyncio
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from auk_local.audio import encode_float_audio
from auk_local.config import LocalPaths
from auk_local.manager import TaskManager
from auk_local.task_templates import (
    TASK_GUIDES,
    TASKS,
    build_instruction,
    content_scaled_seconds,
    nonverbal_duration_delta,
)
from auk_local.ui import build_ui


@pytest.fixture
def workspace(tmp_path):
    paths = LocalPaths.from_root(tmp_path)
    manager = TaskManager(paths, start_worker=False)
    demo = build_ui(manager, paths)
    functions = {entry.fn.__name__: entry.fn for entry in demo.fns.values() if entry.fn}
    try:
        yield manager, demo, functions
    finally:
        manager.close()


def run(functions, state, *, label="描述生成语音", model="AuK-Flash（推荐）", primary="test"):
    return functions["run_task"](label, primary, "", None, 1, model, "42", True, False, None, state)


def create_history(manager, *, succeeded=True):
    record, _ = manager.submit({"task_key": "instruct_tts", "primary": "history", "generation_seconds": 1})
    if not succeeded:
        manager.store.fail(record.request_id, "synthetic failure")
        return record.request_id
    manager.store.claim_next()
    directory = manager.paths.outputs / record.request_id
    directory.mkdir()
    audio = directory / "result.wav"
    metadata = directory / "metadata.json"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 48)
    metadata.write_text("{}", encoding="utf-8")
    manager.store.complete(record.request_id, str(audio), str(metadata))
    return record.request_id


def assert_ends_without_replaying(generator):
    assert next(generator) == tuple({"__type__": "update"} for _ in range(7))
    with pytest.raises(StopIteration):
        next(generator)


def test_history_view_supersedes_old_poll_without_cancelling_task(workspace):
    manager, _, functions = workspace
    history_id = create_history(manager)
    state = {"owner": None}
    old = run(functions, state)
    old_id = next(old)[4]
    shown = list(functions["view_task"](history_id, None, state))[-1]
    assert shown[4] == history_id
    assert_ends_without_replaying(old)
    assert manager.get(old_id).state == "queued"
    # Cancellation follows the selected historical task, never the stale poll.
    assert "succeeded" in functions["cancel_task"](shown[4])
    assert manager.get(old_id).state == "queued"


def test_retry_becomes_current_and_only_that_task_is_cancelled(workspace):
    manager, _, functions = workspace
    failed_id = create_history(manager, succeeded=False)
    state = {"owner": None}
    old = run(functions, state)
    old_id = next(old)[4]
    retried = functions["retry_task"](failed_id, None, state)
    new_id = next(retried)[4]
    assert_ends_without_replaying(old)
    assert "cancelled" in functions["cancel_task"](new_id)
    assert manager.get(old_id).state == "queued"
    assert list(retried)[-1][4] == new_id


def test_consecutive_submissions_do_not_emit_stale_updates(workspace):
    manager, _, functions = workspace
    state = {"owner": None}
    first = run(functions, state)
    first_id = next(first)[4]
    second = run(functions, state)
    second_id = next(second)[4]
    assert_ends_without_replaying(first)
    assert first_id != second_id
    manager.cancel(second_id)
    assert list(second)[-1][4] == second_id


@pytest.mark.parametrize("later_view_succeeds", [True, False])
def test_slow_submit_respects_later_success_but_not_later_failure(workspace, monkeypatch, later_view_succeeds):
    manager, _, functions = workspace
    history_id = create_history(manager, succeeded=False)
    state = {"owner": None}
    entered, release = threading.Event(), threading.Event()
    original = manager.submit
    pending = []
    emitted = []

    def delayed_submit(payload):
        result = original(payload)
        pending.append(result[0].request_id)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(manager, "submit", delayed_submit)
    generator = run(functions, state)

    def first_update():
        emitted.append(next(generator, None))

    thread = threading.Thread(target=first_update)
    thread.start()
    try:
        assert entered.wait(3)
        selected_id = history_id if later_view_succeeds else "not-a-task"
        list(functions["view_task"](selected_id, None, state))
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert manager.get(pending[0]).state == "queued"
    if later_view_succeeds:
        assert emitted == [tuple({"__type__": "update"} for _ in range(7))]
        with pytest.raises(StopIteration):
            next(generator)
    else:
        assert emitted[0][4] == pending[0]
        manager.cancel(pending[0])
        assert list(generator)[-1][4] == pending[0]


@pytest.mark.parametrize("simple_format", [False, True])
def test_gradio_completion_packet_does_not_replay_superseded_status_or_clear_audio(workspace, simple_format):
    from gradio.state_holder import SessionState

    manager, demo, _ = workspace
    history_id = create_history(manager)
    functions = {entry.fn.__name__: entry for entry in demo.fns.values() if entry.fn}
    state = SessionState(demo)

    async def call(name, inputs, iterator=None):
        return await demo.process_api(
            functions[name], inputs, state=state, session_hash="regression-session",
            event_id=name, iterator=iterator, simple_format=simple_format,
        )

    async def exercise():
        running = await call("run_task", [
            "描述生成语音", "test", "", None, 1, "AuK-Flash（极速）", "42", True, "release", None, None,
            True, True, False,
        ])
        assert running["is_generating"]
        viewed = await call("view_task", [history_id, None, None])
        while viewed["is_generating"]:
            viewed = await call("view_task", [], viewed["iterator"])
        assert viewed["data"][0] == "生成完成"
        assert viewed["data"][1]["path"]
        stale = await call("run_task", [], running["iterator"])
        assert stale["is_generating"]  # prime Gradio's final-packet cache with skips
        completed = await call("run_task", [], stale["iterator"])
        assert not completed["is_generating"]
        for index in (0, 1, 2, 3, 6):
            assert completed["data"][index] == {"__type__": "update"}
        current_request = functions["run_task"].outputs[4]
        assert state[current_request._id] == history_id

    asyncio.run(exercise())


def test_actual_gradio_states_are_isolated_between_sessions(workspace):
    from gradio.state_holder import SessionState

    manager, demo, functions = workspace
    block = next(block for block in demo.blocks.values() if getattr(block, "value", None) == {"owner": None})
    session_a, session_b = SessionState(demo), SessionState(demo)
    first = run(functions, session_a[block._id])
    first_id = next(first)[4]
    second = run(functions, session_b[block._id])
    second_id = next(second)[4]
    assert session_a[block._id] is not session_b[block._id]
    manager.cancel(first_id)
    assert list(first)[-1][4] == first_id
    manager.cancel(second_id)
    assert list(second)[-1][4] == second_id


@pytest.mark.parametrize("action", ["run", "retry", "view"])
def test_rejected_action_preserves_display_and_current_task(workspace, action):
    manager, _, functions = workspace
    state = {"owner": None}
    active = run(functions, state)
    active_id = next(active)[4]
    owner = state["owner"]
    if action == "run":
        rejected = run(functions, state, primary="")
    else:
        rejected = functions[f"{action}_task"]("not-a-task", None, state)
    update = next(rejected)
    for value in update[1:6]:
        assert value == {"__type__": "update"}
    assert state["owner"] == owner
    manager.cancel(active_id)
    assert list(active)[-1][4] == active_id


@pytest.mark.parametrize("label,model", [(None, "AuK-Flash（推荐）"), ("描述生成语音", None)])
def test_cleared_dropdown_returns_actionable_error(workspace, label, model):
    manager, _, functions = workspace
    updates = list(run(functions, {"owner": None}, label=label, model=model))
    assert "请选择" in updates[0][0]
    assert not manager.store.list_recent()


def test_cleared_task_and_invalid_duration_do_not_break_preview(workspace):
    _, _, functions = workspace
    assert "请选择" in functions["preview_instruction"](None, "text", "")
    assert "请选择" in functions["budget_html"](None, None, 1)
    assert "无效" in functions["budget_html"]("描述生成语音", None, float("nan"))
    assert "无效" in functions["budget_html"]("参考声音克隆", (0, [1]), 1)
    updates = functions["update_task"](None, None, 1)
    assert all(update == {"__type__": "update"} for update in updates[:3])


@pytest.mark.parametrize(
    ("task_key", "primary", "secondary", "expected"),
    [
        ("instruct_tts", "欢迎回来", "温柔女声", '请基于下面的描述: "温柔女声",生成语音内容"欢迎回来".'),
        ("zero_shot_tts", "欢迎回来", "不应发送", 'Say the following with the same voice: "欢迎回来"'),
        ("content_edit", "夜色那么美改成白天那么美", "不应发送", "把‘夜色那么美’改成‘白天那么美’"),
        ("lyric_edit", "把歌词‘明天你好’改成‘未来你好’", "不应发送", "把这段歌词中的“明天你好”改成“未来你好”。"),
        ("pitch", "+1", "不应发送", "将音调升高1个半音。"),
        ("speed", "0.75", "不应发送", "将语速调整为0.75倍。"),
        ("volume", "-10", "不应发送", "将音量降低10分贝。"),
        ("emotion", "开心", "不应发送", "将情感转变为开心。"),
        ("timbre", "低沉磁性的年轻男声", "不应发送", "请将这段音频的音色修改为符合以下描述的声音：“低沉磁性的年轻男声”。"),
        ("deaccent", "去掉方言口音", "不应发送", "请去掉这段语音里的方言口音，保持说话人音色一致。"),
        ("nonverbal", "在“欢迎回来”后增加笑声", "不应发送", "在“欢迎回来”后增加笑声。"),
        ("whisper", "转换成耳语", "不应发送", "用小声耳语的方式把这段话说出来。"),
        ("enhance", "去噪并去除房间混响", "不应发送", "请对这段语音做纯净化处理，保留所有说话人的人声，并去除其中的噪声和混响，输出与输入等长的干净人声。"),
        ("quality", "去掉电话感", "不应发送", "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech."),
        ("speech_separate", "第一个开始说话的人", "不应发送", "Keep only the first speaker"),
        ("music_separate", "只保留歌声，去掉说话和伴奏", "不应发送", "请只保留歌声，其余声音都去掉。"),
        ("target_speaker", "欢迎大家来到今天的节目", "不应发送", "Keep only the speaker who says “欢迎大家来到今天的节目”"),
    ],
)
def test_each_task_builds_an_official_model_instruction(task_key, primary, secondary, expected):
    assert build_instruction(task_key, primary, secondary) == expected


def test_every_task_has_visible_official_usage_guide(workspace):
    _, _, functions = workspace
    assert set(TASK_GUIDES) == {task.key for task in TASKS}
    for task in TASKS:
        rendered = functions["update_task"](task.label, None, 1)[-1]
        assert "官方用法" in rendered
        assert TASK_GUIDES[task.key].example in rendered


@pytest.mark.parametrize(
    ("task_key", "raw", "expected"),
    [
        ("content_edit", "夜色那么美改成白天那么美", "把‘夜色那么美’改成‘白天那么美’"),
        ("content_edit", "把“今天下午开会”改成“明天上午开会”", "把‘今天下午开会’改成‘明天上午开会’"),
        ("lyric_edit", "把歌词‘明天你好’改成‘未来你好’", "把这段歌词中的“明天你好”改成“未来你好”。"),
    ],
)
def test_replacement_requests_are_normalized_to_official_templates(task_key, raw, expected):
    assert build_instruction(task_key, raw, "整段原文不应发送") == expected


def test_nonverbal_quality_and_replacement_inputs_are_strictly_canonicalized():
    assert build_instruction("nonverbal", "在开头增加笑声") == "在语音开头增加笑声。"
    assert build_instruction("nonverbal", "删除全部呼吸") == "删除音频中所有的呼吸声。"
    with pytest.raises(ValueError, match="未识别非语言声音"):
        build_instruction("nonverbal", "在开头增加火车声")
    assert build_instruction("quality", "去掉电话感") == (
        "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech."
    )
    assert build_instruction("content_edit", "Replace 'old words' with 'new words'.") == (
        "Replace 'old words' with 'new words'."
    )


def test_content_edit_supports_all_official_operation_shapes():
    assert build_instruction("content_edit", "在“你好”后面加上“呀”") == "在‘你好’后面加上‘呀’"
    assert build_instruction("content_edit", "删掉“那个”") == "删掉‘那个’"
    assert build_instruction("content_edit", "删掉“谢谢”后面那个“再见”") == "删掉‘谢谢’后的‘再见’"
    with pytest.raises(ValueError, match="一次只改一处"):
        build_instruction("content_edit", "帮我改一下")


def test_official_content_and_nonverbal_duration_rules():
    assert content_scaled_seconds(
        "content_edit", "把“今天”改成“明天上午”", 10.0, "我们今天开会",
    ) == pytest.approx(13.3333333333)
    assert content_scaled_seconds("lyric_edit", "把歌词“今天”改成“明天上午”", 10.0) == pytest.approx(20.0)
    assert nonverbal_duration_delta("在开头增加呼吸声") == pytest.approx(0.35)
    assert nonverbal_duration_delta("删除所有笑声") == pytest.approx(-1.05)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("sad", "Say this in a sad tone"), ("fearful", "Say this in a afraid tone")],
)
def test_english_emotion_uses_official_demo_wording(value, expected):
    assert build_instruction("emotion", value) == expected


def test_reference_transcript_is_not_sent_as_clone_instruction(workspace):
    import numpy as np

    manager, _, functions = workspace
    updates = functions["run_task"](
        "参考声音克隆", "target text", "reference transcript", (24000, np.zeros(2400, dtype=np.float32)),
        1, "AuK-Flash（推荐）", "42", True, False, None,
    )
    request_id = next(updates)[4]
    assert manager.get(request_id).request["instruction"] == 'Say the following with the same voice: "target text"'
    manager.cancel(request_id)
    assert list(updates)[-1][4] == request_id


def test_ui_auto_duration_and_random_seed_are_saved(workspace, monkeypatch):
    import auk_local.ui

    manager, _, functions = workspace
    monkeypatch.setattr(auk_local.ui.secrets, "randbelow", lambda _limit: 123456789)
    updates = functions["run_task"](
        "描述生成语音", "一只小猫在叫啊", "自然、清晰、温暖", None,
        3.0, "AuK-Flash（推荐）", "42", True, False, None, None, True, True,
    )
    request_id = next(updates)[4]
    request = manager.get(request_id).request
    assert request["generation_seconds"] == 1.7
    assert request["duration_mode"] == "auto"
    assert request["seed"] == 123456789
    assert request["seed_mode"] == "random"
    manager.cancel(request_id)
    list(updates)


def test_source_audio_duration_checkbox_locks_and_saves_exact_audio_length(workspace):
    import numpy as np

    manager, _, functions = workspace
    audio = (48000, np.zeros(587520, dtype=np.float32))
    duration_update, budget = functions["update_duration_control"](
        "语音文字编辑", "删掉“那个”", 3.0, False, True, audio,
    )
    assert duration_update["value"] == pytest.approx(12.24)
    assert duration_update["interactive"] is False
    assert "按原音频时长" in budget
    updates = functions["run_task"](
        "语音文字编辑", "删掉“那个”", "", audio, 3.0,
        "AuK-Flash（推荐）", "42", True, False, None, None, False, False, True,
    )
    request_id = next(updates)[4]
    request = manager.get(request_id).request
    assert request["generation_seconds"] == pytest.approx(12.24)
    assert request["duration_mode"] == "source"
    manager.cancel(request_id)
    list(updates)


def test_duration_modes_are_mutually_exclusive(workspace):
    import numpy as np

    _, _, functions = workspace
    audio = (24000, np.zeros(24000, dtype=np.float32))
    source_update, duration_update, _ = functions["choose_auto_duration"](
        "参考声音克隆", "你好", 3.0, True, True, audio,
    )
    assert source_update["value"] is False
    assert duration_update["interactive"] is False
    auto_update, duration_update, _ = functions["choose_source_duration"](
        "参考声音克隆", "你好", 3.0, True, True, audio,
    )
    assert auto_update["value"] is False
    assert duration_update["value"] == pytest.approx(1.0)


def test_speed_edit_forces_source_scaled_duration_and_ignores_manual_target(workspace):
    import numpy as np

    manager, _, functions = workspace
    audio = (48000, np.zeros(460800, dtype=np.float32))  # 9.6 seconds
    duration_update, budget = functions["update_duration_control"](
        "速度编辑", "1.5", 12.2, False, False, audio,
    )
    assert duration_update["value"] == pytest.approx(6.4)
    assert duration_update["interactive"] is False
    assert "速度倍率自动计算" in budget
    updates = functions["run_task"](
        "速度编辑", "1.5", "", audio, 12.2,
        "AuK-Flash（推荐）", "42", True, False, None, None, False, False, False,
    )
    request_id = next(updates)[4]
    request = manager.get(request_id).request
    assert request["instruction"] == "将语速调整为1.5倍。"
    assert request["generation_seconds"] == pytest.approx(6.4)
    assert request["duration_mode"] == "speed"
    manager.cancel(request_id)
    list(updates)


def test_task_switch_exposes_source_length_option_and_labels_speed_duration(workspace):
    _, _, functions = workspace
    content_updates = functions["update_task"]("语音文字编辑", None, 3.0, "删掉“那个”", False, False)
    assert content_updates[4]["visible"] is True
    assert content_updates[4]["interactive"] is True
    speed_updates = functions["update_task"]("速度编辑", None, 3.0, "1.5", False, False)
    assert speed_updates[4]["interactive"] is False
    assert speed_updates[5]["label"] == "自动输出时长（原音频时长 ÷ 速度倍率）"


def test_emotion_task_uses_official_duration_multiplier(workspace):
    import numpy as np

    manager, _, functions = workspace
    audio = (48000, np.zeros(587520, dtype=np.float32))
    duration_update, budget = functions["update_duration_control"](
        "情绪编辑", "悲伤", 10.9, False, False, audio,
    )
    assert duration_update["value"] == pytest.approx(14.9328)
    assert duration_update["interactive"] is False
    assert "官方情绪系数自动计算" in budget
    updates = functions["run_task"](
        "情绪编辑", "悲伤", "", audio, 10.9,
        "AuK-Flash（推荐）", "42", True, False, None, None, False, False, False,
    )
    request_id = next(updates)[4]
    request = manager.get(request_id).request
    assert request["instruction"] == "将情感转变为悲伤。"
    assert request["generation_seconds"] == pytest.approx(14.9328)
    assert request["duration_mode"] == "emotion"
    manager.cancel(request_id)
    list(updates)


def test_content_and_nonverbal_tasks_lock_official_automatic_duration(workspace):
    import numpy as np

    _, _, functions = workspace
    audio = (24_000, np.zeros(240_000, dtype=np.float32))
    content_update, content_budget = functions["update_duration_control"](
        "语音文字编辑", "把“今天”改成“明天上午”", 48.0, False, False, audio, "我们今天开会",
    )
    assert content_update["value"] == pytest.approx(13.3333333333)
    assert content_update["interactive"] is False
    assert "文字变化" in content_budget
    nonverbal_update, nonverbal_budget = functions["update_duration_control"](
        "非语言声音编辑", "在开头增加笑声", 48.0, False, False, audio,
    )
    assert nonverbal_update["value"] == pytest.approx(10.75)
    assert nonverbal_update["interactive"] is False
    assert "声音增删" in nonverbal_budget


def test_explicit_trim_changes_the_audio_value_used_for_budget(workspace):
    import numpy as np

    _, _, functions = workspace
    audio = (44_100, np.zeros(44_100 * 48, dtype=np.float32))
    clipped, duration_update, budget, info = functions["apply_source_trim"](
        "去口音", "去掉方言口音", "", 48.0, False, False, audio, 22.0, 26.0,
    )
    assert clipped[1].shape[0] == 44_100 * 4
    assert duration_update["value"] == pytest.approx(4.0)
    assert "输入 4.00s · 输出 4.00s" in budget
    assert "实际提交输入：4.00 秒" in info


def test_source_audio_help_explains_native_restore_button(workspace):
    import numpy as np

    _, _, functions = workspace
    info = functions["refresh_source_audio"](
        "去口音", "去掉方言口音", 1.0, False, False,
        (24_000, np.zeros(24_000, dtype=np.float32)), "",
    )[2]
    assert "↶ 可恢复最初上传音频" in info


def test_model_residency_defaults_to_keep_and_maps_radio_values(workspace):
    manager, demo, functions = workspace
    config = demo.get_config_file()
    residence = next(
        component["props"]
        for component in config["components"]
        if component.get("props", {}).get("label") == "模型驻留策略"
    )
    assert residence["value"] == "keep"

    for choice, expected in (("keep", True), ("release", False)):
        updates = functions["run_task"](
            "描述生成语音", "模型常驻测试", "", None, 1.0,
            "AuK-Flash（推荐）", "42", True, choice, None, {"owner": None},
            False, False, False,
        )
        request_id = next(updates)[4]
        assert manager.get(request_id).request["keep_loaded"] is expected
        manager.cancel(request_id)
        list(updates)


def test_history_detail_restores_result_input_prompts_and_instruction(workspace):
    import json
    import numpy as np

    manager, _, functions = workspace
    samples = np.linspace(-0.25, 0.25, 8_000, dtype=np.float32)
    record, _ = manager.submit({
        "task_key": "zero_shot_tts",
        "primary": "历史目标文字",
        "secondary": "历史参考音频文字",
        "generation_seconds": 1.0,
        "audio": encode_float_audio(samples, 8_000),
    })
    manager.store.claim_next()
    directory = manager.paths.outputs / record.request_id
    directory.mkdir()
    result = directory / "result.wav"
    metadata = directory / "metadata.json"
    with wave.open(str(result), "wb") as stream:
        stream.setparams((1, 2, 8_000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 16_000)
    metadata.write_text('{"seed":42}', encoding="utf-8")
    manager.store.complete(record.request_id, str(result), str(metadata))

    detail = functions["history_detail"](record.request_id)
    assert detail[0] == record.request_id
    assert "成功" in detail[1] and "参考/源音频可播放" in detail[1]
    assert detail[2] == str(result)
    assert detail[3][0] == 8_000
    np.testing.assert_allclose(detail[3][1], samples)
    assert detail[4]["value"] == "历史目标文字"
    assert detail[5]["value"] == "历史参考音频文字"
    assert detail[6] == manager.get(record.request_id).request["instruction"]
    payload = json.loads(detail[7])
    assert payload["metadata"]["seed"] == 42
    assert payload["input_available"] is True

    row = functions["recent_rows"]()[0]
    selected = functions["select_history"](SimpleNamespace(row_value=tuple(row)))
    assert selected[0] == record.request_id


def test_history_selection_is_injected_by_actual_gradio_process_api(workspace):
    from gradio.events import EventData

    manager, demo, functions = workspace
    history_id = create_history(manager)
    row = functions["recent_rows"]()[0]
    entry = next(entry for entry in demo.fns.values() if entry.fn and entry.fn.__name__ == "select_history")
    assert entry.collects_event_data
    event = EventData(None, {"index": [0, 0], "value": row[0], "row_value": row, "selected": True})
    response = asyncio.run(demo.process_api(entry, [], event_data=event))
    assert response["data"][0] == history_id
    assert response["data"][2]["path"]
    assert response["data"][4]["value"] == "history"


def test_history_result_saves_from_original_output(workspace, tmp_path, monkeypatch):
    manager, _, functions = workspace
    history_id = create_history(manager)
    destination = tmp_path / "downloads"
    monkeypatch.setattr("auk_local.result_files.downloads_directory", lambda: destination)
    assert "请先" in functions["save_selected_result"]("")
    assert "SHA-256" in functions["save_selected_result"](history_id)
    saved = next(destination.glob("*.wav"))
    assert saved.read_bytes() == Path(manager.get(history_id).result_path).read_bytes()


def test_damaged_history_result_does_not_break_detail_event(workspace):
    manager, demo, _ = workspace
    history_id = create_history(manager)
    Path(manager.get(history_id).result_path).write_bytes(b"broken wav")
    entry = next(entry for entry in demo.fns.values() if entry.fn and entry.fn.__name__ == "history_detail")
    response = asyncio.run(demo.process_api(entry, [history_id]))
    assert response["data"][2] is None
    assert "生成结果损坏" in response["data"][1]
    assert response["data"][4]["value"] == "history"


def test_waveform_scissors_use_change_event_and_reset_explicit_trim(workspace):
    _, demo, _ = workspace
    config = demo.get_config_file()
    audio_id = next(
        component["id"]
        for component in config["components"]
        if component.get("props", {}).get("label") == "待处理音频"
    )
    dependencies = [
        dependency
        for dependency in config["dependencies"]
        if any(target == (audio_id, "change") or target == [audio_id, "change"] for target in dependency["targets"])
    ]
    assert len(dependencies) == 1
    assert dependencies[0]["api_name"].startswith("refresh_source_audio")
    assert len(dependencies[0]["outputs"]) == 5
    assert not any(
        any(target == (audio_id, "input") or target == [audio_id, "input"] for target in dependency["targets"])
        for dependency in config["dependencies"]
    )


def test_automatic_duration_over_30_does_not_break_slider_outputs(workspace):
    import numpy as np

    _, _, functions = workspace
    audio = (24_000, np.zeros(24_000 * 24, dtype=np.float32))
    duration_update, budget = functions["update_duration_control"](
        "语音文字编辑",
        f"把“瓜”改成“{'T8' * 100}”",
        3.0,
        False,
        False,
        audio,
        "",
    )
    assert duration_update["value"] == 30.0
    assert duration_update["interactive"] is False
    assert "已超出限制" in budget
