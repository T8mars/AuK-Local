from __future__ import annotations

import asyncio
import threading
import wave

import pytest

from auk_local.config import LocalPaths
from auk_local.manager import TaskManager
from auk_local.task_templates import TASKS, build_instruction
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
            "描述生成语音", "test", "", None, 1, "AuK-Flash（推荐）", "42", True, False, None, None,
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


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.key)
def test_each_template_keeps_both_user_text_fields(task):
    instruction = build_instruction(task.key, "primary-marker", "secondary-marker")
    assert "primary-marker" in instruction
    assert "secondary-marker" in instruction
    if task.key == "zero_shot_tts":
        assert '参考音频的文字内容为："secondary-marker"' in instruction


def test_reference_transcript_reaches_saved_clone_instruction(workspace):
    import numpy as np

    manager, _, functions = workspace
    updates = functions["run_task"](
        "参考声音克隆", "target text", "reference transcript", (24000, np.zeros(2400, dtype=np.float32)),
        1, "AuK-Flash（推荐）", "42", True, False, None,
    )
    request_id = next(updates)[4]
    assert "reference transcript" in manager.get(request_id).request["instruction"]
    manager.cancel(request_id)
    assert list(updates)[-1][4] == request_id
