from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from .config import LocalPaths, configure_bundled_tools
from .diagnostics import runtime_diagnostic
from .manager import TaskManager
from .version import PROTOCOL_VERSION, VERSION


ACTIVE_TASK_STATES = {"queued", "loading", "encoding", "sampling", "decoding", "saving", "cancelling"}


class RestartController:
    def __init__(self) -> None:
        self.server = None
        self.exit_code = 0

    def attach(self, server) -> None:
        self.server = server

    def request_update_restart(self) -> None:
        if os.environ.get("AUK_LAUNCHED_BY_EXE") != "1":
            raise RuntimeError("一键更新需要通过根目录的 AuK-Local.exe 启动程序")
        if self.server is None:
            raise RuntimeError("服务尚未连接启动器，无法安全重启")
        self.exit_code = 42

        def stop_after_response() -> None:
            self.server.should_exit = True

        timer = threading.Timer(1.25, stop_after_response)
        timer.daemon = True
        timer.start()


def load_or_create_token(paths: LocalPaths) -> str:
    token_path = paths.data / "session-token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if len(token) >= 32:
            return token
    token = secrets.token_urlsafe(32)
    token_path.write_text(token, encoding="utf-8")
    return token


def create_app(paths: LocalPaths, *, with_ui: bool = True, manager: TaskManager | None = None):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    configure_bundled_tools(paths)
    task_manager = manager or TaskManager(paths)
    from .updater import UpdateError, UpdateManager

    update_manager = UpdateManager(paths)
    restart_controller = RestartController()
    token = load_or_create_token(paths)
    instance_id = hashlib.sha256(token.encode("utf-8")).hexdigest()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        task_manager.close()

    app = FastAPI(title="AuK Local", version=VERSION, lifespan=lifespan)
    app.state.task_manager = task_manager
    app.state.update_manager = update_manager
    app.state.restart_controller = restart_controller

    def ensure_no_active_tasks() -> None:
        if any(record.state in ACTIVE_TASK_STATES for record in task_manager.store.list_recent(100)):
            raise UpdateError("仍有任务在排队或运行，请等待任务结束后再更新")

    @app.exception_handler(sqlite3.Error)
    @app.exception_handler(OSError)
    async def storage_error(_request, exc):
        return JSONResponse(status_code=503, content={"detail": f"本地存储暂不可用：{exc}"})

    def authorize(x_auk_token: str | None = Header(default=None)) -> None:
        if not x_auk_token or not secrets.compare_digest(x_auk_token, token):
            raise HTTPException(status_code=401, detail="本机服务令牌无效")

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        scheduler = task_manager.scheduler_health
        return {
            "status": "ok" if scheduler["accepting_tasks"] else "degraded",
            "version": VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "instance_id": instance_id,
            "ui_enabled": with_ui,
            "scheduler": scheduler,
            "inference_ready": scheduler["accepting_tasks"] and all(
                model["status"] == "ready" for model in runtime_diagnostic(paths, probe_torch=False)["models"]
            ),
        }

    @app.get("/api/v1/diagnostics", dependencies=[Depends(authorize)])
    def diagnostics() -> dict[str, Any]:
        return runtime_diagnostic(paths)

    @app.get("/api/v1/update/check", dependencies=[Depends(authorize)])
    def check_update() -> dict[str, Any]:
        try:
            result = update_manager.check()
        except UpdateError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {key: value for key, value in result.items() if key not in {"payload", "asset_url"}}

    @app.post("/api/v1/update/install", dependencies=[Depends(authorize)])
    def install_update() -> dict[str, Any]:
        try:
            ensure_no_active_tasks()
            result = update_manager.stage_latest()
            restart_controller.request_update_restart()
            return result
        except (UpdateError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/tasks", dependencies=[Depends(authorize)])
    def submit_task(payload: dict[str, Any]):
        try:
            record, created = task_manager.submit(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {**record.public_dict(), "created": created}

    @app.get("/api/v1/tasks/{request_id}", dependencies=[Depends(authorize)])
    def task_status(request_id: str):
        try:
            return {**task_manager.get(request_id).public_dict(), "scheduler": task_manager.scheduler_health}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc

    @app.post("/api/v1/tasks/{request_id}/cancel", dependencies=[Depends(authorize)])
    def cancel_task(request_id: str):
        try:
            return {"request_id": request_id, "state": task_manager.cancel(request_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/v1/tasks/{request_id}/retry", dependencies=[Depends(authorize)])
    def retry_task(request_id: str):
        try:
            record, created = task_manager.retry(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {**record.public_dict(), "created": created}

    @app.get("/api/v1/tasks", dependencies=[Depends(authorize)])
    def recent_tasks(limit: int = 20):
        limit = max(1, min(int(limit), 100))
        return [
            {
                **record.public_dict(),
                "task_key": record.request.get("task_key"),
                "model": record.request.get("model"),
                "generation_seconds": record.request.get("generation_seconds"),
                "seed": record.request.get("seed"),
            }
            for record in task_manager.store.list_recent(limit)
        ]

    @app.get("/api/v1/tasks/{request_id}/audio", dependencies=[Depends(authorize)])
    def task_audio(request_id: str):
        try:
            record = task_manager.get(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        if record.state != "succeeded" or not record.result_path or not Path(record.result_path).is_file():
            raise HTTPException(status_code=410, detail="任务结果尚不可用或已清理")
        return FileResponse(record.result_path, media_type="audio/wav", filename=f"auk-{request_id}.wav")

    @app.get("/api/v1/tasks/{request_id}/metadata", dependencies=[Depends(authorize)])
    def task_metadata(request_id: str):
        try:
            record = task_manager.get(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="任务不存在") from exc
        if record.state != "succeeded" or not record.metadata_path or not Path(record.metadata_path).is_file():
            raise HTTPException(status_code=410, detail="任务参数尚不可用或已清理")
        import json

        try:
            return json.loads(Path(record.metadata_path).read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(status_code=410, detail="任务参数文件已损坏，请检查输出目录") from exc

    if with_ui:
        import gradio as gr

        from .ui import CSS, build_ui

        theme = gr.themes.Soft(primary_hue="pink", secondary_hue="blue", neutral_hue="slate")
        app = gr.mount_gradio_app(
            app,
            build_ui(
                task_manager,
                paths,
                update_manager=update_manager,
                request_update_restart=restart_controller.request_update_restart,
                ensure_no_active_tasks=ensure_no_active_tasks,
            ),
            path="/",
            show_error=True,
            theme=theme,
            css=CSS,
        )
        app.state.task_manager = task_manager
        app.state.update_manager = update_manager
        app.state.restart_controller = restart_controller
    return app
