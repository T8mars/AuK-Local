from __future__ import annotations

import argparse
import json
import os
import webbrowser
from pathlib import Path

from .config import LocalPaths
from .diagnostics import runtime_diagnostic


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AuK Windows 本地整合服务")
    parser.add_argument("--root", type=Path, default=None, help="整合包根目录")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="启动本机服务和浅色界面")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    serve.add_argument("--no-ui", action="store_true")
    serve.add_argument("--open-browser", action="store_true")
    diagnose = sub.add_parser("diagnose", help="输出环境和模型诊断")
    diagnose.add_argument("--no-torch", action="store_true")
    diagnose.add_argument("--verify-hashes", action="store_true", help="逐文件执行完整 SHA-256 校验（较慢）")
    download = sub.add_parser("download-models", help="下载固定版本的全部模型")
    download.add_argument("models", nargs="*", metavar="MODEL", help="flash、base、qwen；留空下载全部")
    download.add_argument("--verify-hashes", action="store_true", help="跳过前先完整校验，损坏时重新下载")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = LocalPaths.from_root(args.root)
    os.environ["AUK_LOCAL_HOME"] = str(paths.root)
    if args.command == "diagnose":
        print(
            json.dumps(
                runtime_diagnostic(
                    paths,
                    probe_torch=not args.no_torch,
                    verify_hashes=args.verify_hashes,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "download-models":
        from .download import download_models

        selected = tuple(args.models) if args.models else ("flash", "base", "qwen")
        unknown = sorted(set(selected) - {"flash", "base", "qwen"})
        if unknown:
            raise SystemExit("未知模型：" + ", ".join(unknown))
        download_models(paths, selected, verify_hashes=args.verify_hashes)
        return
    if args.command == "serve":
        import uvicorn

        from .service import create_app

        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            raise SystemExit("AuK Local 首版只允许绑定本机 loopback")
        if args.open_browser and not args.no_ui:
            webbrowser.open(f"http://127.0.0.1:{args.port}")
        uvicorn.run(
            create_app(paths, with_ui=not args.no_ui),
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=False,
        )


if __name__ == "__main__":
    main()
