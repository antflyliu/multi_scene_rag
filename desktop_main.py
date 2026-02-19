#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : desktop_main.py
@Author  : Kevin
@Date    : 2026/02/18
@Description : Phase 4 桌面应用入口（pywebview）.
@Version : 1.0
"""

import os
import sys
import threading
import time
import urllib.request

import uvicorn
import webview

from api_server import app

HOST = "127.0.0.1"
PORT = int(os.getenv("MSRAG_PORT", "8000"))
STARTUP_TIMEOUT_SECONDS = 90


def _prepare_runtime_cwd() -> None:
    """确保桌面版从应用目录运行，避免相对路径异常。"""
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)


def _health_url() -> str:
    return f"http://{HOST}:{PORT}/health"


def _ui_url() -> str:
    return f"http://{HOST}:{PORT}/"


def _is_server_ready(timeout_seconds: int = 2) -> bool:
    try:
        with urllib.request.urlopen(_health_url(), timeout=timeout_seconds) as resp:
            return resp.status == 200
    except Exception:
        return False


class UvicornRunner:
    """管理内嵌 API 服务生命周期。"""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None

    def start(self) -> None:
        if _is_server_ready():
            print("✅ 检测到已有 API 服务，复用现有实例。")
            return

        config = uvicorn.Config(
            app=app,
            host=HOST,
            port=PORT,
            reload=False,
            log_level="info",
        )
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        start_at = time.time()
        while not _is_server_ready():
            if time.time() - start_at > STARTUP_TIMEOUT_SECONDS:
                raise TimeoutError("API 服务启动超时，请检查依赖和端口占用。")
            time.sleep(0.5)

        print("✅ API 服务已启动。")

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        print("✅ API 服务已停止。")


def main() -> None:
    _prepare_runtime_cwd()
    runner = UvicornRunner()
    runner.start()

    try:
        webview.create_window(
            title="Multi-Scene RAG",
            url=_ui_url(),
            width=1200,
            height=820,
            min_size=(980, 680),
        )
        webview.start()
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
