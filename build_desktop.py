#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : build_desktop.py
@Author  : Kevin
@Date    : 2026/02/18
@Description : Phase 4 桌面端打包脚本（PyInstaller）.
@Version : 1.0
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name",
        "MultiSceneRAG",
        "--add-data",
        "web;web",
        "--add-data",
        "data;data",
        "desktop_main.py",
    ]

    print("🔨 开始构建桌面应用（PyInstaller）...")
    subprocess.run(command, cwd=project_root, check=True)
    print("✅ 构建完成。输出目录：dist/MultiSceneRAG")


if __name__ == "__main__":
    main()
