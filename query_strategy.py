#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : query_strategy.py
@Author  : Kevin
@Date    : 2026/02/19
@Description : 查询策略与确定性回答规则.
@Version : 1.0
"""

from __future__ import annotations

import re
from typing import Any, Literal

QueryIntent = Literal["count", "list", "qa"]

_COUNT_PATTERNS = [
    re.compile(r"(几本|几份|几条|几项|几个|多少本|多少份|多少条|多少项|多少个|总数|数量)"),
]
_LIST_PATTERNS = [
    re.compile(r"(有哪些|有什么|列出|清单|名单|目录|列一下|都有什么|全部列出|全部文档)"),
]
_CATALOG_SCOPE_TERMS = {
    "知识库",
    "文档",
    "文件",
    "资料",
    "书",
    "书籍",
    "电子书",
    "epub",
    "pdf",
    "txt",
    "md",
    "docx",
    "库里",
    "目录",
}


def is_catalog_scope_query(user_query: str) -> bool:
    text = (user_query or "").strip().lower()
    if not text:
        return False
    return any(term in text for term in _CATALOG_SCOPE_TERMS)


def detect_query_intent(user_query: str) -> QueryIntent:
    text = (user_query or "").strip().lower()
    if not text:
        return "qa"
    if is_catalog_scope_query(text):
        if any(pattern.search(text) for pattern in _COUNT_PATTERNS):
            return "count"
        if any(pattern.search(text) for pattern in _LIST_PATTERNS):
            return "list"
    return "qa"


def format_catalog_count_answer(scene_name: str, scene_key: str, document_names: list[str]) -> str:
    safe_scene_name = scene_name or scene_key or "未知场景"
    count = len(document_names)
    if count == 0:
        return f"当前 {safe_scene_name}（{scene_key}）知识库暂无文档。"
    preview_names = "；".join(document_names[:5])
    if count > 5:
        preview_names = f"{preview_names}；..."
    return (
        f"当前 {safe_scene_name}（{scene_key}）知识库共有 {count} 份文档。"
        f"已收录文档：{preview_names}"
    )


def format_catalog_list_answer(scene_name: str, scene_key: str, document_names: list[str]) -> str:
    safe_scene_name = scene_name or scene_key or "未知场景"
    count = len(document_names)
    if count == 0:
        return f"当前 {safe_scene_name}（{scene_key}）知识库暂无文档。"
    numbered_lines = [f"{idx}. {name}" for idx, name in enumerate(document_names, start=1)]
    return (
        f"当前 {safe_scene_name}（{scene_key}）知识库共 {count} 份文档：\n"
        + "\n".join(numbered_lines)
    )


def resolve_scene_for_catalog_query(
    user_query: str,
    scenes: dict[str, dict[str, Any]],
    fallback_scene: str,
) -> str:
    text = (user_query or "").strip().lower()
    if not text:
        return fallback_scene

    for scene_key, info in scenes.items():
        if scene_key.lower() in text:
            return scene_key
        scene_name = str(info.get("name", "")).strip().lower()
        if scene_name and scene_name in text:
            return scene_key

    if any(marker in text for marker in ("书", "书籍", "电子书", "epub")) and "book" in scenes:
        return "book"

    return fallback_scene
