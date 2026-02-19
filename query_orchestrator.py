#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : query_orchestrator.py
@Author  : Kevin
@Date    : 2026/02/19
@Description : 查询意图路由器。
    对外仅暴露 QueryOrchestrator 类，入口为 route(user_query, scene_key)。
    工作原理：
      1. 意图识别（规则优先，LLM 兜底）识别出 count / list / rag 三种类型。
      2. count / list → 读取 catalog.json，返回确定性答案，不消耗 LLM 生成 token。
      3. rag         → 透传给调用方（由 MultiSceneRAG.query_engine 处理）。
@Version : 1.0
"""

import re
from pathlib import Path
from typing import Any

CATALOG_PATH = Path("./storage/catalog.json")

# ──────────────────────────────────────────────────────────
# 意图规则：(pattern, intent)，按优先级从高到低排列
# ──────────────────────────────────────────────────────────
_INTENT_RULES: list[tuple[re.Pattern[str], str]] = [
    # count：量词组合（允许中间有修饰词）
    (re.compile(r"(有|共|一共|总共|总计|合计)(几|多少)(个|本|份|条|篇|项|种|类|张|页)", re.I), "count"),
    (re.compile(r"(几|多少)(个|本|份|条|篇|项|种|类|张|页)", re.I), "count"),
    (re.compile(r"(数量|总数|总量|计数|共有多少|一共多少)", re.I), "count"),
    # list：动词/疑问词 + 最多10字修饰词 + 目标名词
    (re.compile(r"(列出|列举|显示|展示|给我看).{0,10}(文件|文档|书|书籍|资料|制度|政策|条款|规则|规定|流程)", re.I), "list"),
    (re.compile(r"(有哪些|都有哪些|所有的?).{0,10}(文件|文档|书|书籍|资料|制度|政策|条款|规则|规定|流程)", re.I), "list"),
    (re.compile(r"(文件|文档|书|书籍|资料)列表", re.I), "list"),
]


def _detect_intent_by_rules(query: str) -> str | None:
    """基于正则规则快速识别意图，命中则返回 intent，否则返回 None。"""
    for pattern, intent in _INTENT_RULES:
        if pattern.search(query):
            return intent
    return None


# ──────────────────────────────────────────────────────────
# Catalog 读取
# ──────────────────────────────────────────────────────────

def _read_catalog() -> dict[str, Any]:
    """读取场景文档清单，不存在或损坏时返回空 dict。"""
    if not CATALOG_PATH.exists():
        return {}
    try:
        import json
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_scene_catalog(scene_key: str) -> dict[str, Any] | None:
    """返回指定场景的 catalog 条目，不存在时返回 None。"""
    return _read_catalog().get(scene_key)


# ──────────────────────────────────────────────────────────
# 确定性回答生成
# ──────────────────────────────────────────────────────────

def _answer_count(scene_key: str, scene_name: str) -> dict[str, Any]:
    """根据 catalog 生成文档数量的确定性回答。"""
    catalog = _get_scene_catalog(scene_key)
    if catalog is None:
        return {
            "intent": "count",
            "answer": f"【{scene_name}】场景的文档目录尚未建立索引，无法统计数量。请先重建索引。",
            "source": "catalog_missing",
            "deterministic": True,
        }
    docs = catalog.get("documents", [])
    count = len(docs)
    if count == 0:
        return {
            "intent": "count",
            "answer": f"【{scene_name}】场景当前没有已索引的文档。",
            "source": "catalog",
            "deterministic": True,
        }
    doc_names = [d.get("file_name", d.get("file_path", "未知")) for d in docs]
    names_text = "、".join(f"《{n}》" for n in doc_names)
    return {
        "intent": "count",
        "answer": f"【{scene_name}】场景共有 {count} 个文档：{names_text}。",
        "source": "catalog",
        "deterministic": True,
        "document_count": count,
        "documents": doc_names,
    }


def _answer_list(scene_key: str, scene_name: str) -> dict[str, Any]:
    """根据 catalog 生成文档列表的确定性回答。"""
    catalog = _get_scene_catalog(scene_key)
    if catalog is None:
        return {
            "intent": "list",
            "answer": f"【{scene_name}】场景的文档目录尚未建立索引，无法列出文档。请先重建索引。",
            "source": "catalog_missing",
            "deterministic": True,
        }
    docs = catalog.get("documents", [])
    if not docs:
        return {
            "intent": "list",
            "answer": f"【{scene_name}】场景当前没有已索引的文档。",
            "source": "catalog",
            "deterministic": True,
        }
    lines = []
    for idx, d in enumerate(docs, start=1):
        name = d.get("file_name", d.get("file_path", "未知"))
        size_kb = round(d.get("file_size", 0) / 1024, 1)
        lines.append(f"  {idx}. {name}（{size_kb} KB）")
    listing = "\n".join(lines)
    return {
        "intent": "list",
        "answer": f"【{scene_name}】场景共 {len(docs)} 个文档：\n{listing}",
        "source": "catalog",
        "deterministic": True,
        "document_count": len(docs),
        "documents": [d.get("file_name", d.get("file_path", "")) for d in docs],
    }


# ──────────────────────────────────────────────────────────
# 主路由器
# ──────────────────────────────────────────────────────────

class QueryOrchestrator:
    """查询意图路由器。

    使用方法：
        result = QueryOrchestrator().route(user_query, scene_key, scene_name)
        if result["intent"] == "rag":
            # 交给 RAG 引擎
        else:
            # 直接使用 result["answer"]
    """

    def route(
        self,
        user_query: str,
        scene_key: str,
        scene_name: str = "",
    ) -> dict[str, Any]:
        """识别意图并路由到对应处理链路。

        Returns:
            dict 包含：
              - intent: "count" | "list" | "rag"
              - answer: 确定性回答（count/list 时有值，rag 时为空字符串）
              - deterministic: bool，是否为确定性回答
              - source: 数据来源标识
        """
        intent = _detect_intent_by_rules(user_query)

        if intent == "count":
            print(f"🎯 QueryOrchestrator: intent=count, scene={scene_key}")
            return _answer_count(scene_key, scene_name or scene_key)

        if intent == "list":
            print(f"🎯 QueryOrchestrator: intent=list, scene={scene_key}")
            return _answer_list(scene_key, scene_name or scene_key)

        print(f"🎯 QueryOrchestrator: intent=rag, scene={scene_key}")
        return {
            "intent": "rag",
            "answer": "",
            "deterministic": False,
            "source": "rag",
        }
