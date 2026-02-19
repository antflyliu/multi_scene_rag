#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回归测试：QueryOrchestrator 意图识别 + catalog 确定性回答。

覆盖范围：
  1. intent 识别准确率（count / list / rag），跨全部场景关键词变体
  2. catalog 缺失时的安全降级
  3. catalog 存在时 count/list 返回确定性结果
  4. rag intent 的透传行为

运行方式（项目根目录）：
    python -m pytest tests/test_query_orchestrator.py -v
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ── 保证从项目根可以 import ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from query_orchestrator import (
    CATALOG_PATH,
    QueryOrchestrator,
    _detect_intent_by_rules,
)


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _make_catalog(tmp_path: Path, data: dict) -> Path:
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return catalog_file


# ══════════════════════════════════════════════════════════════
# 1. 意图识别 — count
# ══════════════════════════════════════════════════════════════

class TestIntentCount:
    """count intent 覆盖各场景典型问法"""

    @pytest.mark.parametrize("query", [
        # 通用计数
        "有几本书",
        "一共有几本书",
        "共几本书",
        "知识库里总共有几本",
        "有多少本书",
        "有几个文件",
        "有多少份文档",
        "有几篇文章",
        "有几条规则",
        "总共有几项政策",
        # finance 场景
        "有几条报销规定",
        "财务制度有多少条",
        # hr 场景
        "人事政策有几个",
        "HR文档有多少份",
        # it 场景
        "IT手册有几本",
        # 数量/统计
        "数量是多少",
        "总数是多少",
        "合计多少个文档",
    ])
    def test_count_intent_detected(self, query: str):
        assert _detect_intent_by_rules(query) == "count", f"应识别为 count: {query!r}"


# ══════════════════════════════════════════════════════════════
# 2. 意图识别 — list
# ══════════════════════════════════════════════════════════════

class TestIntentList:
    """list intent 覆盖各场景典型问法"""

    @pytest.mark.parametrize("query", [
        "列出所有书籍",
        "列举所有文件",
        "显示所有文档",
        "展示所有资料",
        "有哪些书",
        "都有哪些书籍",
        "有哪些报销文件",
        "有哪些IT文档",
        "有哪些政策",
        "所有的规定有哪些",
        "给我看所有的制度",
        "文件列表",
        "书籍列表",
        "文档列表",
    ])
    def test_list_intent_detected(self, query: str):
        assert _detect_intent_by_rules(query) == "list", f"应识别为 list: {query!r}"


# ══════════════════════════════════════════════════════════════
# 3. 意图识别 — rag（不应被误匹配）
# ══════════════════════════════════════════════════════════════

class TestIntentRag:
    """rag intent：正常问答不应被误识别为 count/list"""

    @pytest.mark.parametrize("query", [
        "年假怎么申请",
        "报销差旅费需要什么材料",
        "电脑无法连接VPN怎么办",
        "RAG是什么技术",
        "怎么处理员工离职手续",
        "发票丢失怎么报销",
        "介绍一下这本书的主要内容",
        "书里讲了什么",
        "告诉我更多关于RAG的信息",
    ])
    def test_rag_intent_not_intercepted(self, query: str):
        result = _detect_intent_by_rules(query)
        assert result is None, f"不应被规则匹配: {query!r}，实际={result}"


# ══════════════════════════════════════════════════════════════
# 4. Orchestrator.route — catalog 缺失时安全降级
# ══════════════════════════════════════════════════════════════

class TestOrchestratorCatalogMissing:
    def test_count_without_catalog(self, tmp_path):
        missing_path = tmp_path / "nonexistent_catalog.json"
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", missing_path):
            result = orch.route("有几本书", "book", "书籍")
        assert result["intent"] == "count"
        assert result["deterministic"] is True
        assert "尚未建立索引" in result["answer"] or "无法统计" in result["answer"]

    def test_list_without_catalog(self, tmp_path):
        missing_path = tmp_path / "nonexistent_catalog.json"
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", missing_path):
            result = orch.route("列出所有书", "book", "书籍")
        assert result["intent"] == "list"
        assert result["deterministic"] is True
        assert "尚未建立索引" in result["answer"] or "无法列出" in result["answer"]


# ══════════════════════════════════════════════════════════════
# 5. Orchestrator.route — catalog 存在时返回确定性结果
# ══════════════════════════════════════════════════════════════

class TestOrchestratorWithCatalog:
    """各场景 catalog 存在时验证 count/list 的准确性"""

    @pytest.fixture
    def catalog_2_books(self, tmp_path):
        data = {
            "book": {
                "scene_key": "book",
                "data_path": "./data/books",
                "document_count": 2,
                "documents": [
                    {"file_name": "Mastering RAG for AI Build Scalable.epub", "file_path": "", "file_size": 2800000},
                    {"file_name": "Mastering RAG for AI Agents.epub", "file_path": "", "file_size": 480000},
                ],
            }
        }
        return _make_catalog(tmp_path, data)

    @pytest.fixture
    def catalog_finance(self, tmp_path):
        data = {
            "finance": {
                "scene_key": "finance",
                "data_path": "./data/finance",
                "document_count": 1,
                "documents": [
                    {"file_name": "财务报销制度.txt", "file_path": "", "file_size": 86000},
                ],
            }
        }
        return _make_catalog(tmp_path, data)

    @pytest.fixture
    def catalog_hr(self, tmp_path):
        data = {
            "hr": {
                "scene_key": "hr",
                "data_path": "./data/hr",
                "document_count": 1,
                "documents": [
                    {"file_name": "人力资源管理制度.txt", "file_path": "", "file_size": 269000},
                ],
            }
        }
        return _make_catalog(tmp_path, data)

    # ── count ──────────────────────────────────────────────────

    def test_count_books_returns_2(self, catalog_2_books):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_2_books):
            result = orch.route("有几本书", "book", "书籍")
        assert result["intent"] == "count"
        assert result["deterministic"] is True
        assert result["document_count"] == 2
        assert "2" in result["answer"]

    def test_count_finance(self, catalog_finance):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_finance):
            result = orch.route("财务制度有多少条文件", "finance", "财务报销")
        assert result["intent"] == "count"
        assert result["document_count"] == 1

    def test_count_hr(self, catalog_hr):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_hr):
            result = orch.route("HR有几份政策文档", "hr", "人力资源政策")
        assert result["intent"] == "count"
        assert result["document_count"] == 1

    # ── list ───────────────────────────────────────────────────

    def test_list_books_contains_both_titles(self, catalog_2_books):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_2_books):
            result = orch.route("列出所有书籍", "book", "书籍")
        assert result["intent"] == "list"
        assert result["deterministic"] is True
        assert result["document_count"] == 2
        assert "Mastering RAG for AI Build Scalable.epub" in result["documents"]
        assert "Mastering RAG for AI Agents.epub" in result["documents"]
        # 答案里要有序号
        assert "1." in result["answer"]
        assert "2." in result["answer"]

    def test_list_finance(self, catalog_finance):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_finance):
            result = orch.route("有哪些报销文件", "finance", "财务报销")
        assert result["intent"] == "list"
        assert "财务报销制度.txt" in result["documents"]

    # ── rag 透传 ───────────────────────────────────────────────

    def test_rag_passthrough(self, catalog_2_books):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_2_books):
            result = orch.route("书里讲了什么RAG技术", "book", "书籍")
        assert result["intent"] == "rag"
        assert result["deterministic"] is False
        assert result["answer"] == ""

    # ── source 字段 ────────────────────────────────────────────

    def test_source_is_catalog_when_deterministic(self, catalog_2_books):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_2_books):
            result = orch.route("有几本书", "book", "书籍")
        assert result["source"] == "catalog"

    def test_source_is_rag_when_passthrough(self, catalog_2_books):
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_2_books):
            result = orch.route("RAG和传统搜索有什么区别", "book", "书籍")
        assert result["source"] == "rag"


# ══════════════════════════════════════════════════════════════
# 6. 空文档场景的边界处理
# ══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_documents_count(self, tmp_path):
        data = {"it": {"scene_key": "it", "data_path": "./data/it", "document_count": 0, "documents": []}}
        catalog_file = _make_catalog(tmp_path, data)
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_file):
            result = orch.route("IT文档有几个", "it", "IT支持")
        assert result["intent"] == "count"
        assert result["deterministic"] is True
        assert "没有" in result["answer"]

    def test_empty_documents_list(self, tmp_path):
        data = {"it": {"scene_key": "it", "data_path": "./data/it", "document_count": 0, "documents": []}}
        catalog_file = _make_catalog(tmp_path, data)
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_file):
            result = orch.route("IT有哪些文档", "it", "IT支持")
        assert result["intent"] == "list"
        assert "没有" in result["answer"]

    def test_scene_not_in_catalog(self, tmp_path):
        """catalog 存在但没有该场景条目"""
        data = {"hr": {"scene_key": "hr", "documents": []}}
        catalog_file = _make_catalog(tmp_path, data)
        orch = QueryOrchestrator()
        with patch("query_orchestrator.CATALOG_PATH", catalog_file):
            result = orch.route("有几本书", "book", "书籍")
        assert result["deterministic"] is True
        assert "尚未建立索引" in result["answer"] or "无法统计" in result["answer"]
