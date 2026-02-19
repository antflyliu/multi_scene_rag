#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接口回归：/api/chat 统计问题返回确定性结果。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_api_server(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    target = repo_root / "api_server.py"
    spec = importlib.util.spec_from_file_location("api_server_under_test", str(target))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 api_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeEngine:
    def query_with_usage(self, query: str):
        if "几本书" in query:
            return {
                "answer": "当前场景“书籍”共有 2 份文档。例如：A.epub; B.epub。",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "embedding_tokens": 0},
            }
        return {
            "answer": "FAKE_RAG",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "embedding_tokens": 0},
        }


def run_regression() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_api_server(repo_root)

    module._engine_instance = _FakeEngine()
    module._rebuild_running = False
    module._get_engine = lambda eager_init=False: module._engine_instance

    # 不引入第三方依赖，直接调用路由函数做接口层验证。
    count_resp = module.chat(module.ChatRequest(query="你知识库里有几本书？"))
    assert "2 份文档" in str(count_resp.get("answer", "")), count_resp
    assert int((count_resp.get("usage") or {}).get("total_tokens", -1)) == 0, count_resp

    rag_resp = module.chat(module.ChatRequest(query="请介绍报销流程"))
    assert rag_resp.get("answer") == "FAKE_RAG", rag_resp
    assert int((rag_resp.get("usage") or {}).get("total_tokens", -1)) == 15, rag_resp

    print("PASS: api-chat regression")


if __name__ == "__main__":
    run_regression()
