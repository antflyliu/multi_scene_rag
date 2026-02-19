#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归测试：统计/列表查询分流与 RAG 回退。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path


def _stub_dependencies() -> None:
    # ---- llama_index stubs ----
    core = types.ModuleType("llama_index.core")
    core.Settings = types.SimpleNamespace(llm=None, embed_model=None, callback_manager=None)

    class _DummyIndex:
        def __init__(self) -> None:
            self.storage_context = types.SimpleNamespace(persist=lambda persist_dir: None)

        def as_query_engine(self):
            return types.SimpleNamespace(query=lambda q: f"DUMMY_RAG:{q}")

    core.VectorStoreIndex = types.SimpleNamespace(from_documents=lambda *args, **kwargs: _DummyIndex())
    core.SimpleDirectoryReader = lambda path: types.SimpleNamespace(load_data=lambda: [])
    core.StorageContext = types.SimpleNamespace(from_defaults=lambda **kwargs: types.SimpleNamespace())
    core.load_index_from_storage = lambda storage_context: _DummyIndex()

    callbacks = types.ModuleType("llama_index.core.callbacks")
    callbacks.CallbackManager = lambda handlers: types.SimpleNamespace(handlers=handlers)
    callbacks.TokenCountingHandler = lambda: types.SimpleNamespace(
        prompt_llm_token_count=0,
        completion_llm_token_count=0,
        total_llm_token_count=0,
        total_embedding_token_count=0,
    )

    vector_stores_chroma = types.ModuleType("llama_index.vector_stores.chroma")
    vector_stores_chroma.ChromaVectorStore = lambda chroma_collection: types.SimpleNamespace()

    llm_dashscope = types.ModuleType("llama_index.llms.dashscope")
    llm_dashscope.DashScope = lambda **kwargs: types.SimpleNamespace()
    llm_dashscope.DashScopeGenerationModels = types.SimpleNamespace(QWEN_MAX="qwen-max")

    llm_openai = types.ModuleType("llama_index.llms.openai")
    llm_openai.OpenAI = lambda **kwargs: types.SimpleNamespace()

    emb_openai = types.ModuleType("llama_index.embeddings.openai")
    emb_openai.OpenAIEmbedding = lambda **kwargs: types.SimpleNamespace()

    emb_hf = types.ModuleType("llama_index.embeddings.huggingface")
    emb_hf.HuggingFaceEmbedding = lambda **kwargs: types.SimpleNamespace()

    emb_dashscope = types.ModuleType("llama_index.embeddings.dashscope")
    emb_dashscope.DashScopeEmbedding = lambda **kwargs: types.SimpleNamespace()
    emb_dashscope.DashScopeTextEmbeddingModels = types.SimpleNamespace(TEXT_EMBEDDING_V1="text-embedding-v1")
    emb_dashscope.DashScopeTextEmbeddingType = types.SimpleNamespace(TEXT_TYPE_DOCUMENT="document")

    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = lambda path: types.SimpleNamespace(
        get_or_create_collection=lambda name: types.SimpleNamespace(),
        delete_collection=lambda name: None,
    )

    classifier = types.ModuleType("classifier")
    classifier.classify_scene = lambda query, runtime_mode="web": "hr"

    config_store = types.ModuleType("config_store")
    config_store.LLM_VENDOR_BASE_URLS = {"dashscope": "x", "openai": "y"}
    config_store.get_api_key = lambda runtime_mode="web": "dummy-key"
    config_store.get_base_url = lambda runtime_mode="web": "https://example.com/v1"
    config_store.get_default_scene_key = lambda: "hr"
    config_store.get_embedding_device = lambda runtime_mode="web": "cpu"
    config_store.get_embedding_model = lambda runtime_mode="web": "text-embedding-v1"
    config_store.get_embedding_provider = lambda runtime_mode="web": "dashscope"
    config_store.get_embedding_source = lambda runtime_mode="web": "huggingface"
    config_store.get_llm_vendor = lambda runtime_mode="web": "dashscope"
    config_store.get_model_name = lambda runtime_mode="web": "qwen-flash"
    config_store.get_scenes = lambda: {}

    sys.modules["llama_index"] = types.ModuleType("llama_index")
    sys.modules["llama_index.core"] = core
    sys.modules["llama_index.core.callbacks"] = callbacks
    sys.modules["llama_index.vector_stores.chroma"] = vector_stores_chroma
    sys.modules["llama_index.llms.dashscope"] = llm_dashscope
    sys.modules["llama_index.llms.openai"] = llm_openai
    sys.modules["llama_index.embeddings.openai"] = emb_openai
    sys.modules["llama_index.embeddings.huggingface"] = emb_hf
    sys.modules["llama_index.embeddings.dashscope"] = emb_dashscope
    sys.modules["chromadb"] = chromadb
    sys.modules["classifier"] = classifier
    sys.modules["config_store"] = config_store


def _load_rag_engine_module(repo_root: Path):
    _stub_dependencies()
    target = repo_root / "rag_engine.py"
    spec = importlib.util.spec_from_file_location("rag_engine_under_test", str(target))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 rag_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_regression() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_rag_engine_module(repo_root)
    MultiSceneRAG = module.MultiSceneRAG

    with tempfile.TemporaryDirectory(prefix="rag-reg-") as tmp_dir:
        tmp = Path(tmp_dir)
        book_dir = tmp / "books"
        hr_dir = tmp / "hr"
        book_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / "A.epub").write_text("A", encoding="utf-8")
        (book_dir / "B.epub").write_text("B", encoding="utf-8")
        (hr_dir / "policy.txt").write_text("HR policy", encoding="utf-8")

        engine = MultiSceneRAG.__new__(MultiSceneRAG)
        engine.runtime_mode = "web"
        engine._token_counter = None
        engine.indices = {"hr": types.SimpleNamespace(query=lambda q: "HR_RAG_ANSWER")}
        engine.scenes = {
            "hr": {"name": "人力资源政策", "keywords": ["请假"], "path": str(hr_dir)},
            "book": {"name": "书籍", "keywords": ["书"], "path": str(book_dir)},
        }
        # 指向临时 catalog，避免污染仓库数据。
        module.SCENE_CATALOG_PATH = tmp / "scene_catalog.json"

        # 1) 统计问题应命中 deterministic 分支，并能覆盖路由误差（默认 classify_scene 返回 hr）。
        result_count = engine.query_with_usage("你知识库里有几本书？")
        assert "2 份文档" in result_count["answer"], result_count
        assert result_count["usage"]["total_tokens"] == 0, result_count

        # 2) 列表问题应列出具体文件名。
        result_list = engine.query_with_usage("请列出所有书籍")
        assert "A.epub" in result_list["answer"] and "B.epub" in result_list["answer"], result_list

        # 3) 非统计问题应走 RAG 分支。
        rag_called = {"value": False}

        def _ensure(scene: str) -> None:
            rag_called["value"] = True

        engine._ensure_scene_index = _ensure
        result_rag = engine.query_with_usage("请介绍一下请假制度")
        assert rag_called["value"] is True, result_rag
        assert result_rag["answer"] == "HR_RAG_ANSWER", result_rag

    print("PASS: structured-query regression")


if __name__ == "__main__":
    run_regression()
