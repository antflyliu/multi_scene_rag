#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量回归测试套件。

覆盖范围：
  - 每场景 count / list / qa 各 1 条
  - 跨场景路由纠偏（路由到 hr 但问"几本书"）
  - answer_mode 字段存在且值正确
  - structured 分支 token 消耗为 0
  - qa 分支走 RAG 引擎
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

def _stub_dependencies(classify_result: str = "hr") -> None:
    core = types.ModuleType("llama_index.core")
    core.Settings = types.SimpleNamespace(llm=None, embed_model=None, callback_manager=None)

    class _DummyIndex:
        def __init__(self) -> None:
            self.storage_context = types.SimpleNamespace(persist=lambda persist_dir: None)

        def as_query_engine(self):
            return types.SimpleNamespace(query=lambda q: f"RAG:{q}")

    core.VectorStoreIndex = types.SimpleNamespace(from_documents=lambda *a, **kw: _DummyIndex())
    core.SimpleDirectoryReader = lambda path: types.SimpleNamespace(load_data=lambda: [])
    core.StorageContext = types.SimpleNamespace(from_defaults=lambda **kw: types.SimpleNamespace())
    core.load_index_from_storage = lambda sc: _DummyIndex()

    cb = types.ModuleType("llama_index.core.callbacks")
    cb.CallbackManager = lambda h: types.SimpleNamespace()
    cb.TokenCountingHandler = lambda: types.SimpleNamespace(
        prompt_llm_token_count=0, completion_llm_token_count=0,
        total_llm_token_count=0, total_embedding_token_count=0,
    )

    def _stub(name: str, **attrs: object) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    sys.modules["llama_index"] = types.ModuleType("llama_index")
    sys.modules["llama_index.core"] = core
    sys.modules["llama_index.core.callbacks"] = cb
    sys.modules["llama_index.vector_stores.chroma"] = _stub(
        "llama_index.vector_stores.chroma",
        ChromaVectorStore=lambda chroma_collection: types.SimpleNamespace(),
    )
    sys.modules["llama_index.llms.dashscope"] = _stub(
        "llama_index.llms.dashscope",
        DashScope=lambda **kw: None,
        DashScopeGenerationModels=types.SimpleNamespace(QWEN_MAX="qwen-max"),
    )
    sys.modules["llama_index.llms.openai"] = _stub(
        "llama_index.llms.openai", OpenAI=lambda **kw: None
    )
    sys.modules["llama_index.embeddings.openai"] = _stub(
        "llama_index.embeddings.openai", OpenAIEmbedding=lambda **kw: None
    )
    sys.modules["llama_index.embeddings.huggingface"] = _stub(
        "llama_index.embeddings.huggingface", HuggingFaceEmbedding=lambda **kw: None
    )
    sys.modules["llama_index.embeddings.dashscope"] = _stub(
        "llama_index.embeddings.dashscope",
        DashScopeEmbedding=lambda **kw: None,
        DashScopeTextEmbeddingModels=types.SimpleNamespace(TEXT_EMBEDDING_V1="text-embedding-v1"),
        DashScopeTextEmbeddingType=types.SimpleNamespace(TEXT_TYPE_DOCUMENT="document"),
    )
    sys.modules["chromadb"] = _stub(
        "chromadb",
        PersistentClient=lambda path: types.SimpleNamespace(
            get_or_create_collection=lambda n: types.SimpleNamespace(),
            delete_collection=lambda n: None,
        ),
    )

    classifier = types.ModuleType("classifier")
    classifier.classify_scene = lambda query, runtime_mode="web": classify_result
    sys.modules["classifier"] = classifier

    config_store = types.ModuleType("config_store")
    config_store.LLM_VENDOR_BASE_URLS = {"dashscope": "x", "openai": "y"}
    config_store.get_api_key = lambda runtime_mode="web": "dummy"
    config_store.get_base_url = lambda runtime_mode="web": "https://x"
    config_store.get_default_scene_key = lambda: "hr"
    config_store.get_embedding_device = lambda runtime_mode="web": "cpu"
    config_store.get_embedding_model = lambda runtime_mode="web": "text-embedding-v1"
    config_store.get_embedding_provider = lambda runtime_mode="web": "dashscope"
    config_store.get_embedding_source = lambda runtime_mode="web": "huggingface"
    config_store.get_llm_vendor = lambda runtime_mode="web": "dashscope"
    config_store.get_model_name = lambda runtime_mode="web": "qwen-flash"
    config_store.get_scenes = lambda: {}
    sys.modules["config_store"] = config_store


def _load_engine(repo_root: Path):
    _stub_dependencies()
    spec = importlib.util.spec_from_file_location(
        "rag_engine_under_test", str(repo_root / "rag_engine.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class SceneFixture:
    scene_key: str
    scene_name: str
    files: list[str]
    data_dir: Path
    extra_scenes: dict[str, dict] = field(default_factory=dict)


def _make_engine(mod, tmp: Path, fixture: SceneFixture, route_to: str):
    """构建无 LLM 依赖的 MultiSceneRAG 实例。"""
    engine = mod.MultiSceneRAG.__new__(mod.MultiSceneRAG)
    engine.runtime_mode = "web"
    engine._token_counter = None
    engine.scenes = {
        fixture.scene_key: {
            "name": fixture.scene_name,
            "keywords": [],
            "path": str(fixture.data_dir),
        },
        **fixture.extra_scenes,
    }
    engine.indices = {
        fixture.scene_key: types.SimpleNamespace(query=lambda q: f"RAG_ANSWER:{q}")
    }
    for extra_key in fixture.extra_scenes:
        engine.indices[extra_key] = types.SimpleNamespace(query=lambda q: f"RAG_ANSWER:{q}")

    mod.SCENE_CATALOG_PATH = tmp / "catalog.json"

    sys.modules["classifier"].classify_scene = lambda query, runtime_mode="web": route_to
    return engine


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    desc: str
    scene_key: str
    scene_name: str
    files: list[str]
    route_to: str
    query: str
    expect_mode: str
    expect_in_answer: list[str] = field(default_factory=list)
    extra_scenes: dict[str, dict] = field(default_factory=dict)


CASES: list[Case] = [
    # book 场景
    Case(
        desc="[book] count: 几本书",
        scene_key="book", scene_name="书籍",
        files=["A.epub", "B.epub"],
        route_to="hr",
        query="你知识库里有几本书？",
        expect_mode="structured",
        expect_in_answer=["2 份文档"],
    ),
    Case(
        desc="[book] list: 列出所有书籍",
        scene_key="book", scene_name="书籍",
        files=["A.epub", "B.epub"],
        route_to="hr",
        query="请列出所有书籍",
        expect_mode="structured",
        expect_in_answer=["A.epub", "B.epub"],
    ),
    Case(
        desc="[book] qa: 普通问答走 RAG",
        scene_key="book", scene_name="书籍",
        files=["A.epub"],
        route_to="book",
        query="RAG框架的核心原理是什么？",
        expect_mode="rag",
        expect_in_answer=["RAG_ANSWER"],
    ),
    # hr 场景
    Case(
        desc="[hr] count: 有几种假期",
        scene_key="hr", scene_name="人力资源政策",
        files=["policy.txt"],
        route_to="hr",
        query="公司有几种假期？",
        expect_mode="structured",
        expect_in_answer=["1 份文档"],
    ),
    Case(
        desc="[hr] list: 列出所有政策文件",
        scene_key="hr", scene_name="人力资源政策",
        files=["policy.txt"],
        route_to="hr",
        query="列出所有政策文档",
        expect_mode="structured",
        expect_in_answer=["policy.txt"],
    ),
    Case(
        desc="[hr] qa: 请假规定走 RAG",
        scene_key="hr", scene_name="人力资源政策",
        files=["policy.txt"],
        route_to="hr",
        query="请假需要提前多久申请？",
        expect_mode="rag",
        expect_in_answer=["RAG_ANSWER"],
    ),
    # finance 场景
    Case(
        desc="[finance] count: 有几条报销规则",
        scene_key="finance", scene_name="财务报销",
        files=["rules.txt"],
        route_to="finance",
        query="有几条报销规则？",
        expect_mode="structured",
        expect_in_answer=["1 份文档"],
    ),
    Case(
        desc="[finance] list: 列出所有报销文档",
        scene_key="finance", scene_name="财务报销",
        files=["rules.txt"],
        route_to="finance",
        query="列出所有报销文档",
        expect_mode="structured",
        expect_in_answer=["rules.txt"],
    ),
    Case(
        desc="[finance] qa: 差旅报销上限走 RAG",
        scene_key="finance", scene_name="财务报销",
        files=["rules.txt"],
        route_to="finance",
        query="差旅报销的上限是多少？",
        expect_mode="rag",
        expect_in_answer=["RAG_ANSWER"],
    ),
    # it 场景
    Case(
        desc="[it] count: 有几类故障",
        scene_key="it", scene_name="IT支持",
        files=["guide.txt"],
        route_to="it",
        query="公司有几类IT故障？",
        expect_mode="structured",
        expect_in_answer=["1 份文档"],
    ),
    Case(
        desc="[it] list: 列出所有系统文档",
        scene_key="it", scene_name="IT支持",
        files=["guide.txt"],
        route_to="it",
        query="列出所有系统文档",
        expect_mode="structured",
        expect_in_answer=["guide.txt"],
    ),
    Case(
        desc="[it] qa: VPN 设置走 RAG",
        scene_key="it", scene_name="IT支持",
        files=["guide.txt"],
        route_to="it",
        query="如何配置VPN？",
        expect_mode="rag",
        expect_in_answer=["RAG_ANSWER"],
    ),
    # answer_mode 字段存在性
    Case(
        desc="[meta] answer_mode 字段存在且值为 structured",
        scene_key="book", scene_name="书籍",
        files=["X.epub"],
        route_to="book",
        query="共有几本书",
        expect_mode="structured",
        expect_in_answer=[],
    ),
    Case(
        desc="[meta] answer_mode 字段存在且值为 rag",
        scene_key="hr", scene_name="人力资源政策",
        files=["p.txt"],
        route_to="hr",
        query="入职流程是什么？",
        expect_mode="rag",
        expect_in_answer=[],
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_full_suite(repo_root: Path) -> None:
    mod = _load_engine(repo_root)
    passed = 0
    failed = 0
    for case in CASES:
        with tempfile.TemporaryDirectory(prefix="reg-") as tmp_dir:
            tmp = Path(tmp_dir)
            data_dir = tmp / case.scene_key
            data_dir.mkdir(parents=True, exist_ok=True)
            for fname in case.files:
                (data_dir / fname).write_text(fname, encoding="utf-8")

            extra_scenes: dict[str, dict] = {}
            if "book" not in case.extra_scenes and case.scene_key != "book":
                pass
            if case.scene_key != "book":
                book_dir = tmp / "book"
                book_dir.mkdir(exist_ok=True)
                (book_dir / "A.epub").write_text("A", encoding="utf-8")
                (book_dir / "B.epub").write_text("B", encoding="utf-8")
                extra_scenes["book"] = {"name": "书籍", "keywords": [], "path": str(book_dir)}

            fixture = SceneFixture(
                scene_key=case.scene_key,
                scene_name=case.scene_name,
                files=case.files,
                data_dir=data_dir,
                extra_scenes=extra_scenes,
            )
            engine = _make_engine(mod, tmp, fixture, route_to=case.route_to)

            try:
                result = engine.query_with_usage(case.query)
                answer = str(result.get("answer", ""))
                mode = str(result.get("answer_mode", ""))
                usage = result.get("usage", {})

                assert mode == case.expect_mode, (
                    f"answer_mode 期望 {case.expect_mode!r}，实际 {mode!r}"
                )
                for fragment in case.expect_in_answer:
                    assert fragment in answer, (
                        f"答案中缺少 {fragment!r}，实际: {answer[:200]!r}"
                    )
                if case.expect_mode == "structured":
                    assert int(usage.get("total_tokens", -1)) == 0, (
                        f"structured 分支 total_tokens 应为 0，实际 {usage}"
                    )
                print(f"  PASS  {case.desc}")
                passed += 1
            except AssertionError as exc:
                print(f"  FAIL  {case.desc}")
                print(f"        {exc}")
                failed += 1

    total = passed + failed
    print(f"\n{'='*55}")
    print(f"全量回归结果: {passed}/{total} passed  {'✅ ALL PASS' if failed == 0 else f'❌ {failed} FAILED'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    run_full_suite(repo_root)
