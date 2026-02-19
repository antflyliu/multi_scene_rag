#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : rag_engine.py
@Author  : Kevin
@Date    : 2025/10/26
@Description : 多场景RAG引擎.
@Version : 1.0
"""

import os
import shutil
import time
import json
import hashlib
from pathlib import Path
import re
from typing import Any, Callable
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from classifier import classify_scene
from config_store import (
    LLM_VENDOR_BASE_URLS,
    get_api_key,
    get_base_url,
    get_default_scene_key,
    get_embedding_device,
    get_embedding_model,
    get_embedding_provider,
    get_embedding_source,
    get_llm_vendor,
    get_model_name,
    get_scenes,
)

RebuildProgressCallback = Callable[[dict[str, Any]], None]
REBUILD_MANIFEST_PATH = Path("./storage/rebuild_manifest.json")
SCENE_CATALOG_PATH = Path("./storage/scene_catalog.json")

# 全局默认意图匹配模式——所有场景公用，优先级低于场景级自定义词典。
_DEFAULT_COUNT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(几|多少).*(本|个|条|份|篇|文件|文档|书|规则|政策|制度|流程|问题|项目|种|类)", re.IGNORECASE),
    re.compile(r"(有|共|总共).*(几|多少)", re.IGNORECASE),
    re.compile(r"(数量|总数|总计|共计)", re.IGNORECASE),
    re.compile(r"(有几本|有几个|有几条|有几份|有几类|有几种)", re.IGNORECASE),
]
_DEFAULT_LIST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(列出|罗列|枚举|清单|目录|有哪些|都有什么|全部|所有).*(书|文档|文件|资料|内容|规则|政策|流程|问题|项目)?", re.IGNORECASE),
    re.compile(r"(有哪些|包含哪些|涵盖哪些).*(文档|资料|内容|书|规则|政策|制度)?", re.IGNORECASE),
]

# 场景级自定义意图词典：scene_key -> {"count": [...], "list": [...]}
# config.py 中可通过 SCENE_INTENT_OVERRIDES 覆盖（可选）。
_BUILTIN_SCENE_INTENT_OVERRIDES: dict[str, dict[str, list[re.Pattern[str]]]] = {
    "book": {
        "count": [re.compile(r"(几本书|多少本书|书的数量|有几本书|共.*本书)", re.IGNORECASE)],
        "list": [re.compile(r"(列出.*书|书.*清单|有哪些书|所有书籍|书目|书籍.*目录)", re.IGNORECASE)],
    },
    "hr": {
        "count": [re.compile(r"(几条.*规定|几种.*假|几类.*福利|几个.*流程)", re.IGNORECASE)],
        "list": [re.compile(r"(列出.*政策|有哪些.*制度|所有.*流程|假期.*种类)", re.IGNORECASE)],
    },
    "finance": {
        "count": [re.compile(r"(几条.*报销|几类.*费用|几种.*发票|几个.*规则)", re.IGNORECASE)],
        "list": [re.compile(r"(列出.*报销|有哪些.*费用|所有.*发票类型)", re.IGNORECASE)],
    },
    "it": {
        "count": [re.compile(r"(几类.*问题|几种.*故障|几个.*账号|几套.*系统)", re.IGNORECASE)],
        "list": [re.compile(r"(列出.*软件|有哪些.*系统|所有.*账号|支持.*哪些.*设备)", re.IGNORECASE)],
    },
}


def _compile_scene_intent_overrides() -> dict[str, dict[str, list[re.Pattern[str]]]]:
    """合并内置场景词典与 config 中的可选扩展。"""
    try:
        from config import SCENE_INTENT_OVERRIDES as _extra  # type: ignore[import]
        merged: dict[str, dict[str, list[re.Pattern[str]]]] = {}
        all_keys = set(_BUILTIN_SCENE_INTENT_OVERRIDES) | set(_extra)
        for key in all_keys:
            merged[key] = {}
            for intent in ("count", "list"):
                builtin_pats = _BUILTIN_SCENE_INTENT_OVERRIDES.get(key, {}).get(intent, [])
                extra_pats = _extra.get(key, {}).get(intent, [])
                merged[key][intent] = builtin_pats + [
                    re.compile(p, re.IGNORECASE) if isinstance(p, str) else p
                    for p in extra_pats
                ]
        return merged
    except (ImportError, AttributeError):
        return _BUILTIN_SCENE_INTENT_OVERRIDES


SCENE_INTENT_OVERRIDES: dict[str, dict[str, list[re.Pattern[str]]]] = _compile_scene_intent_overrides()


def configure_runtime_models(runtime_mode: str = "web") -> None:
    """按运行时配置设置全局 LLM 与 Embedding 模型。

    当前策略：
    - LLM: 按厂商路由（dashscope 走原生 SDK，其它走 OpenAI 兼容接口）
    - Embedding: DashScope（保持原有向量构建链路稳定）
    """
    api_key = get_api_key(runtime_mode=runtime_mode) or os.getenv("DASHSCOPE_API_KEY", "")
    llm_vendor = get_llm_vendor(runtime_mode=runtime_mode)
    base_url = get_base_url(runtime_mode=runtime_mode)
    model_name = get_model_name(runtime_mode=runtime_mode)
    embedding_provider = get_embedding_provider(runtime_mode=runtime_mode)
    embedding_model = get_embedding_model(runtime_mode=runtime_mode)
    embedding_source = get_embedding_source(runtime_mode=runtime_mode)
    embedding_device = _resolve_embedding_device(get_embedding_device(runtime_mode=runtime_mode))
    if not api_key:
        raise ValueError("API_KEY 未配置，请先在设置页保存 API Key。")

    openai_compatible_vendors = {vendor for vendor in LLM_VENDOR_BASE_URLS.keys() if vendor != "dashscope"}

    # max_tokens 需小于 context_window，否则 prompt_helper 计算 available_context 会为负导致 ValueError。
    # QWEN_MAX context=8192，需预留空间给 RAG 检索块+模板，4096 既能保证长答案完整又避免 -125 类错误。
    _max_tokens = 4096

    if llm_vendor == "dashscope":
        Settings.llm = DashScope(
            api_key=api_key,
            model_name=model_name or DashScopeGenerationModels.QWEN_MAX,
            max_tokens=_max_tokens,
        )
    elif llm_vendor in openai_compatible_vendors:
        Settings.llm = LlamaOpenAI(
            api_key=api_key,
            api_base=base_url,
            model=model_name,
            max_tokens=_max_tokens,
        )
    else:
        raise ValueError(
            f"不支持的 llm_vendor: {llm_vendor}。"
            "请使用 dashscope、openai、claude、gemini、glm、kimi 或 custom。"
        )
    if embedding_provider == "dashscope":
        Settings.embed_model = DashScopeEmbedding(
            api_key=api_key,
            model_name=embedding_model or DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        )
    elif embedding_provider == "openai":
        Settings.embed_model = OpenAIEmbedding(
            api_key=api_key,
            api_base=base_url,
            model=embedding_model,
        )
    elif embedding_provider == "local":
        local_model_path_or_id = _resolve_local_embedding_model(embedding_source, embedding_model)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=local_model_path_or_id,
            device=embedding_device,
        )
    else:
        raise ValueError(
            f"不支持的 embedding_provider: {embedding_provider}。"
            "请使用 dashscope、openai 或 local。"
        )


def _resolve_embedding_device(config_device: str) -> str:
    normalized = (config_device or "cpu").lower()
    if normalized not in {"cpu", "cuda"}:
        return "cpu"
    if normalized == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            print("⚠️ embedding_device=cuda 但未检测到可用 CUDA，自动回退到 cpu。")
            return "cpu"
        except Exception:
            print("⚠️ 当前环境缺少 torch/CUDA 支持，embedding_device 自动回退到 cpu。")
            return "cpu"
    return "cpu"


def _resolve_local_embedding_model(source: str, embedding_model: str) -> str:
    normalized_source = (source or "huggingface").lower()
    model_value = (embedding_model or "").strip()
    if not model_value:
        raise ValueError("embedding_model 不能为空。")

    if normalized_source == "huggingface":
        return _resolve_huggingface_model_path(model_value)

    if normalized_source == "local":
        model_path = Path(model_value)
        if not model_path.exists():
            raise FileNotFoundError(f"本地 embedding 模型路径不存在: {model_value}")
        return str(model_path)

    if normalized_source == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "使用 modelscope 需要先安装 modelscope：pip install modelscope"
            ) from exc
        cache_dir = Path("./embedding/modelscope")
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = snapshot_download(model_id=model_value, cache_dir=str(cache_dir))
        return str(downloaded_path)

    raise ValueError(
        f"不支持的 embedding_source: {normalized_source}。"
        "请使用 huggingface、modelscope 或 local。"
    )


def _sanitize_model_id(model_id: str) -> str:
    """将模型 ID 转为可作为目录名的安全字符串。"""
    return re.sub(r"[^\w\-.]+", "__", model_id)


def _resolve_huggingface_model_path(model_id: str) -> str:
    """将 HuggingFace 模型下载/缓存到 embedding/huggingface。"""
    target_root = Path("./embedding/huggingface")
    target_root.mkdir(parents=True, exist_ok=True)
    model_dir = target_root / _sanitize_model_id(model_id)

    # 若目录已存在并且非空，视为已缓存，直接复用。
    if model_dir.exists() and any(model_dir.iterdir()):
        return str(model_dir)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "使用 huggingface 作为 embedding_source 需要安装 huggingface_hub。"
        ) from exc

    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    return str(model_dir)

class MultiSceneRAG:
    def __init__(self, runtime_mode: str = "web", eager_init: bool = True):
        self.runtime_mode = runtime_mode
        configure_runtime_models(runtime_mode=runtime_mode)
        self._token_counter: TokenCountingHandler | None = None
        try:
            self._token_counter = TokenCountingHandler()
            Settings.callback_manager = CallbackManager([self._token_counter])
        except Exception:
            # 兼容不同 llama-index 版本；不可用时回退为不统计 token。
            self._token_counter = None
        self.scenes = get_scenes()
        self.indices = {}
        self._eager_init = bool(eager_init)
        if self._eager_init:
            self._init_indices()

    @staticmethod
    def _read_rebuild_manifest() -> dict[str, Any]:
        if not REBUILD_MANIFEST_PATH.exists():
            return {"scenes": {}}
        try:
            data = json.loads(REBUILD_MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                scenes_data = data.get("scenes", {})
                if isinstance(scenes_data, dict):
                    return {"scenes": scenes_data}
        except Exception:
            pass
        return {"scenes": {}}

    @staticmethod
    def _write_rebuild_manifest(data: dict[str, Any]) -> None:
        REBUILD_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        REBUILD_MANIFEST_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _read_scene_catalog() -> dict[str, Any]:
        if not SCENE_CATALOG_PATH.exists():
            return {"scenes": {}}
        try:
            data = json.loads(SCENE_CATALOG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("scenes"), dict):
                return data
        except Exception:
            pass
        return {"scenes": {}}

    @staticmethod
    def _write_scene_catalog(catalog: dict[str, Any]) -> None:
        SCENE_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCENE_CATALOG_PATH.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_embedding_signature(self) -> dict[str, str]:
        return {
            "embedding_provider": get_embedding_provider(runtime_mode=self.runtime_mode),
            "embedding_model": get_embedding_model(runtime_mode=self.runtime_mode),
            "embedding_source": get_embedding_source(runtime_mode=self.runtime_mode),
            "embedding_device": get_embedding_device(runtime_mode=self.runtime_mode),
        }

    @staticmethod
    def _build_data_signature(data_path: str) -> dict[str, Any]:
        root = Path(data_path)
        if not root.exists():
            return {"exists": False, "files": [], "file_count": 0, "total_size": 0}
        file_entries: list[dict[str, Any]] = []
        total_size = 0
        for file_path in sorted([p for p in root.rglob("*") if p.is_file()]):
            stat = file_path.stat()
            relative_path = file_path.relative_to(root).as_posix()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
            total_size += size
            file_entries.append(
                {
                    "path": relative_path,
                    "size": size,
                    "mtime_ns": mtime_ns,
                }
            )
        return {
            "exists": True,
            "files": file_entries,
            "file_count": len(file_entries),
            "total_size": total_size,
        }

    def _scene_rebuild_signature(self, scene_key: str, scene_info: dict[str, Any]) -> str:
        signature_payload = {
            "scene_key": scene_key,
            "scene_config": {
                "name": scene_info.get("name", ""),
                "keywords": scene_info.get("keywords", []),
                "path": scene_info.get("path", ""),
            },
            "embedding": self._get_embedding_signature(),
            "data": self._build_data_signature(str(scene_info.get("path", ""))),
        }
        raw = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _scene_file_entries(data_path: str) -> list[dict[str, Any]]:
        root = Path(data_path)
        if not root.exists():
            return []
        entries: list[dict[str, Any]] = []
        for file_path in sorted([p for p in root.rglob("*") if p.is_file()]):
            stat = file_path.stat()
            entries.append(
                {
                    "file_name": file_path.name,
                    "file_path": str(file_path.resolve()),
                    "relative_path": file_path.relative_to(root).as_posix(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        return entries

    def _update_scene_catalog(self, scene_key: str) -> None:
        scene_info = self.scenes.get(scene_key)
        if not scene_info:
            return
        data_path = str(scene_info.get("path", ""))
        files = self._scene_file_entries(data_path)
        data_signature = self._scene_rebuild_signature(scene_key, scene_info)
        catalog = self._read_scene_catalog()
        scenes_data = catalog.setdefault("scenes", {})
        scenes_data[scene_key] = {
            "scene_name": str(scene_info.get("name", scene_key)),
            "data_path": data_path,
            "data_signature": data_signature,
            "updated_at": int(time.time()),
            "files": files,
        }
        self._write_scene_catalog(catalog)

    def _get_scene_catalog_files(self, scene_key: str) -> list[dict[str, Any]]:
        scene_info = self.scenes.get(scene_key, {})
        current_signature = self._scene_rebuild_signature(scene_key, scene_info)
        catalog = self._read_scene_catalog()
        scene_data = catalog.get("scenes", {}).get(scene_key, {})
        if (
            isinstance(scene_data, dict)
            and scene_data.get("data_signature") == current_signature
            and isinstance(scene_data.get("files"), list)
        ):
            return [f for f in scene_data.get("files", []) if isinstance(f, dict)]
        # 目录或配置变化后自动回源刷新，避免统计答案滞后。
        self._update_scene_catalog(scene_key)
        refreshed = self._read_scene_catalog().get("scenes", {}).get(scene_key, {})
        files = refreshed.get("files", []) if isinstance(refreshed, dict) else []
        return [f for f in files if isinstance(f, dict)]

    def _detect_query_intent(self, user_query: str, scene_key: str = "") -> str:
        """识别查询意图：count / list / qa。

        优先匹配场景级词典，其次使用全局默认词典。
        """
        text = (user_query or "").strip()
        scene_overrides = SCENE_INTENT_OVERRIDES.get(scene_key, {})
        for pattern in scene_overrides.get("count", []):
            if pattern.search(text):
                return "count"
        for pattern in scene_overrides.get("list", []):
            if pattern.search(text):
                return "list"
        for pattern in _DEFAULT_COUNT_PATTERNS:
            if pattern.search(text):
                return "count"
        for pattern in _DEFAULT_LIST_PATTERNS:
            if pattern.search(text):
                return "list"
        return "qa"

    def _build_structured_answer(self, scene_key: str, user_query: str) -> str | None:
        intent = self._detect_query_intent(user_query, scene_key=scene_key)
        if intent not in {"count", "list"}:
            return None
        scene_info = self.scenes.get(scene_key)
        if not scene_info:
            return None
        files = self._get_scene_catalog_files(scene_key)
        scene_name = str(scene_info.get("name", scene_key))
        count = len(files)
        if intent == "count":
            sample_names = [str(item.get("file_name", "")) for item in files[:3]]
            if count == 0:
                return f"当前场景“{scene_name}”暂无文档。"
            if sample_names:
                return (
                    f"当前场景“{scene_name}”共有 {count} 份文档。"
                    f"例如：{'; '.join(sample_names)}。"
                )
            return f"当前场景“{scene_name}”共有 {count} 份文档。"

        if count == 0:
            return f"当前场景“{scene_name}”暂无可列出的文档。"
        lines = [f"当前场景“{scene_name}”共有 {count} 份文档："]
        for idx, item in enumerate(files, start=1):
            lines.append(f"{idx}. {item.get('file_name', '未命名文档')}")
            if idx >= 30:
                lines.append(f"... 其余 {count - 30} 份文档已省略。")
                break
        return "\n".join(lines)

    def _resolve_scene_for_structured_query(self, routed_scene: str, user_query: str) -> str:
        intent = self._detect_query_intent(user_query, scene_key=routed_scene)
        if intent not in {"count", "list"}:
            return routed_scene
        text = (user_query or "").strip().lower()
        matched_scenes: list[str] = []
        for scene_key, scene_info in self.scenes.items():
            name = str(scene_info.get("name", "")).strip().lower()
            if scene_key.lower() in text or (name and name in text):
                matched_scenes.append(scene_key)
        if len(matched_scenes) == 1:
            return matched_scenes[0]
        if "book" in self.scenes and ("书" in user_query or "epub" in text):
            return "book"
        return routed_scene

    def _collect_changed_scenes(self) -> tuple[list[str], list[dict[str, str]]]:
        manifest = self._read_rebuild_manifest()
        scene_signatures = manifest.get("scenes", {})
        changed: list[str] = []
        reasons: list[dict[str, str]] = []
        for scene_key, scene_info in self.scenes.items():
            current_signature = self._scene_rebuild_signature(scene_key, scene_info)
            previous_signature = str(scene_signatures.get(scene_key, ""))
            persist_dir = Path(f"./storage/{scene_key}")
            docstore_path = persist_dir / "docstore.json"
            if not previous_signature:
                changed.append(scene_key)
                reasons.append({"scene_key": scene_key, "reason": "首次构建或缺少历史签名"})
                continue
            if not persist_dir.exists() or not docstore_path.exists():
                changed.append(scene_key)
                reasons.append({"scene_key": scene_key, "reason": "索引文件缺失"})
                continue
            if current_signature != previous_signature:
                changed.append(scene_key)
                reasons.append({"scene_key": scene_key, "reason": "数据或配置发生变化"})
        return changed, reasons

    def _update_scene_manifest(self, scene_key: str) -> None:
        scene_info = self.scenes.get(scene_key)
        if not scene_info:
            return
        manifest = self._read_rebuild_manifest()
        scenes_data = manifest.setdefault("scenes", {})
        scenes_data[scene_key] = self._scene_rebuild_signature(scene_key, scene_info)
        self._write_rebuild_manifest(manifest)
        self._update_scene_catalog(scene_key)

    def _init_indices(self):
        """为每个场景构建或加载向量索引"""
        for scene, info in self.scenes.items():
            self._load_or_build_scene(scene=scene, info=info, force_rebuild=False)

    def _get_scene_storage_context(self, scene: str):
        client = chromadb.PersistentClient(path="./storage/chroma_db")
        collection = client.get_or_create_collection(f"scene_{scene}")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return client, vector_store, storage_context

    def _load_or_build_scene(self, scene: str, info: dict, force_rebuild: bool = False) -> None:
        """加载或构建单个场景索引。"""
        started = time.perf_counter()
        print(f"Loading index for scene: {scene}")
        client, vector_store, storage_context = self._get_scene_storage_context(scene)
        persist_dir = f"./storage/{scene}"
        docstore_path = os.path.join(persist_dir, "docstore.json")

        if force_rebuild:
            # 增量重建时先清理场景存储，避免重复写入向量库。
            try:
                client.delete_collection(f"scene_{scene}")
            except Exception:
                pass
            shutil.rmtree(persist_dir, ignore_errors=True)
            _, _, storage_context = self._get_scene_storage_context(scene)
            print(f"🔄 Rebuilding index for scene: {scene}")
            index = self._build_new_index(info["path"], storage_context, persist_dir)
            self.indices[scene] = index.as_query_engine()
            elapsed = time.perf_counter() - started
            print(f"✅ Scene {scene} rebuild finished in {elapsed:.2f}s")
            return

        if os.path.exists(persist_dir) and os.path.exists(docstore_path):
            try:
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=persist_dir
                )
                index = load_index_from_storage(storage_context)
                print(f"✅ Successfully loaded existing index for scene: {scene}")
            except Exception as e:
                print(f"⚠️  Failed to load existing index for scene {scene}: {e}")
                print(f"🔄 Rebuilding index for scene: {scene}")
                index = self._build_new_index(info["path"], storage_context, persist_dir)
        else:
            print(f"🔄 Building new index for scene: {scene}")
            index = self._build_new_index(info["path"], storage_context, persist_dir)

        self.indices[scene] = index.as_query_engine()
        elapsed = time.perf_counter() - started
        print(f"✅ Scene {scene} load/build finished in {elapsed:.2f}s")

    def _ensure_scene_index(self, scene: str) -> None:
        """按需确保场景索引可用，避免引擎初始化时全量加载。"""
        if scene in self.indices:
            return
        scene_info = self.scenes.get(scene)
        if not scene_info:
            raise KeyError(f"无效场景: {scene}")
        self._load_or_build_scene(scene=scene, info=scene_info, force_rebuild=False)

    def _build_new_index(self, data_path, storage_context, persist_dir):
        """构建新的索引"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        documents = SimpleDirectoryReader(data_path).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )

        # 确保持久化目录存在
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"✅ Successfully built and persisted index for scene")

        return index

    def rebuild_scene(self, scene_key: str) -> None:
        """增量重建指定场景索引。"""
        self.rebuild_scene_with_progress(scene_key=scene_key, on_progress=None)

    def rebuild_scene_with_progress(
        self,
        scene_key: str,
        on_progress: RebuildProgressCallback | None = None,
    ) -> None:
        """增量重建指定场景索引（支持进度回调）。"""
        self.scenes = get_scenes()
        if scene_key not in self.scenes:
            raise KeyError(f"无效场景: {scene_key}")
        self._emit_rebuild_progress(
            on_progress=on_progress,
            stage="scene_start",
            scene_key=scene_key,
            scene_index=1,
            total_scenes=1,
            message=f"开始重建场景: {scene_key}",
        )
        started = time.perf_counter()
        self._load_or_build_scene(scene=scene_key, info=self.scenes[scene_key], force_rebuild=True)
        self._update_scene_manifest(scene_key)
        elapsed_seconds = time.perf_counter() - started
        self._emit_rebuild_progress(
            on_progress=on_progress,
            stage="scene_done",
            scene_key=scene_key,
            scene_index=1,
            total_scenes=1,
            elapsed_seconds=elapsed_seconds,
            message=f"场景重建完成: {scene_key}（{elapsed_seconds:.2f}s）",
        )

    def rebuild_all(self) -> None:
        """重建发生变化的场景索引。"""
        self.rebuild_all_with_progress(on_progress=None)

    def rebuild_all_with_progress(self, on_progress: RebuildProgressCallback | None = None) -> dict[str, Any]:
        """仅重建发生变化的场景索引（支持进度回调）。"""
        self.scenes = get_scenes()
        for key in list(self.indices.keys()):
            if key not in self.scenes:
                self.indices.pop(key, None)
        changed_scenes, changed_reasons = self._collect_changed_scenes()
        scene_items = [(scene_key, self.scenes[scene_key]) for scene_key in changed_scenes]
        total_scenes = len(scene_items)
        if total_scenes == 0:
            self._emit_rebuild_progress(
                on_progress=on_progress,
                stage="all_done",
                scene_index=0,
                total_scenes=0,
                message="未检测到变更场景，已跳过重建",
            )
            return {
                "changed_scene_count": 0,
                "changed_scenes": [],
                "changed_reasons": changed_reasons,
                "skipped": True,
            }
        reason_text = "；".join(
            [f"{item['scene_key']}({item['reason']})" for item in changed_reasons if item.get("scene_key") in changed_scenes]
        )
        self._emit_rebuild_progress(
            on_progress=on_progress,
            stage="all_start",
            scene_index=0,
            total_scenes=total_scenes,
            message=f"检测到 {total_scenes} 个变更场景，开始重建：{reason_text}",
        )
        for idx, (scene, info) in enumerate(scene_items, start=1):
            self._emit_rebuild_progress(
                on_progress=on_progress,
                stage="scene_start",
                scene_key=scene,
                scene_index=idx,
                total_scenes=total_scenes,
                message=f"正在重建场景 {idx}/{total_scenes}: {scene}",
            )
            started = time.perf_counter()
            self._load_or_build_scene(scene=scene, info=info, force_rebuild=True)
            self._update_scene_manifest(scene)
            elapsed_seconds = time.perf_counter() - started
            self._emit_rebuild_progress(
                on_progress=on_progress,
                stage="scene_done",
                scene_key=scene,
                scene_index=idx,
                total_scenes=total_scenes,
                elapsed_seconds=elapsed_seconds,
                message=f"场景重建完成 {idx}/{total_scenes}: {scene}（{elapsed_seconds:.2f}s）",
            )
        self._emit_rebuild_progress(
            on_progress=on_progress,
            stage="all_done",
            scene_index=total_scenes,
            total_scenes=total_scenes,
            message=f"变更场景重建完成，共 {total_scenes} 个场景",
        )
        return {
            "changed_scene_count": total_scenes,
            "changed_scenes": changed_scenes,
            "changed_reasons": changed_reasons,
            "skipped": False,
        }

    @staticmethod
    def _emit_rebuild_progress(
        on_progress: RebuildProgressCallback | None,
        **payload: Any,
    ) -> None:
        if on_progress is None:
            return
        try:
            on_progress(payload)
        except Exception:
            # 进度回调不应影响重建主流程。
            pass

    def _snapshot_token_counts(self) -> dict[str, int]:
        if self._token_counter is None:
            return {
                "prompt_llm_tokens": 0,
                "completion_llm_tokens": 0,
                "total_llm_tokens": 0,
                "total_embedding_tokens": 0,
            }
        return {
            "prompt_llm_tokens": int(getattr(self._token_counter, "prompt_llm_token_count", 0) or 0),
            "completion_llm_tokens": int(getattr(self._token_counter, "completion_llm_token_count", 0) or 0),
            "total_llm_tokens": int(getattr(self._token_counter, "total_llm_token_count", 0) or 0),
            "total_embedding_tokens": int(getattr(self._token_counter, "total_embedding_token_count", 0) or 0),
        }

    @staticmethod
    def _calc_usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        input_tokens = max(after["prompt_llm_tokens"] - before["prompt_llm_tokens"], 0)
        output_tokens = max(after["completion_llm_tokens"] - before["completion_llm_tokens"], 0)
        total_tokens = max(after["total_llm_tokens"] - before["total_llm_tokens"], input_tokens + output_tokens)
        embedding_tokens = max(
            after["total_embedding_tokens"] - before["total_embedding_tokens"],
            0,
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "embedding_tokens": embedding_tokens,
        }

    def query_with_usage(self, user_query: str) -> dict[str, Any]:
        scene = classify_scene(user_query, runtime_mode=self.runtime_mode)
        scene_info = self.scenes.get(scene)
        if not scene_info:
            fallback_scene = get_default_scene_key()
            if fallback_scene not in self.scenes:
                fallback_scene = next(iter(self.scenes), "")
            scene = fallback_scene
            scene_info = self.scenes.get(scene, {})
            print(f"⚠️ 未命中有效场景，回退到: {scene_info.get('name', scene)} ({scene})")
        else:
            print(f"🔍 路由到场景: {scene_info['name']} ({scene})")

        routed_scene = scene
        scene = self._resolve_scene_for_structured_query(scene, user_query)
        scene_info = self.scenes.get(scene, scene_info)
        structured_answer = self._build_structured_answer(scene, user_query)
        if structured_answer:
            print(f"📊 结构化分流: intent=count/list, scene={scene}")
            return {
                "answer": structured_answer,
                "answer_mode": "structured",
                "routed_scene": routed_scene,
                "answered_scene": scene,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "embedding_tokens": 0,
                },
            }
        self._ensure_scene_index(scene)
        before = self._snapshot_token_counts()
        query_engine = self.indices[scene]
        response = query_engine.query(user_query)
        after = self._snapshot_token_counts()
        return {
            "answer": str(response),
            "answer_mode": "rag",
            "routed_scene": routed_scene,
            "answered_scene": scene,
            "usage": self._calc_usage_delta(before, after),
        }

    def query(self, user_query: str) -> str:
        return str(self.query_with_usage(user_query).get("answer", ""))
