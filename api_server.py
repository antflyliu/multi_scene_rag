#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : api_server.py
@Author  : Kevin
@Date    : 2026/02/18
@Description : Phase 1 API 骨架.
@Version : 1.0
"""

import uuid
import json
import re
from threading import RLock, Thread
from typing import Any
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config_store import (
    get_api_key,
    get_app_config,
    get_base_url,
    get_llm_vendor,
    get_model_name,
    get_scenes,
    save_scenes,
    save_settings,
)

app = FastAPI(title="Multi-Scene RAG API", version="0.1.0")
WEB_DIR = Path(__file__).resolve().parent / "web"
STATIC_DIR = WEB_DIR / "static"

_engine_lock = RLock()
_engine_op_lock = RLock()
_engine_instance: Any = None
_rebuild_lock = RLock()
_rebuild_running = False
_active_rebuild_task_id = ""
_rebuild_tasks: dict[str, dict[str, Any]] = {}
_embedding_download_lock = RLock()
_embedding_download_tasks: dict[str, dict[str, Any]] = {}
_embedding_download_active_by_key: dict[str, str] = {}
EMBEDDING_REGISTRY_PATH = Path("./embedding/download_registry.json")
EMBEDDING_CAPABILITIES: dict[str, Any] = {
    "fields": {
        "embedding_provider": {
            "default": "dashscope",
            "customizable": True,
            "options": [
                {"value": "dashscope", "label": "dashscope", "description": "DashScope 向量服务"},
                {"value": "openai", "label": "openai", "description": "OpenAI 向量服务"},
                {"value": "local", "label": "local", "description": "本地 HuggingFace/ModelScope 模型"},
            ],
        },
        "embedding_source": {
            "default": "huggingface",
            "customizable": True,
            "options": [
                {"value": "huggingface", "label": "huggingface", "description": "从 HuggingFace 下载或缓存"},
                {"value": "modelscope", "label": "modelscope", "description": "从 ModelScope 下载或缓存"},
                {"value": "local", "label": "local", "description": "本地目录路径"},
            ],
        },
        "embedding_device": {
            "default": "cpu",
            "customizable": True,
            "options": [
                {"value": "cpu", "label": "cpu", "description": "CPU 运行"},
                {"value": "cuda", "label": "cuda", "description": "GPU 运行，不可用时自动回退 cpu"},
            ],
        },
    },
    "models_by_provider": {
        "dashscope": [
            {
                "value": "text-embedding-v1",
                "label": "text-embedding-v1",
                "description": "DashScope 通用文本向量模型",
            },
        ],
        "openai": [
            {
                "value": "text-embedding-3-small",
                "label": "text-embedding-3-small",
                "description": "OpenAI 轻量级向量模型",
            },
            {
                "value": "text-embedding-3-large",
                "label": "text-embedding-3-large",
                "description": "OpenAI 高精度向量模型",
            },
            {
                "value": "text-embedding-ada-002",
                "label": "text-embedding-ada-002",
                "description": "OpenAI 兼容老版本模型",
            },
        ],
        "local": [
            {
                "value": "BAAI/bge-m3",
                "label": "BAAI/bge-m3",
                "description": "推荐本地/离线多语种向量模型",
            },
            {
                "value": "BAAI/bge-large-zh-v1.5",
                "label": "BAAI/bge-large-zh-v1.5",
                "description": "中文检索常用模型",
            },
        ],
    },
}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class SettingsUpdateRequest(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    llm_vendor: str | None = None
    model_name: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_source: str | None = None
    embedding_device: str | None = None


class ScenesUpdateRequest(BaseModel):
    scenes: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")


class RebuildIndexRequest(BaseModel):
    scene_key: str = "all"


class EmbeddingModelStatusRequest(BaseModel):
    embedding_provider: str = "dashscope"
    embedding_source: str = "huggingface"
    embedding_model: str = ""


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^\w\-.]+", "__", model_id or "")


def _load_embedding_registry() -> dict[str, str]:
    if not EMBEDDING_REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(EMBEDDING_REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}
    except Exception:
        return {}


def _save_embedding_registry(registry: dict[str, str]) -> None:
    EMBEDDING_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDING_REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _embedding_registry_key(source: str, model: str) -> str:
    return f"{(source or '').strip().lower()}::{(model or '').strip()}"


def _is_non_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _check_local_embedding_status(source: str, model: str) -> dict[str, Any]:
    normalized_source = (source or "").strip().lower()
    model_value = (model or "").strip()
    if not model_value:
        return {
            "available": False,
            "needs_download": False,
            "resolved_path": "",
            "message": "embedding_model 不能为空。",
        }

    if normalized_source == "local":
        local_path = Path(model_value)
        ready = _is_non_empty_dir(local_path)
        return {
            "available": ready,
            "needs_download": False,
            "resolved_path": str(local_path),
            "message": "本地目录可用" if ready else "本地目录不存在或为空，请先准备模型文件。",
        }

    if normalized_source == "huggingface":
        target_dir = Path("./embedding/huggingface") / _sanitize_model_id(model_value)
        ready = _is_non_empty_dir(target_dir)
        return {
            "available": ready,
            "needs_download": not ready,
            "resolved_path": str(target_dir),
            "message": "模型已下载到本地" if ready else "模型未下载，请先下载模型。",
        }

    if normalized_source == "modelscope":
        registry = _load_embedding_registry()
        registry_path = registry.get(_embedding_registry_key(normalized_source, model_value), "")
        if registry_path:
            cached_path = Path(registry_path)
            ready = _is_non_empty_dir(cached_path)
            return {
                "available": ready,
                "needs_download": not ready,
                "resolved_path": str(cached_path),
                "message": "模型已下载到本地" if ready else "模型缓存目录不存在或为空，请重新下载。",
            }
        cache_root = Path("./embedding/modelscope")
        ready = _is_non_empty_dir(cache_root)
        return {
            "available": ready,
            "needs_download": not ready,
            "resolved_path": str(cache_root),
            "message": "检测到 ModelScope 缓存目录" if ready else "未检测到 ModelScope 缓存，请先下载模型。",
        }

    return {
        "available": False,
        "needs_download": False,
        "resolved_path": "",
        "message": f"不支持的 embedding_source: {normalized_source}",
    }


def _download_local_embedding_model(source: str, model: str) -> dict[str, Any]:
    normalized_source = (source or "").strip().lower()
    model_value = (model or "").strip()
    if not model_value:
        raise ValueError("embedding_model 不能为空。")

    if normalized_source == "local":
        raise ValueError("embedding_source=local 需使用本地路径，不支持自动下载。")

    if normalized_source == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError("请先安装 huggingface_hub：pip install huggingface_hub") from exc

        target_dir = Path("./embedding/huggingface") / _sanitize_model_id(model_value)
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_value,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        return {
            "resolved_path": str(target_dir),
            "message": "模型下载完成。",
        }

    if normalized_source == "modelscope":
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as exc:
            raise RuntimeError("请先安装 modelscope：pip install modelscope") from exc

        cache_dir = Path("./embedding/modelscope")
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = snapshot_download(model_id=model_value, cache_dir=str(cache_dir))
        registry = _load_embedding_registry()
        registry[_embedding_registry_key(normalized_source, model_value)] = str(downloaded_path)
        _save_embedding_registry(registry)
        return {
            "resolved_path": str(downloaded_path),
            "message": "模型下载完成。",
        }

    raise ValueError(f"不支持的 embedding_source: {normalized_source}")


def _embedding_download_key(source: str, model: str) -> str:
    return f"{(source or '').strip().lower()}::{(model or '').strip()}"


def _update_embedding_download_task(task_id: str, **kwargs: Any) -> None:
    with _embedding_download_lock:
        task = _embedding_download_tasks.get(task_id, {})
        task.update(kwargs)
        _embedding_download_tasks[task_id] = task


def _start_embedding_download_task(source: str, model: str) -> dict[str, Any]:
    normalized_source = (source or "").strip().lower()
    model_value = (model or "").strip()
    if not model_value:
        raise ValueError("embedding_model 不能为空。")
    if normalized_source == "local":
        raise ValueError("embedding_source=local 需使用本地路径，不支持自动下载。")

    task_key = _embedding_download_key(normalized_source, model_value)
    with _embedding_download_lock:
        active_task_id = _embedding_download_active_by_key.get(task_key, "")
        active_task = _embedding_download_tasks.get(active_task_id, {})
        if active_task and active_task.get("status") in {"queued", "running"}:
            return {
                "task_id": active_task_id,
                "status": str(active_task.get("status", "running")),
                "message": "已有相同模型下载任务在执行，已复用任务。",
            }

        task_id = uuid.uuid4().hex
        _embedding_download_tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "message": f"下载任务已创建：{normalized_source}/{model_value}",
            "source": normalized_source,
            "model": model_value,
            "resolved_path": "",
            "error": "",
        }
        _embedding_download_active_by_key[task_key] = task_id

    def _run() -> None:
        try:
            _update_embedding_download_task(
                task_id,
                status="running",
                progress=10,
                message="开始下载模型...",
            )
            result = _download_local_embedding_model(normalized_source, model_value)
            status_result = _check_local_embedding_status(normalized_source, model_value)
            _update_embedding_download_task(
                task_id,
                status="completed",
                progress=100,
                message=result.get("message", "模型下载完成。"),
                resolved_path=result.get("resolved_path", ""),
                available=bool(status_result.get("available", False)),
                needs_download=bool(status_result.get("needs_download", True)),
            )
        except Exception as exc:
            _update_embedding_download_task(
                task_id,
                status="failed",
                progress=100,
                message="模型下载失败。",
                error=str(exc),
            )
        finally:
            with _embedding_download_lock:
                current_task = _embedding_download_tasks.get(task_id, {})
                current_key = _embedding_download_key(
                    str(current_task.get("source", normalized_source)),
                    str(current_task.get("model", model_value)),
                )
                if _embedding_download_active_by_key.get(current_key) == task_id:
                    _embedding_download_active_by_key.pop(current_key, None)

    Thread(target=_run, daemon=True).start()
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "模型下载任务已启动。",
    }


def _reset_engine() -> None:
    global _engine_instance
    with _engine_lock:
        _engine_instance = None


def _get_engine(eager_init: bool = False) -> Any:
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            # 延迟导入，避免在只操作设置接口时触发重型依赖加载失败。
            from rag_engine import MultiSceneRAG
            _engine_instance = MultiSceneRAG(runtime_mode="web", eager_init=eager_init)
        return _engine_instance


def _set_rebuild_running(running: bool, task_id: str = "") -> None:
    global _rebuild_running, _active_rebuild_task_id
    with _rebuild_lock:
        _rebuild_running = running
        _active_rebuild_task_id = task_id if running else ""


def _is_rebuild_running() -> bool:
    with _rebuild_lock:
        return _rebuild_running


def _get_active_task_id() -> str:
    with _rebuild_lock:
        return _active_rebuild_task_id


def _update_task(task_key: str, **kwargs: Any) -> None:
    with _rebuild_lock:
        task = _rebuild_tasks.get(task_key, {})
        task.update(kwargs)
        _rebuild_tasks[task_key] = task


def _start_rebuild_task(scene_key: str) -> str:
    if _is_rebuild_running():
        raise RuntimeError(f"当前已有重建任务在运行: {_get_active_task_id()}")

    task_id = uuid.uuid4().hex
    _update_task(
        task_id,
        task_id=task_id,
        scene_key=scene_key,
        status="queued",
        progress=0,
        message=f"任务已创建，等待执行: {scene_key}",
        error="",
        stage="queued",
        current_scene="",
        scene_index=0,
        total_scenes=0,
        elapsed_seconds=0.0,
        changed_scene_count=0,
        changed_scenes=[],
    )
    _set_rebuild_running(True, task_id)

    def _run() -> None:
        def _emit_progress(event: dict[str, Any]) -> None:
            stage = str(event.get("stage", "") or "")
            scene_index = int(event.get("scene_index", 0) or 0)
            total_scenes = int(event.get("total_scenes", 0) or 0)
            current_scene = str(event.get("scene_key", "") or "")
            elapsed_seconds = float(event.get("elapsed_seconds", 0.0) or 0.0)
            event_message = str(event.get("message", "") or "重建进行中...")

            progress = 40
            if total_scenes > 0:
                safe_index = min(max(scene_index, 0), total_scenes)
                progress = 40 + int(55 * safe_index / total_scenes)
            elif stage in {"all_done", "scene_done"}:
                progress = 95

            if stage in {"all_done", "scene_done"}:
                progress = max(progress, 95)

            _update_task(
                task_id,
                progress=min(progress, 99),
                message=event_message,
                stage=stage,
                current_scene=current_scene,
                scene_index=scene_index,
                total_scenes=total_scenes,
                elapsed_seconds=elapsed_seconds,
            )

        try:
            _update_task(task_id, status="running", progress=10, message="开始重建索引...", stage="start")
            with _engine_op_lock:
                _update_task(task_id, progress=20, message="加载引擎（惰性模式）...", stage="engine_loading")
                engine = _get_engine(eager_init=False)
                if scene_key == "all":
                    _update_task(task_id, progress=40, message="正在检测并重建变更场景...", stage="all_start")
                    rebuild_summary = engine.rebuild_all_with_progress(on_progress=_emit_progress)
                    changed_scene_count = int(rebuild_summary.get("changed_scene_count", 0) or 0)
                    changed_scenes = list(rebuild_summary.get("changed_scenes", []) or [])
                    if changed_scene_count == 0:
                        final_message = "未检测到变更场景，已跳过重建"
                    else:
                        final_message = f"变更场景重建完成，共 {changed_scene_count} 个场景"
                    _update_task(
                        task_id,
                        status="completed",
                        progress=100,
                        message=final_message,
                        stage="completed",
                        changed_scene_count=changed_scene_count,
                        changed_scenes=changed_scenes,
                    )
                else:
                    _update_task(task_id, progress=40, message=f"正在增量重建场景: {scene_key}", stage="scene_start")
                    engine.rebuild_scene_with_progress(scene_key=scene_key, on_progress=_emit_progress)
                    _update_task(
                        task_id,
                        status="completed",
                        progress=100,
                        message=f"增量重建完成: {scene_key}",
                        stage="completed",
                        changed_scene_count=1,
                        changed_scenes=[scene_key],
                    )
        except Exception as exc:
            _update_task(task_id, status="failed", progress=100, message="重建失败", error=str(exc), stage="failed")
        finally:
            _set_rebuild_running(False)

    Thread(target=_run, daemon=True).start()
    return task_id


def _startup_load_runtime_config() -> None:
    """服务启动后自动加载场景设置，并按条件预热引擎。"""
    try:
        config = get_app_config()
        scenes = config.get("scenes", {})
        print(
            f"✅ Startup loaded scene settings: {len(scenes)} scene(s), "
            f"model={config.get('model_name', get_model_name(runtime_mode='web'))}"
        )
    except Exception as exc:
        print(f"⚠️ Startup failed to load scene settings: {exc}")
        return

    # 若未配置 API Key，则仅加载配置，不做引擎预热，避免启动失败。
    if not get_api_key(runtime_mode="web"):
        print("⚠️ DASHSCOPE_API_KEY 未配置，跳过引擎预热。")
        return

    def _warmup_engine() -> None:
        try:
            with _engine_op_lock:
                # 预热仅初始化模型配置，不触发全场景索引加载。
                _get_engine(eager_init=False)
            print("✅ Startup engine warmup completed.")
        except Exception as exc:
            print(f"⚠️ Startup engine warmup failed: {exc}")

    Thread(target=_warmup_engine, daemon=True).start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    _startup_load_runtime_config()


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在，请先创建 web/index.html")
    return FileResponse(str(index_path))


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    config = get_app_config()
    return {
        "api_key": _mask_api_key(str(config.get("api_key", ""))),
        "has_api_key": bool(config.get("api_key", "")),
        "base_url": get_base_url(runtime_mode="web"),
        "llm_vendor": str(config.get("llm_vendor", get_llm_vendor(runtime_mode="web"))),
        "model_name": str(config.get("model_name", "")),
        "embedding_provider": str(config.get("embedding_provider", "dashscope")),
        "embedding_model": str(config.get("embedding_model", "text-embedding-v1")),
        "embedding_source": str(config.get("embedding_source", "huggingface")),
        "embedding_device": str(config.get("embedding_device", "cpu")),
    }


@app.get("/api/embedding-models")
def list_embedding_models() -> dict[str, Any]:
    # 兼容旧前端：仅返回模型选项。
    return {
        "providers": EMBEDDING_CAPABILITIES.get("models_by_provider", {}),
        "fallback": EMBEDDING_CAPABILITIES.get("models_by_provider", {}).get("dashscope", []),
    }


@app.get("/api/embedding-capabilities")
def get_embedding_capabilities() -> dict[str, Any]:
    return EMBEDDING_CAPABILITIES


@app.post("/api/embedding/model-status")
def check_embedding_model_status(request: EmbeddingModelStatusRequest) -> dict[str, Any]:
    provider = (request.embedding_provider or "").strip().lower()
    source = (request.embedding_source or "").strip().lower()
    model = (request.embedding_model or "").strip()

    if provider != "local":
        return {
            "available": True,
            "needs_download": False,
            "resolved_path": "",
            "message": "当前 provider 非 local，无需本地模型检查。",
        }

    result = _check_local_embedding_status(source, model)
    return result


@app.post("/api/embedding/model-download")
def download_embedding_model(request: EmbeddingModelStatusRequest) -> dict[str, Any]:
    provider = (request.embedding_provider or "").strip().lower()
    source = (request.embedding_source or "").strip().lower()
    model = (request.embedding_model or "").strip()

    if provider != "local":
        raise HTTPException(status_code=400, detail="仅 embedding_provider=local 时支持自动下载。")

    try:
        return _start_embedding_download_task(source, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型下载失败: {exc}") from exc


@app.get("/api/embedding/model-download-status/{task_id}")
def embedding_model_download_status(task_id: str) -> dict[str, Any]:
    with _embedding_download_lock:
        task = _embedding_download_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"下载任务不存在: {task_id}")
        return {
            "task_id": task.get("task_id", task_id),
            "status": task.get("status", "unknown"),
            "progress": task.get("progress", 0),
            "message": task.get("message", ""),
            "source": task.get("source", ""),
            "model": task.get("model", ""),
            "resolved_path": task.get("resolved_path", ""),
            "error": task.get("error", ""),
            "available": bool(task.get("available", False)),
            "needs_download": bool(task.get("needs_download", True)),
        }


@app.put("/api/settings")
def update_settings(request: SettingsUpdateRequest) -> dict[str, Any]:
    previous = get_app_config()
    config = save_settings(
        api_key=request.api_key,
        base_url=request.base_url,
        llm_vendor=request.llm_vendor,
        model_name=request.model_name,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
        embedding_source=request.embedding_source,
        embedding_device=request.embedding_device,
    )
    _reset_engine()
    embedding_changed = (
        str(previous.get("embedding_provider", "")).lower() != str(config.get("embedding_provider", "")).lower()
        or str(previous.get("embedding_model", "")) != str(config.get("embedding_model", ""))
        or str(previous.get("embedding_source", "")).lower() != str(config.get("embedding_source", "")).lower()
        or str(previous.get("embedding_device", "")).lower() != str(config.get("embedding_device", "")).lower()
    )
    reminder = ""
    if embedding_changed:
        reminder = "Embedding 配置已变更，建议执行一次“重建索引（全量）”以避免检索结果不一致。"
    return {
        "message": "设置已保存",
        "api_key": _mask_api_key(str(config.get("api_key", ""))),
        "has_api_key": bool(config.get("api_key", "")),
        "base_url": get_base_url(runtime_mode="web"),
        "llm_vendor": str(config.get("llm_vendor", get_llm_vendor(runtime_mode="web"))),
        "model_name": str(config.get("model_name", "")),
        "embedding_provider": str(config.get("embedding_provider", "dashscope")),
        "embedding_model": str(config.get("embedding_model", "text-embedding-v1")),
        "embedding_source": str(config.get("embedding_source", "huggingface")),
        "embedding_device": str(config.get("embedding_device", "cpu")),
        "embedding_changed": embedding_changed,
        "reminder": reminder,
    }


@app.get("/api/scenes")
def list_scenes() -> dict[str, dict[str, Any]]:
    return get_scenes()


@app.put("/api/scenes")
def update_scenes(request: ScenesUpdateRequest) -> dict[str, Any]:
    saved = save_scenes(request.scenes)
    _reset_engine()
    return {
        "message": "场景配置已保存",
        "scene_count": len(saved.get("scenes", {})),
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    if _is_rebuild_running():
        raise HTTPException(
            status_code=409,
            detail=f"索引重建中，请稍后再试。task_id={_get_active_task_id()}",
        )

    try:
        with _engine_op_lock:
            engine = _get_engine(eager_init=False)
            result = engine.query_with_usage(request.query)
        return {
            "query": request.query,
            "answer": str(result.get("answer", "")),
            "answer_mode": str(result.get("answer_mode", "rag")),
            "routed_scene": str(result.get("routed_scene", "")),
            "answered_scene": str(result.get("answered_scene", "")),
            "usage": result.get("usage", {}),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc


@app.post("/api/rebuild-index")
def rebuild_index(request: RebuildIndexRequest) -> dict[str, str]:
    scene_key = request.scene_key.strip().lower() or "all"
    available_scenes = get_scenes()
    if scene_key != "all" and scene_key not in available_scenes:
        raise HTTPException(status_code=400, detail=f"无效场景: {scene_key}")

    try:
        task_id = _start_rebuild_task(scene_key)
        return {"task_id": task_id, "status": "queued", "message": f"重建任务已启动: {scene_key}"}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc


@app.get("/api/rebuild-status/{task_id}")
def rebuild_status(task_id: str) -> dict[str, Any]:
    with _rebuild_lock:
        task = _rebuild_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
        return {
            "task_id": task.get("task_id", task_id),
            "scene_key": task.get("scene_key", ""),
            "status": task.get("status", "unknown"),
            "progress": task.get("progress", 0),
            "message": task.get("message", ""),
            "error": task.get("error", ""),
            "stage": task.get("stage", ""),
            "current_scene": task.get("current_scene", ""),
            "scene_index": task.get("scene_index", 0),
            "total_scenes": task.get("total_scenes", 0),
            "elapsed_seconds": task.get("elapsed_seconds", 0.0),
            "changed_scene_count": task.get("changed_scene_count", 0),
            "changed_scenes": task.get("changed_scenes", []),
            "running": _rebuild_running,
        }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
