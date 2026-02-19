#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : config_store.py
@Author  : Kevin
@Date    : 2026/02/18
@Description : 应用配置持久化管理.
@Version : 1.0
"""

import copy
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from config import DEFAULT_SCENES

# DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_VENDOR = "dashscope"
LLM_VENDOR_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
    "claude": "",
    "gemini": "",
    "glm": "",
    "kimi": "",
    "deepseek": "https://api.deepseek.com/v1",
    "custom": "",
}
DEFAULT_MODEL_NAME = "qwen-flash"
DEFAULT_EMBEDDING_PROVIDER = "dashscope"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v1"
DEFAULT_EMBEDDING_SOURCE = "huggingface"
DEFAULT_EMBEDDING_DEVICE = "cpu"
CONFIG_FILE_PATH = Path(__file__).resolve().parent / "app_config.json"
ENV_CANDIDATES = {
    # 通用命名优先，其次兼容不同厂商，避免配置被单一供应商变量名绑定。
    "api_key": [
        "MSRAG_API_KEY",
        "OPENAI_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
    ],
    "base_url": [
        "MSRAG_BASE_URL",
        "OPENAI_BASE_URL",
        "DASHSCOPE_BASE_URL",
        "DEEPSEEK_BASE_URL",
    ],
    "llm_vendor": [
        "MSRAG_LLM_VENDOR",
    ],
    "model_name": [
        "MSRAG_MODEL_NAME",
        "OPENAI_MODEL",
        "DASHSCOPE_MODEL_NAME",
        "DEEPSEEK_MODEL",
    ],
    "embedding_provider": [
        "MSRAG_EMBEDDING_PROVIDER",
    ],
    "embedding_model": [
        "MSRAG_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "DASHSCOPE_EMBEDDING_MODEL",
    ],
    "embedding_source": [
        "MSRAG_EMBEDDING_SOURCE",
    ],
    "embedding_device": [
        "MSRAG_EMBEDDING_DEVICE",
    ],
}


def _normalize_scenes(raw_scenes: Any) -> dict[str, dict[str, Any]]:
    """校验并标准化场景配置。"""
    if not isinstance(raw_scenes, dict) or not raw_scenes:
        return copy.deepcopy(DEFAULT_SCENES)

    normalized: dict[str, dict[str, Any]] = {}
    for scene_key, scene_info in raw_scenes.items():
        if not isinstance(scene_key, str) or not scene_key.strip():
            continue
        if not isinstance(scene_info, dict):
            continue

        name = scene_info.get("name")
        keywords = scene_info.get("keywords")
        path = scene_info.get("path")

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(path, str) or not path.strip():
            continue

        keyword_list = keywords if isinstance(keywords, list) else []
        keyword_list = [kw for kw in keyword_list if isinstance(kw, str) and kw.strip()]

        normalized[scene_key.strip().lower()] = {
            "name": name.strip(),
            "keywords": keyword_list,
            "path": path.strip(),
        }

    return normalized or copy.deepcopy(DEFAULT_SCENES)


class ConfigStore:
    """应用配置读写器（线程安全）。"""

    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        self._config_path = config_path
        self._lock = RLock()

    def _default_config(self) -> dict[str, Any]:
        env_api_key = _get_env_value("api_key", default="")
        env_base_url = _get_env_value("base_url", default=DEFAULT_BASE_URL)
        env_llm_vendor = _get_env_value("llm_vendor", default=DEFAULT_LLM_VENDOR)
        env_model_name = _get_env_value("model_name", default=DEFAULT_MODEL_NAME)
        env_embedding_provider = _get_env_value("embedding_provider", default=DEFAULT_EMBEDDING_PROVIDER)
        env_embedding_model = _get_env_value("embedding_model", default=DEFAULT_EMBEDDING_MODEL)
        env_embedding_source = _get_env_value("embedding_source", default=DEFAULT_EMBEDDING_SOURCE)
        env_embedding_device = _get_env_value("embedding_device", default=DEFAULT_EMBEDDING_DEVICE)
        return {
            "api_key": env_api_key,
            "base_url": env_base_url,
            "llm_vendor": env_llm_vendor,
            "model_name": env_model_name,
            "embedding_provider": env_embedding_provider,
            "embedding_model": env_embedding_model,
            "embedding_source": env_embedding_source,
            "embedding_device": env_embedding_device,
            "scenes": copy.deepcopy(DEFAULT_SCENES),
        }

    def load(self) -> dict[str, Any]:
        """加载配置，不存在时创建默认配置文件。"""
        with self._lock:
            default_data = self._default_config()
            if not self._config_path.exists():
                self._save(default_data)
                return default_data

            try:
                file_data = json.loads(self._config_path.read_text(encoding="utf-8"))
                merged_data = {
                    "api_key": str(file_data.get("api_key", default_data["api_key"])),
                    "base_url": str(file_data.get("base_url", default_data["base_url"])),
                    "llm_vendor": str(file_data.get("llm_vendor", default_data["llm_vendor"])).lower(),
                    "model_name": str(file_data.get("model_name", default_data["model_name"])),
                    "embedding_provider": str(
                        file_data.get("embedding_provider", default_data["embedding_provider"])
                    ).lower(),
                    "embedding_model": str(file_data.get("embedding_model", default_data["embedding_model"])),
                    "embedding_source": str(
                        file_data.get("embedding_source", default_data["embedding_source"])
                    ).lower(),
                    "embedding_device": str(
                        file_data.get("embedding_device", default_data["embedding_device"])
                    ).lower(),
                    "scenes": _normalize_scenes(file_data.get("scenes", default_data["scenes"])),
                }
                return merged_data
            except Exception:
                # 读取失败时回退默认配置，避免应用无法启动。
                self._save(default_data)
                return default_data

    def _save(self, data: dict[str, Any]) -> None:
        self._config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        """保存完整配置并返回标准化后的结果。"""
        with self._lock:
            normalized = {
                "api_key": str(data.get("api_key", "")),
                "base_url": str(data.get("base_url", DEFAULT_BASE_URL)),
                "llm_vendor": str(data.get("llm_vendor", DEFAULT_LLM_VENDOR)).lower(),
                "model_name": str(data.get("model_name", DEFAULT_MODEL_NAME)),
                "embedding_provider": str(
                    data.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
                ).lower(),
                "embedding_model": str(data.get("embedding_model", DEFAULT_EMBEDDING_MODEL)),
                "embedding_source": str(
                    data.get("embedding_source", DEFAULT_EMBEDDING_SOURCE)
                ).lower(),
                "embedding_device": str(
                    data.get("embedding_device", DEFAULT_EMBEDDING_DEVICE)
                ).lower(),
                "scenes": _normalize_scenes(data.get("scenes", {})),
            }
            self._save(normalized)
            return normalized

    def update_settings(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        llm_vendor: str | None = None,
        model_name: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_source: str | None = None,
        embedding_device: str | None = None,
    ) -> dict[str, Any]:
        """更新 API 相关配置。"""
        with self._lock:
            current = self.load()
            if api_key is not None:
                current["api_key"] = api_key
            if base_url is not None:
                current["base_url"] = base_url
            if llm_vendor is not None:
                current["llm_vendor"] = llm_vendor
            if model_name is not None:
                current["model_name"] = model_name
            if embedding_provider is not None:
                current["embedding_provider"] = embedding_provider
            if embedding_model is not None:
                current["embedding_model"] = embedding_model
            if embedding_source is not None:
                current["embedding_source"] = embedding_source
            if embedding_device is not None:
                current["embedding_device"] = embedding_device
            return self.save(current)

    def update_scenes(self, scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """更新场景配置。"""
        with self._lock:
            current = self.load()
            current["scenes"] = scenes
            return self.save(current)


_STORE = ConfigStore()


def get_app_config() -> dict[str, Any]:
    return _STORE.load()


def get_scenes() -> dict[str, dict[str, Any]]:
    return _STORE.load().get("scenes", copy.deepcopy(DEFAULT_SCENES))


def get_default_scene_key() -> str:
    scenes = get_scenes()
    if "hr" in scenes:
        return "hr"
    return next(iter(scenes.keys()))


def _get_env_value(key: str, default: str = "") -> str:
    for env_name in ENV_CANDIDATES.get(key, []):
        value = os.getenv(env_name)
        if value:
            return value
    return default


def _pick_value(runtime_mode: str, key: str, config_value: str, default_value: str = "") -> str:
    env_value = _get_env_value(key, default="")
    normalized_mode = runtime_mode.lower().strip() if runtime_mode else "web"

    # 命令行模式优先环境变量；桌面/Web模式优先前端配置。
    if normalized_mode == "cli":
        return env_value or config_value or default_value
    return config_value or env_value or default_value


def _normalize_llm_vendor(value: str) -> str:
    normalized = (value or "").strip().lower()
    return normalized or DEFAULT_LLM_VENDOR


def _resolve_llm_base_url(llm_vendor: str, base_url: str) -> str:
    """根据厂商选择解析最终 LLM Base URL。"""
    normalized_vendor = _normalize_llm_vendor(llm_vendor)
    preset_url = LLM_VENDOR_BASE_URLS.get(normalized_vendor, "")
    if normalized_vendor != "custom" and preset_url:
        return preset_url
    return base_url or DEFAULT_BASE_URL


def get_effective_settings(runtime_mode: str = "web") -> dict[str, Any]:
    config = _STORE.load()
    llm_vendor = _normalize_llm_vendor(
        _pick_value(runtime_mode, "llm_vendor", str(config.get("llm_vendor", "")), DEFAULT_LLM_VENDOR)
    )
    base_url = _resolve_llm_base_url(
        llm_vendor,
        _pick_value(runtime_mode, "base_url", str(config.get("base_url", "")), DEFAULT_BASE_URL),
    )
    return {
        "api_key": _pick_value(runtime_mode, "api_key", str(config.get("api_key", "")), ""),
        "base_url": base_url,
        "llm_vendor": llm_vendor,
        "model_name": _pick_value(runtime_mode, "model_name", str(config.get("model_name", "")), DEFAULT_MODEL_NAME),
        "embedding_provider": _pick_value(
            runtime_mode,
            "embedding_provider",
            str(config.get("embedding_provider", "")),
            DEFAULT_EMBEDDING_PROVIDER,
        ).lower(),
        "embedding_model": _pick_value(
            runtime_mode,
            "embedding_model",
            str(config.get("embedding_model", "")),
            DEFAULT_EMBEDDING_MODEL,
        ),
        "embedding_source": _pick_value(
            runtime_mode,
            "embedding_source",
            str(config.get("embedding_source", "")),
            DEFAULT_EMBEDDING_SOURCE,
        ).lower(),
        "embedding_device": _pick_value(
            runtime_mode,
            "embedding_device",
            str(config.get("embedding_device", "")),
            DEFAULT_EMBEDDING_DEVICE,
        ).lower(),
    }


def get_api_key(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("api_key", ""))


def get_base_url(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("base_url", DEFAULT_BASE_URL))


def get_llm_vendor(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("llm_vendor", DEFAULT_LLM_VENDOR)).lower()


def get_model_name(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("model_name", DEFAULT_MODEL_NAME))


def get_embedding_provider(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)).lower()


def get_embedding_model(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("embedding_model", DEFAULT_EMBEDDING_MODEL))


def get_embedding_source(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("embedding_source", DEFAULT_EMBEDDING_SOURCE)).lower()


def get_embedding_device(runtime_mode: str = "web") -> str:
    return str(get_effective_settings(runtime_mode).get("embedding_device", DEFAULT_EMBEDDING_DEVICE)).lower()


def save_settings(
    api_key: str | None = None,
    base_url: str | None = None,
    llm_vendor: str | None = None,
    model_name: str | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_source: str | None = None,
    embedding_device: str | None = None,
) -> dict[str, Any]:
    return _STORE.update_settings(
        api_key=api_key,
        base_url=base_url,
        llm_vendor=llm_vendor,
        model_name=model_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_source=embedding_source,
        embedding_device=embedding_device,
    )


def save_scenes(scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return _STORE.update_scenes(scenes=scenes)
