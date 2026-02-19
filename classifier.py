#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : classifier.py
@Author  : Kevin
@Date    : 2025/10/26
@Description : 场景分类器.
@Version : 1.0
"""

from openai import OpenAI
from config_store import get_api_key, get_base_url, get_default_scene_key, get_model_name, get_scenes


def _build_client(runtime_mode: str = "web") -> OpenAI:
    """按运行时配置创建 OpenAI 客户端。"""
    return OpenAI(
        api_key=get_api_key(runtime_mode=runtime_mode),
        base_url=get_base_url(runtime_mode=runtime_mode),
    )


def classify_scene_by_keywords(query: str, scenes: dict | None = None) -> str | None:
    """基于关键词快速匹配"""
    runtime_scenes = scenes or get_scenes()
    for scene, info in runtime_scenes.items():
        if any(kw in query for kw in info["keywords"]):
            return scene
    return None


def classify_scene_by_llm(query: str, runtime_mode: str = "web") -> str:
    """LLM零样本分类（兜底）"""
    scenes = get_scenes()
    default_scene = get_default_scene_key()
    scene_descs = [f"{k}: {v['name']}" for k, v in scenes.items()]
    model_name = get_model_name(runtime_mode=runtime_mode)

    if not get_api_key(runtime_mode=runtime_mode):
        print("⚠️ API_KEY 未配置，分类器回退到默认场景。")
        return default_scene

    prompt = f"""
你是一个企业知识库路由系统。请根据用户问题，判断其最可能属于以下哪个业务场景：
{chr(10).join(scene_descs)}

问题：{query}
要求：只输出场景英文标识（如 hr, it, finance），不要解释。
"""
    try:
        client = _build_client(runtime_mode=runtime_mode)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        pred = response.choices[0].message.content.strip().lower()
        return pred if pred in scenes else default_scene
    except Exception as e:
        print(f"LLM分类失败: {e}")
        return default_scene


def classify_scene(query: str, runtime_mode: str = "web") -> str:
    """主分类函数：先关键词，再LLM"""
    scenes = get_scenes()
    scene = classify_scene_by_keywords(query, scenes=scenes)
    if scene:
        return scene
    return classify_scene_by_llm(query, runtime_mode=runtime_mode)
