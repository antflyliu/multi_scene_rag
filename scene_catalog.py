#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : scene_catalog.py
@Author  : Kevin
@Date    : 2026/02/19
@Description : 场景文档目录管理（用于确定性统计问答）.
@Version : 1.0
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

SCENE_CATALOG_PATH = Path("./storage/scene_catalog.json")


def _read_catalog(catalog_path: Path = SCENE_CATALOG_PATH) -> dict[str, Any]:
    if not catalog_path.exists():
        return {"scenes": {}}
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("scenes"), dict):
            return data
    except Exception:
        pass
    return {"scenes": {}}


def _write_catalog(data: dict[str, Any], catalog_path: Path = SCENE_CATALOG_PATH) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _scene_documents_signature(documents: list[dict[str, Any]]) -> str:
    raw = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def scan_scene_documents(scene_key: str, scene_info: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(str(scene_info.get("path", "") or ""))
    scene_name = str(scene_info.get("name", scene_key) or scene_key)
    documents: list[dict[str, Any]] = []
    if data_path.exists():
        for file_path in sorted([p for p in data_path.rglob("*") if p.is_file()]):
            stat = file_path.stat()
            documents.append(
                {
                    "file_name": file_path.name,
                    "file_path": str(file_path.resolve()),
                    "relative_path": file_path.relative_to(data_path).as_posix(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    signature = _scene_documents_signature(documents)
    return {
        "scene_key": scene_key,
        "scene_name": scene_name,
        "scene_path": str(data_path),
        "doc_count": len(documents),
        "documents": documents,
        "signature": signature,
    }


def ensure_scene_catalog(scene_key: str, scene_info: dict[str, Any], catalog_path: Path = SCENE_CATALOG_PATH) -> dict[str, Any]:
    catalog = _read_catalog(catalog_path=catalog_path)
    scene_map = catalog.setdefault("scenes", {})
    existing = scene_map.get(scene_key, {})
    scanned = scan_scene_documents(scene_key=scene_key, scene_info=scene_info)
    if isinstance(existing, dict) and existing.get("signature") == scanned.get("signature"):
        return existing

    updated = {
        **scanned,
        "updated_at_epoch": int(time.time()),
    }
    scene_map[scene_key] = updated
    _write_catalog(catalog, catalog_path=catalog_path)
    return updated


def get_scene_document_names(scene_key: str, scene_info: dict[str, Any], catalog_path: Path = SCENE_CATALOG_PATH) -> list[str]:
    scene_entry = ensure_scene_catalog(scene_key=scene_key, scene_info=scene_info, catalog_path=catalog_path)
    docs = scene_entry.get("documents", [])
    if not isinstance(docs, list):
        return []
    names = [str(item.get("file_name", "")).strip() for item in docs if isinstance(item, dict)]
    return [name for name in names if name]
