#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path

from scene_catalog import ensure_scene_catalog, get_scene_document_names


class SceneCatalogTests(unittest.TestCase):
    def test_ensure_scene_catalog_and_refresh_when_files_changed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            data_dir = tmp_root / "books"
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "a.epub").write_text("A", encoding="utf-8")
            (data_dir / "b.epub").write_text("B", encoding="utf-8")

            catalog_path = tmp_root / "scene_catalog.json"
            scene_info = {
                "name": "书籍",
                "path": str(data_dir),
            }

            first = ensure_scene_catalog(
                scene_key="book",
                scene_info=scene_info,
                catalog_path=catalog_path,
            )
            self.assertEqual(first.get("doc_count"), 2)
            first_sig = str(first.get("signature", ""))
            self.assertTrue(first_sig)

            (data_dir / "c.epub").write_text("C", encoding="utf-8")
            second = ensure_scene_catalog(
                scene_key="book",
                scene_info=scene_info,
                catalog_path=catalog_path,
            )
            self.assertEqual(second.get("doc_count"), 3)
            self.assertNotEqual(first_sig, str(second.get("signature", "")))

            names = get_scene_document_names(
                scene_key="book",
                scene_info=scene_info,
                catalog_path=catalog_path,
            )
            self.assertEqual(sorted(names), ["a.epub", "b.epub", "c.epub"])


if __name__ == "__main__":
    unittest.main()
