#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from query_strategy import (
    detect_query_intent,
    format_catalog_count_answer,
    format_catalog_list_answer,
    is_catalog_scope_query,
    resolve_scene_for_catalog_query,
)


class QueryStrategyTests(unittest.TestCase):
    def test_detect_count_intent_for_catalog_query(self):
        query = "你知识库里有几本书？"
        self.assertEqual(detect_query_intent(query), "count")
        self.assertTrue(is_catalog_scope_query(query))

    def test_detect_list_intent_for_catalog_query(self):
        query = "请列出 book 场景知识库全部文档"
        self.assertEqual(detect_query_intent(query), "list")
        self.assertTrue(is_catalog_scope_query(query))

    def test_non_catalog_count_query_should_fallback_to_qa(self):
        query = "报销有几种方式？"
        self.assertEqual(detect_query_intent(query), "qa")
        self.assertFalse(is_catalog_scope_query(query))

    def test_format_catalog_count_answer(self):
        answer = format_catalog_count_answer(
            scene_name="书籍",
            scene_key="book",
            document_names=["a.epub", "b.epub"],
        )
        self.assertIn("共有 2 份文档", answer)
        self.assertIn("a.epub", answer)
        self.assertIn("b.epub", answer)

    def test_format_catalog_list_answer(self):
        answer = format_catalog_list_answer(
            scene_name="书籍",
            scene_key="book",
            document_names=["a.epub", "b.epub"],
        )
        self.assertIn("共 2 份文档", answer)
        self.assertIn("1. a.epub", answer)
        self.assertIn("2. b.epub", answer)

    def test_resolve_scene_for_catalog_query_by_scene_name(self):
        scenes = {
            "book": {"name": "书籍", "keywords": [], "path": "./data/books"},
            "finance": {"name": "财务报销", "keywords": [], "path": "./data/finance"},
        }
        resolved = resolve_scene_for_catalog_query(
            user_query="请列出财务报销知识库里的文档",
            scenes=scenes,
            fallback_scene="book",
        )
        self.assertEqual(resolved, "finance")

    def test_resolve_scene_for_catalog_query_by_book_marker(self):
        scenes = {
            "book": {"name": "书籍", "keywords": [], "path": "./data/books"},
            "finance": {"name": "财务报销", "keywords": [], "path": "./data/finance"},
        }
        resolved = resolve_scene_for_catalog_query(
            user_query="你知识库里有几本书？",
            scenes=scenes,
            fallback_scene="finance",
        )
        self.assertEqual(resolved, "book")


if __name__ == "__main__":
    unittest.main()
