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
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from config import SCENES
from classifier import classify_scene

# 设置全局 LLM 和 Embedding Model
Settings.llm = DashScope(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model_name=DashScopeGenerationModels.QWEN_MAX
)
Settings.embed_model = DashScopeEmbedding(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT
)

class MultiSceneRAG:
    def __init__(self):
        self.indices = {}
        self._init_indices()

    def _init_indices(self):
        """为每个场景构建或加载向量索引"""
        client = chromadb.PersistentClient(path="./storage/chroma_db")

        for scene, info in SCENES.items():
            print(f"Loading index for scene: {scene}")
            collection = client.get_or_create_collection(f"scene_{scene}")
            vector_store = ChromaVectorStore(chroma_collection=collection)

            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
            )

            persist_dir = f"./storage/{scene}"

            # 检查完整的持久化目录是否存在（包含docstore.json等文件）
            docstore_path = os.path.join(persist_dir, "docstore.json")

            if os.path.exists(f"./storage/{scene}") and os.path.exists(docstore_path):
                # 加载已有索引
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
                    # 如果加载失败，重新构建索引
                    index = self._build_new_index(info["path"], storage_context, persist_dir)
            else:
                print(f"🔄 Building new index for scene: {scene}")
                # 首次构建索引或目录不存在
                index = self._build_new_index(info["path"], storage_context, persist_dir)

            self.indices[scene] = index.as_query_engine()

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

    def query(self, user_query: str) -> str:
        scene = classify_scene(user_query)
        print(f"🔍 路由到场景: {SCENES[scene]['name']} ({scene})")

        query_engine = self.indices[scene]
        response = query_engine.query(user_query)
        return str(response)
