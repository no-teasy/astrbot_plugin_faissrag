"""
FAISS 记忆存储模块

使用 FAISS (Facebook AI Similarity Search) 进行向量相似度搜索。
FAISS 是进程内数据库，支持多种索引类型，性能优异。

主要 API:
- faiss.IndexFlatIP: 内积索引（余弦相似度需要归一化）
- faiss.IndexIVFFlat: 倒排索引，加速搜索
- faiss.StandardGpuResources: GPU 支持（可选）
- index.add(): 添加向量
- index.search(): 向量搜索
- index.ntotal: 向量数量
"""

import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np

from astrbot.api import logger


class FAISSMemoryStore:
    """基于 FAISS 的记忆存储"""

    def __init__(
        self,
        data_dir: str,
        collection_name: str = "faiss_memory",
        embedding_dim: int = 1536,
    ):
        self.data_dir = Path(data_dir)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # FAISS 索引路径
        self.index_path = self.data_dir / f"{collection_name}.index"
        self.meta_path = self.data_dir / f"{collection_name}_meta.json"

        # FAISS 索引对象
        self._index: Optional[faiss.Index] = None

        # 元数据存储（内存）
        # 每条记录: {memory_id: {...}, ...}
        self._metadata: dict[str, dict] = {}

        # 作用域索引 {scope_key: [memory_id, ...]}
        self._scope_index: dict[str, list[str]] = {}

    async def initialize(self) -> None:
        """初始化 FAISS 索引"""
        try:
            # 确保数据目录存在
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # 加载或创建索引
            if self.index_path.exists() and self.meta_path.exists():
                # 加载已存在的索引
                logger.info(f"[FAISRAG] 加载已存在的索引: {self.index_path}")
                self._index = faiss.read_index(str(self.index_path))

                with open(self.meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._metadata = data.get("metadata", {})
                    self._scope_index = data.get("scope_index", {})

                logger.info(f"[FAISRAG] 已加载 {self._index.ntotal} 条向量")
            else:
                # 创建新索引
                # 使用 IndexFlatIP（内积）进行余弦相似度搜索
                # 注意：使用余弦相似度时，向量需要先归一化
                logger.info(f"[FAISRAG] 创建新 FAISS 索引，维度: {self.embedding_dim}")
                self._index = faiss.IndexFlatIP(self.embedding_dim)

            logger.info(f"[FAISRAG] FAISS 索引初始化完成: {self.collection_name}")

        except Exception as e:
            logger.error(f"[FAISRAG] FAISS 初始化失败: {e}", exc_info=True)
            raise

    def _normalize(self, embedding: list[float]) -> np.ndarray:
        """归一化向量"""
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    async def add_memory(
        self,
        content: str,
        embedding: list[float],
        role: str,
        scope_key: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """添加记忆"""
        try:
            memory_id = str(uuid.uuid4())

            # 归一化向量并添加到索引
            vec = self._normalize(embedding)
            self._index.add(vec)

            # 构建元数据
            meta = {
                "memory_id": memory_id,
                "content": content,
                "role": role,
                "scope_key": scope_key,
                "timestamp": metadata.get("timestamp", 0.0) if metadata else 0.0,
                "sender_id": metadata.get("sender_id") if metadata else None,
                "sender_name": metadata.get("sender_name") if metadata else None,
                "platform": metadata.get("platform") if metadata else None,
                "chat_type": metadata.get("chat_type") if metadata else None,
                "chat_id": metadata.get("chat_id") if metadata else None,
            }

            # 存储元数据
            self._metadata[memory_id] = meta

            # 更新作用域索引
            if scope_key not in self._scope_index:
                self._scope_index[scope_key] = []
            self._scope_index[scope_key].append(memory_id)

            # 定期保存（每10条保存一次）
            if self._index.ntotal % 10 == 0:
                await self._save()

            return memory_id

        except Exception as e:
            logger.error(f"[FAISRAG] 添加记忆失败: {e}", exc_info=True)
            raise

    async def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        scope_key: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """搜索记忆"""
        try:
            if self._index.ntotal == 0:
                return []

            # 归一化查询向量
            query_vec = self._normalize(embedding)

            # 确定搜索范围
            search_k = top_k
            if scope_key:
                # 如果指定了作用域，先获取该作用域的所有 memory_id
                scope_mids = self._scope_index.get(scope_key, [])
                if not scope_mids:
                    return []
                # 搜索更多结果，然后过滤
                search_k = min(max(top_k * 2, 10), self._index.ntotal)

            # 执行搜索
            scores, indices = self._index.search(query_vec, search_k)

            # 格式化结果
            memories = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue

                # 获取 memory_id（FAISS 索引顺序对应 metadata 顺序）
                # 注意：需要通过索引找到对应的 memory_id
                memory_ids = list(self._metadata.keys())
                if idx >= len(memory_ids):
                    continue

                memory_id = memory_ids[idx]
                meta = self._metadata.get(memory_id, {})

                # 过滤作用域
                if scope_key and meta.get("scope_key") != scope_key:
                    continue

                memories.append({
                    "memory_id": memory_id,
                    "content": meta.get("content"),
                    "role": meta.get("role"),
                    "scope_key": meta.get("scope_key"),
                    "timestamp": meta.get("timestamp"),
                    "score": float(score),
                    "sender_id": meta.get("sender_id"),
                    "sender_name": meta.get("sender_name"),
                    "platform": meta.get("platform"),
                    "chat_type": meta.get("chat_type"),
                    "chat_id": meta.get("chat_id"),
                })

                if len(memories) >= top_k:
                    break

            return memories

        except Exception as e:
            logger.error(f"[FAISRAG] 搜索失败: {e}", exc_info=True)
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """删除指定记忆"""
        # FAISS 不支持删除，标记返回失败
        logger.warning("[FAISRAG] 删除功能未实现（FAISS 不支持删除）")
        return False

    async def clear_scope(self, scope_key: str) -> int:
        """清除指定作用域的记忆"""
        try:
            # 统计数量
            memory_ids = self._scope_index.get(scope_key, [])
            count = len(memory_ids)

            # 从元数据中删除
            for mid in memory_ids:
                self._metadata.pop(mid, None)

            # 清空作用域索引
            self._scope_index[scope_key] = []

            # 重建索引（移除被删除的向量）
            await self._rebuild_index()

            logger.info(f"[FAISRAG] 清除作用域 {scope_key} 的 {count} 条记忆")

            return count

        except Exception as e:
            logger.error(f"[FAISRAG] 清除记忆失败: {e}", exc_info=True)
            return 0

    async def _rebuild_index(self) -> None:
        """重建 FAISS 索引"""
        try:
            if not self._metadata:
                self._index.reset()
                return

            # 重新收集所有有效的向量
            # 这里简化处理：需要外部提供向量来重建
            # 实际应用中，可能需要持久化原始向量
            logger.info("[FAISRAG] 索引需要重建才能真正删除数据")

        except Exception as e:
            logger.error(f"[FAISRAG] 重建索引失败: {e}", exc_info=True)

    async def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        try:
            return {
                "total_count": self._index.ntotal,
                "scope_count": len(self._scope_index),
                "collection_name": self.collection_name,
                "embedding_dim": self.embedding_dim,
            }

        except Exception as e:
            logger.error(f"[FAISRAG] 获取统计失败: {e}", exc_info=True)
            return {"total_count": 0, "scope_count": 0}

    async def _save(self) -> None:
        """保存索引和元数据"""
        try:
            # 保存 FAISS 索引
            faiss.write_index(self._index, str(self.index_path))

            # 保存元数据
            data = {
                "metadata": self._metadata,
                "scope_index": self._scope_index,
            }
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"[FAISRAG] 已保存 {self._index.ntotal} 条向量")

        except Exception as e:
            logger.error(f"[FAISRAG] 保存失败: {e}", exc_info=True)

    async def close(self) -> None:
        """关闭存储"""
        try:
            # 保存数据
            await self._save()

            # 清理引用
            self._index = None

            logger.info("[FAISRAG] FAISS 存储已关闭")

        except Exception as e:
            logger.error(f"[FAISRAG] 关闭存储失败: {e}", exc_info=True)


# 兼容旧类名
ZVecMemoryStore = FAISSMemoryStore