"""嵌入向量生成模块 - 使用 AstrBot 框架的嵌入提供商

ZVec API Reference: https://zvec.org/api-reference/python/
- 核心组件: CollectionSchema, Doc, VectorQuery, HnswIndexParam, MetricType, DataType
"""

from typing import Optional

from astrbot.api import logger


class EmbeddingProvider:
    """嵌入向量生成器 - 优先使用 AstrBot 框架的嵌入提供商"""

    def __init__(
        self,
        provider=None,
        model: str = "text-embedding-3-small",
    ):
        """
        初始化嵌入提供商

        Args:
            provider: AstrBot 的嵌入提供商实例（优先使用）
            model: 嵌入模型名称（当使用自定义 API 时有效）
        """
        self.provider = provider
        self.model = model

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """
        获取文本的嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量列表，失败返回 None
        """
        if not text:
            return None

        try:
            # 使用 AstrBot 的嵌入提供商
            if self.provider:
                return await self._get_embedding_from_provider(text)

            logger.error("[ZVecRAG] 未配置嵌入提供商，请确保 AstrBot 配置了嵌入提供商")
            return None

        except Exception as e:
            logger.error(f"[ZVecRAG] 获取嵌入失败: {e}", exc_info=True)
            return None

    async def _get_embedding_from_provider(self, text: str) -> Optional[list[float]]:
        """从 AstrBot 提供商获取嵌入"""
        try:
            # 尝试使用 embed_texts 方法
            if hasattr(self.provider, "embed_texts"):
                results = await self.provider.embed_texts([text])
                if results and len(results) > 0:
                    return results[0]

            # 尝试使用 get_embedding 方法
            if hasattr(self.provider, "get_embedding"):
                return await self.provider.get_embedding(text)

            # 尝试使用 get_embeddings 方法
            if hasattr(self.provider, "get_embeddings"):
                results = await self.provider.get_embeddings([text])
                if results and len(results) > 0:
                    return results[0]

            logger.warning(f"[ZVecRAG] 提供商没有可用的嵌入方法")
            return None

        except Exception as e:
            logger.error(f"[ZVecRAG] 从提供商获取嵌入失败: {e}", exc_info=True)
            return None

    async def get_embedding_batch(
        self, texts: list[str]
    ) -> Optional[list[list[float]]]:
        """
        批量获取嵌入向量

        Args:
            texts: 输入文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return None

        try:
            # 使用 AstrBot 的嵌入提供商
            if self.provider:
                if hasattr(self.provider, "embed_texts"):
                    return await self.provider.embed_texts(texts)
                if hasattr(self.provider, "get_embeddings"):
                    return await self.provider.get_embeddings(texts)

            logger.error("[ZVecRAG] 未配置嵌入提供商，无法批量获取嵌入")
            return None

        except Exception as e:
            logger.error(f"[ZVecRAG] 批量获取嵌入失败: {e}", exc_info=True)
            return None
