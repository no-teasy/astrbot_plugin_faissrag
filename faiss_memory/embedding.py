"""
嵌入向量生成模块 - 使用 AstrBot 框架的嵌入提供者

参考: AstrBot 嵌入提供者 API
"""

from typing import Optional

from astrbot.api import logger


class EmbeddingProvider:
    """嵌入向量生成器 - 使用 AstrBot 框架的嵌入提供者"""

    def __init__(
        self,
        provider=None,
        model: str = "text-embedding-3-small",
    ):
        """
        初始化嵌入提供者

        参数:
            provider: AstrBot 的嵌入提供者实例（优先使用）
            model: 嵌入模型名称（使用自定义 API 时有效）
        """
        self.provider = provider
        self.model = model

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """
        获取文本的嵌入向量

        参数:
            text: 输入文本

        返回:
            嵌入向量列表，失败时返回 None
        """
        if not text:
            return None

        try:
            if self.provider:
                return await self._get_embedding_from_provider(text)

            logger.error("[FAISSRAG] 未配置嵌入提供者")
            return None

        except Exception as e:
            logger.error(f"[FAISSRAG] 获取嵌入失败: {e}", exc_info=True)
            return None

    async def _get_embedding_from_provider(self, text: str) -> Optional[list[float]]:
        """从 AstrBot 提供者获取嵌入"""
        try:
            # Try embed_texts method
            if hasattr(self.provider, "embed_texts"):
                results = await self.provider.embed_texts([text])
                if results and len(results) > 0:
                    return results[0]

            # Try get_embedding method
            if hasattr(self.provider, "get_embedding"):
                return await self.provider.get_embedding(text)

            # Try get_embeddings method
            if hasattr(self.provider, "get_embeddings"):
                results = await self.provider.get_embeddings([text])
                if results and len(results) > 0:
                    return results[0]

            logger.warning(f"[FAISSRAG] 提供者没有可用的嵌入方法")
            return None

        except Exception as e:
            logger.error(f"[FAISSRAG] 从提供者获取嵌入失败: {e}", exc_info=True)
            return None

    async def get_embedding_batch(
        self, texts: list[str]
    ) -> Optional[list[list[float]]]:
        """
        批量获取嵌入向量

        参数:
            texts: 输入文本列表

        返回:
            嵌入向量列表
        """
        if not texts:
            return None

        try:
            if self.provider:
                if hasattr(self.provider, "embed_texts"):
                    return await self.provider.embed_texts(texts)
                if hasattr(self.provider, "get_embeddings"):
                    return await self.provider.get_embeddings(texts)

            logger.error("[FAISSRAG] 未配置嵌入提供者")
            return None

        except Exception as e:
            logger.error(f"[FAISSRAG] 批量获取嵌入失败: {e}", exc_info=True)
            return None