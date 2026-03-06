"""
Embedding Vector Generation Module - Using AstrBot framework's embedding provider

Reference: AstrBot Embedding Provider API
"""

from typing import Optional

from astrbot.api import logger


class EmbeddingProvider:
    """Embedding vector generator - use AstrBot framework's embedding provider"""

    def __init__(
        self,
        provider=None,
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize embedding provider

        Args:
            provider: AstrBot's embedding provider instance (preferred)
            model: Embedding model name (effective when using custom API)
        """
        self.provider = provider
        self.model = model

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """
        Get embedding vector for text

        Args:
            text: Input text

        Returns:
            Embedding vector list, None on failure
        """
        if not text:
            return None

        try:
            if self.provider:
                return await self._get_embedding_from_provider(text)

            logger.error("[FAISSRAG] No embedding provider configured")
            return None

        except Exception as e:
            logger.error(f"[FAISSRAG] Get embedding failed: {e}", exc_info=True)
            return None

    async def _get_embedding_from_provider(self, text: str) -> Optional[list[float]]:
        """Get embedding from AstrBot provider"""
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

            logger.warning(f"[FAISSRAG] Provider has no available embedding method")
            return None

        except Exception as e:
            logger.error(f"[FAISSRAG] Get embedding from provider failed: {e}", exc_info=True)
            return None

    async def get_embedding_batch(
        self, texts: list[str]
    ) -> Optional[list[list[float]]]:
        """
        Batch get embedding vectors

        Args:
            texts: Input text list

        Returns:
            List of embedding vectors
        """
        if not texts:
            return None

        try:
            if self.provider:
                if hasattr(self.provider, "embed_texts"):
                    return await self.provider.embed_texts(texts)
                if hasattr(self.provider, "get_embeddings"):
                    return await self.provider.get_embeddings(texts)

            logger.error("[FAISSRAG] No embedding provider configured")
            return None

        except Exception as e:
            logger.error(f"[FAISSRAG] Batch get embedding failed: {e}", exc_info=True)
            return None