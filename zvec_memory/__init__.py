"""FAISSRAG - 基于 FAISS 的 AstrBot 长期记忆插件"""

from .vector_store import FAISSMemoryStore
from .embedding import EmbeddingProvider

__all__ = ["FAISSMemoryStore", "EmbeddingProvider"]