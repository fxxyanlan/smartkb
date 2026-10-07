"""Embedding 模型封装：把文本转成向量。

轻量方案：用 sentence-transformers 本地跑 bge-small-zh。
云端方案：可换 OpenAI 兼容的 embedding API。

任何模块获取 embedder 都通过 get_embedder()。
"""
from functools import lru_cache

from app.config import get_settings


class EmbeddingClient:
    """文本转向量的客户端。"""

    def __init__(self) -> None:
        s = get_settings()
        self.model_name = s.embedding_model
        self.device = s.embedding_device
        self._model = None  # TODO: 实际加载 sentence-transformers 模型

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入：返回每段文本的向量。"""
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name,device=self.device)
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """单个查询的嵌入（部分模型对 query 和 doc 区分对待）。"""
        return self.embed_texts([query])[0]


@lru_cache
def get_embedder() -> EmbeddingClient:
    """单例。"""
    return EmbeddingClient()
