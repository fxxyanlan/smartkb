"""向量数据库封装：Chroma（开发）/ Milvus（生产可换）。

核心三操作：
    - add:    把文本块 + 向量 + 元数据写进去
    - search: 给定查询向量，返回 top-k 个最相似的块
    - delete: 按 doc_id 删除某文档的全部块
"""
from app.config import get_settings
from app.core.embeddings import get_embedder


class VectorStore:
    """统一的向量库接口。"""

    def __init__(self) -> None:
        s = get_settings()
        self._embedder = get_embedder()

        if s.vector_store_type != "chroma":
            raise ValueError(f"暂不支持的向量库类型: {s.vector_store_type}")

        import chromadb
        self._client = chromadb.PersistentClient(path=str(s.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        doc_id: str,
        doc_name: str,
        chunks: list[str],
    ) -> list[str]:
        """把文档的所有 chunks 写入向量库，返回每个 chunk 的 ID。"""

        vectors = self._embedder.embed_texts(chunks)
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self._collection.add(
            ids=chunk_ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=[{"doc_id": doc_id, "doc_name": doc_name}] * len(chunks),
        )
        return chunk_ids

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """检索：返回 [{"doc_id", "doc_name", "chunk_text", "score"}, ...]。"""
        qvec = self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[qvec],
            n_results=top_k or get_settings().top_k,
        )
        return [
            {
                "doc_id": m["doc_id"],
                "doc_name": m["doc_name"],
                "chunk_text": doc,
                "score": 1 - dist,  # distance 转相似度
            }
            for doc, m, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def delete(self, doc_id: str) -> int:
        """删除某文档的全部 chunk，返回条数。"""
        existing = self._collection.get(where={"doc_id": doc_id})
        n = len(existing["ids"])
        if n:
            self._collection.delete(where={"doc_id": doc_id})
        return n

    def list_docs(self) -> list[dict]:
        """列出库里所有文档（按 doc_id 去重）。"""
        items = self._collection.get(include=["metadatas"])
        seen: dict[str, str] = {}
        for m in items["metadatas"]:
            seen[m["doc_id"]] = m["doc_name"]
        return [
            {"doc_id": did, "doc_name": name}
            for did, name in seen.items()
        ]

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """全局单例。"""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store