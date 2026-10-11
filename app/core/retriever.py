"""混合检索：向量语义召回 + BM25 关键词召回，用 RRF 融合取长补短。

动机（实测驱动）：纯向量检索对"超纲问题"仍会给出约 0.49 的相似度，
区分力有限；引入关键词通道可显著提升专有名词、编号、缩写类问题的召回。

设计：
    - 两个通道各自召回 retrieve_candidates 个候选；
    - 用 RRF（倒数排名融合）合并——只依赖名次、不依赖量纲，鲁棒且无需调权；
    - BM25 索引按 store 的版本号惰性重建，避免每次查询全量重算；
    - 外层可套 RerankedRetriever 精排（两段式检索）与 CachedRetriever 缓存。
"""
import logging

from app.config import get_settings
from app.core.bm25 import BM25Index
from app.core.cache import get_retrieval_cache
from app.core.reranker import get_reranker

logger = logging.getLogger(__name__)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """倒数排名融合：score(d) = Σ 1/(k + rank)，rank 从 1 开始。

    只使用名次信息，天然规避了"向量相似度"与"BM25 分数"量纲不可比的问题。
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


class StoreRetriever:
    """退路：仅用向量检索（当 store 不支持全量取块时）。"""

    def __init__(self, store) -> None:
        self._store = store

    def search(self, query: str, top_k: int) -> list[dict]:
        return self._store.search(query, top_k=top_k)


class HybridRetriever:
    """向量 + BM25 混合检索器。"""

    def __init__(self, store, hybrid: bool | None = None) -> None:
        s = get_settings()
        self._store = store
        # hybrid=False 时退化为纯向量，便于做 A/B 对照实验
        self._hybrid = s.hybrid_enabled if hybrid is None else hybrid
        self._rrf_k = s.rrf_k
        self._candidates = s.retrieve_candidates
        self._index = BM25Index(k1=s.bm25_k1, b=s.bm25_b)
        self._rev: int | None = None
        self._built = False

    def _ensure_index(self) -> None:
        """按 store 版本号惰性重建 BM25 索引。"""
        rev = getattr(self._store, "revision", None)
        if self._built and rev is not None and rev == self._rev:
            return
        chunks = self._store.get_all_chunks()
        self._index.build(chunks)
        self._rev = rev
        self._built = True
        logger.info("BM25 索引已重建 | chunks=%d", len(chunks))

    def search(self, query: str, top_k: int) -> list[dict]:
        """返回融合后的 top_k 个片段（保留向量相似度作为展示分数）。"""
        cand = max(top_k, self._candidates)
        by_id: dict[str, dict] = {}
        rankings: list[list[str]] = []

        # 通道 1：向量语义检索
        vec = self._store.search(query, top_k=cand)
        rankings.append([r["chunk_id"] for r in vec])
        for r in vec:
            by_id.setdefault(r["chunk_id"], r)

        # 通道 2：BM25 关键词检索
        if self._hybrid:
            self._ensure_index()
            bm = self._index.search(query, top_k=cand)
            rankings.append([r["chunk_id"] for r in bm])
            for r in bm:
                by_id.setdefault(r["chunk_id"], r)

        scores = rrf_fuse(rankings, k=self._rrf_k)
        ordered = sorted(
            by_id.values(),
            key=lambda r: scores.get(r["chunk_id"], 0.0),
            reverse=True,
        )
        return ordered[:top_k]


class RerankedRetriever:
    """精排装饰器：先粗召回候选池，再交给重排器取 top_k。

    必须多召回一些候选（rerank_candidates > top_k），重排才有纠错空间；
    重排器不可用时自动退化为截断，链路始终可用。
    """

    def __init__(self, inner, reranker, candidates: int) -> None:
        self._inner = inner
        self._reranker = reranker
        self._candidates = max(1, candidates)

    def search(self, query: str, top_k: int) -> list[dict]:
        pool = max(top_k, self._candidates)
        candidates = self._inner.search(query, pool)
        return self._reranker.rerank(query, candidates, top_k)


class CachedRetriever:
    """检索缓存装饰器：命中时直接返回，跳过 Embedding + 向量库 + BM25 全流程。

    缓存键 = (问题, top_k, 数据版本号)。版本号来自 store，文档增删后旧键不再命中，
    因此不需要显式清理缓存；TTL / LRU 只负责兜底与容量控制。
    """

    def __init__(self, inner, cache, revision_provider=None) -> None:
        self._inner = inner
        self._cache = cache
        self._revision_provider = revision_provider

    def search(self, query: str, top_k: int) -> list[dict]:
        rev = self._revision_provider() if self._revision_provider else None
        key = (query.strip(), top_k, rev)

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._inner.search(query, top_k)
        self._cache.put(key, result)
        return result


def build_retriever(store):
    """工厂：组装「粗召回（混合/纯向量）→ 精排 → 缓存」三层链路。

    - store 支持全量取块则用混合检索，否则退回纯向量；
    - rerank_enabled 时套上精排装饰器；
    - cache_enabled 时最外层套缓存（缓存的是精排后的最终结果，命中即全免）。
    """
    if hasattr(store, "get_all_chunks"):
        inner = HybridRetriever(store)
    else:
        inner = StoreRetriever(store)

    s = get_settings()
    if s.rerank_enabled:
        inner = RerankedRetriever(inner, get_reranker(), s.rerank_candidates)

    if not s.cache_enabled:
        return inner

    return CachedRetriever(
        inner,
        get_retrieval_cache(),
        revision_provider=lambda: getattr(store, "revision", None),
    )
