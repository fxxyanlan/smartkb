"""检索缓存与缓存装饰器的单元测试。"""
from app.core import cache as cache_mod
from app.core.cache import RetrievalCache
from app.core.retriever import CachedRetriever, StoreRetriever
from tests.fakes import FakeStore


class _CountingRetriever:
    """记录被真实调用次数的检索器替身。"""

    def __init__(self, results=None) -> None:
        self._results = results or [{"chunk_id": "c1", "doc_id": "d1", "chunk_text": "片段"}]
        self.calls = 0

    def search(self, query: str, top_k: int) -> list[dict]:
        self.calls += 1
        return [dict(r) for r in self._results]


def test_cache_hit_and_miss_stats():
    cache = RetrievalCache(maxsize=8, ttl=0)
    cache.put(("q", 4, 1), [{"a": 1}])

    assert cache.get(("q", 4, 1)) == [{"a": 1}]
    assert cache.get(("q", 4, 1)) == [{"a": 1}]
    assert cache.get(("other", 4, 1)) is None

    stats = cache.stats
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["hit_rate"] == round(2 / 3, 4)


def test_cache_evicts_least_recently_used():
    cache = RetrievalCache(maxsize=2, ttl=0)
    cache.put(("a",), ["A"])
    cache.put(("b",), ["B"])

    cache.get(("a",))  # a 变成最近使用

    cache.put(("c",), ["C"])  # 淘汰最久未使用的 b

    assert cache.get(("b",)) is None
    assert cache.get(("a",)) == ["A"]
    assert cache.get(("c",)) == ["C"]


def test_cache_entries_expire_by_ttl(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now["t"])

    cache = RetrievalCache(maxsize=8, ttl=10.0)
    cache.put(("q",), ["X"])

    now["t"] += 5
    assert cache.get(("q",)) == ["X"]

    now["t"] += 6  # 累计 11s > ttl
    assert cache.get(("q",)) is None


def test_cache_returns_copy_not_internal_list():
    """返回列表浅拷贝：调用方增删条目不会污染缓存。"""
    cache = RetrievalCache(maxsize=8, ttl=0)
    cache.put(("q",), [{"chunk_id": "c1"}])

    got = cache.get(("q",))
    got.append({"chunk_id": "多余条目"})
    del got[0]

    assert cache.get(("q",)) == [{"chunk_id": "c1"}]


def test_cached_retriever_skips_inner_on_repeat():
    inner = _CountingRetriever()
    retriever = CachedRetriever(inner, RetrievalCache(8, 0), revision_provider=lambda: 1)

    first = retriever.search("问题", 4)
    second = retriever.search("问题", 4)

    assert inner.calls == 1  # 第二次命中缓存，没有真正检索
    assert first == second


def test_cached_retriever_key_includes_top_k():
    inner = _CountingRetriever()
    retriever = CachedRetriever(inner, RetrievalCache(8, 0), revision_provider=lambda: 1)

    retriever.search("问题", 4)
    retriever.search("问题", 8)

    assert inner.calls == 2  # top_k 不同 → 不同的缓存键


def test_cached_retriever_invalidates_on_revision_bump():
    revision = {"v": 1}
    inner = _CountingRetriever()
    retriever = CachedRetriever(
        inner, RetrievalCache(8, 0), revision_provider=lambda: revision["v"]
    )

    retriever.search("问题", 4)
    retriever.search("问题", 4)
    assert inner.calls == 1

    revision["v"] = 2  # 模拟文档增删
    retriever.search("问题", 4)
    assert inner.calls == 2


def test_cache_invalidated_after_document_added():
    """端到端：新增文档后，同一问题必须重新检索（不能命中旧结果）。"""
    store = FakeStore()
    store.add("d1", "a.md", ["苹果"])
    retriever = CachedRetriever(
        StoreRetriever(store), RetrievalCache(8, 0), revision_provider=lambda: store.revision
    )

    before = retriever.search("苹果", 4)
    store.add("d2", "b.md", ["香蕉"])
    after = retriever.search("苹果", 4)

    assert len(before) == 1
    assert len(after) == 2
