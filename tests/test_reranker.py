"""重排序（Rerank）单元测试：排序正确性、失败降级、链路接线。"""
from app.core import reranker as reranker_mod
from app.core.reranker import CrossEncoderReranker, NoopReranker, get_reranker
from app.core.retriever import (
    CachedRetriever,
    HybridRetriever,
    RerankedRetriever,
    build_retriever,
)
from tests.fakes import FakeStore


def _chunk(cid: str, text: str) -> dict:
    return {"chunk_id": cid, "doc_id": cid, "doc_name": f"{cid}.md", "chunk_text": text}


class _FakeCrossEncoder:
    """假交叉编码器：返回指定分数，或按"问题字符命中数"打分（确定性，便于断言）。"""

    def __init__(self, scores=None, error=None) -> None:
        self.scores = scores
        self.error = error
        self.pairs: list | None = None

    def predict(self, pairs, batch_size=16):
        self.pairs = list(pairs)
        if self.error:
            raise self.error
        if self.scores is not None:
            return self.scores
        return [float(sum(1 for ch in q if ch in d)) for q, d in pairs]


# ---------- NoopReranker ----------

def test_noop_reranker_truncates_candidates():
    out = NoopReranker().rerank("q", [_chunk("a", "1"), _chunk("b", "2")], top_k=1)

    assert [c["chunk_id"] for c in out] == ["a"]


# ---------- CrossEncoderReranker ----------

def test_cross_encoder_reorders_by_score():
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder(scores=[0.1, 0.9]))

    out = reranker.rerank("问题", [_chunk("a", "低相关"), _chunk("b", "高相关")], top_k=2)

    assert [c["chunk_id"] for c in out] == ["b", "a"]
    assert out[0]["rerank_score"] == 0.9


def test_cross_encoder_available_with_injected_model():
    assert CrossEncoderReranker(model=_FakeCrossEncoder()).available is True


def test_cross_encoder_builds_query_document_pairs():
    """传给交叉编码器的必须是 (问题, 片段原文) 成对输入，否则模型无法建模交互。"""
    model = _FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)

    reranker.rerank("什么是RRF", [_chunk("a", "RRF 是倒数排名融合")], top_k=1)

    assert model.pairs == [("什么是RRF", "RRF 是倒数排名融合")]


def test_cross_encoder_degrades_when_model_load_fails(monkeypatch):
    """模型加载失败：降级为不重排（原顺序截断），不抛异常。"""
    import sentence_transformers

    def _boom(*args, **kwargs):
        raise OSError("模型不存在")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _boom)
    reranker = CrossEncoderReranker(model_name="no/such-model")

    assert reranker.available is False
    out = reranker.rerank("q", [_chunk("a", "x"), _chunk("b", "y")], top_k=1)

    assert [c["chunk_id"] for c in out] == ["a"]
    assert "rerank_score" not in out[0]


def test_cross_encoder_degrades_when_scoring_fails():
    """打分阶段异常：退回原顺序，保证链路可用。"""
    reranker = CrossEncoderReranker(model=_FakeCrossEncoder(error=RuntimeError("打分失败")))

    out = reranker.rerank("q", [_chunk("a", "x"), _chunk("b", "y")], top_k=2)

    assert [c["chunk_id"] for c in out] == ["a", "b"]


def test_cross_encoder_empty_candidates_returns_empty():
    assert CrossEncoderReranker(model=_FakeCrossEncoder()).rerank("q", [], 4) == []


# ---------- RerankedRetriever ----------

class _RecordingRetriever:
    """记录被请求的 top_k，用于断言精排确实扩大了粗召回池。"""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.requested: list[int] = []

    def search(self, query: str, top_k: int) -> list[dict]:
        self.requested.append(top_k)
        return self.chunks[:top_k]


def test_reranked_retriever_pools_more_then_cuts_to_top_k():
    inner = _RecordingRetriever([_chunk(f"c{i}", f"内容{i}") for i in range(10)])
    model = _FakeCrossEncoder(scores=list(range(10)))  # 分数递增 → 顺序翻转
    retriever = RerankedRetriever(inner, CrossEncoderReranker(model=model), candidates=10)

    out = retriever.search("问题", top_k=2)

    assert inner.requested == [10]  # 按候选池大小粗召回，而非 top_k
    assert len(out) == 2
    assert out[0]["chunk_id"] == "c9"  # 分数最高者被提到首位


def test_reranked_retriever_pool_never_below_top_k():
    inner = _RecordingRetriever([_chunk(f"c{i}", f"内容{i}") for i in range(10)])
    retriever = RerankedRetriever(inner, NoopReranker(), candidates=2)

    out = retriever.search("问题", top_k=4)

    assert inner.requested == [4]  # 候选池小于 top_k 时至少取 top_k
    assert len(out) == 4


# ---------- build_retriever 接线 ----------

def test_build_retriever_skips_rerank_when_disabled(monkeypatch, settings):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "cache_enabled", False)

    retriever = build_retriever(FakeStore())

    assert isinstance(retriever, HybridRetriever)


def test_build_retriever_wraps_rerank_when_enabled(monkeypatch, settings):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(reranker_mod, "_reranker", NoopReranker())

    retriever = build_retriever(FakeStore())

    assert isinstance(retriever, RerankedRetriever)


def test_build_retriever_cache_is_outermost(monkeypatch, settings):
    """缓存必须最外层：缓存的是精排后的最终结果，命中即全链路免算。"""
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(reranker_mod, "_reranker", NoopReranker())

    retriever = build_retriever(FakeStore())

    assert isinstance(retriever, CachedRetriever)


def test_get_reranker_is_noop_when_disabled(monkeypatch, settings):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(reranker_mod, "_reranker", None)

    assert isinstance(get_reranker(), NoopReranker)
