"""BM25 / RRF / 混合检索单元测试。"""
from app.core.bm25 import BM25Index, tokenize
from app.core.retriever import HybridRetriever, rrf_fuse


class _FusedStore:
    """测试用 store：向量通道只返回前 2 个（模拟向量漏召），用于验证 BM25 补位。"""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self._revision = 1

    @property
    def revision(self) -> int:
        return self._revision

    def get_all_chunks(self) -> list[dict]:
        return self.chunks

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        return [{**c, "score": 0.9 - i * 0.1} for i, c in enumerate(self.chunks[:2])]


def _chunk(cid: str, text: str) -> dict:
    return {"chunk_id": cid, "doc_id": cid, "doc_name": f"{cid}.md", "chunk_text": text}


# ---------- tokenize ----------

def test_tokenize_ascii_keeps_whole_word():
    assert tokenize("DeepSeek RRF") == ["deepseek", "rrf"]


def test_tokenize_chinese_emits_unigram_and_bigram():
    tokens = tokenize("知识库")
    assert "知" in tokens and "知识" in tokens and "识库" in tokens


# ---------- BM25 ----------

def test_bm25_ranks_keyword_match_first():
    index = BM25Index()
    index.build([
        _chunk("a", "系统使用大语言模型生成答案"),
        _chunk("b", "混合检索的融合参数 rrf_k 默认 60"),
        _chunk("c", "员工年假上限 15 天"),
    ])

    results = index.search("rrf_k", top_k=3)

    assert results[0]["chunk_id"] == "b"
    assert results[0]["bm25_score"] > 0


def test_bm25_empty_index_returns_empty():
    assert BM25Index().search("任意", top_k=3) == []


# ---------- RRF ----------

def test_rrf_fuse_rewards_multichannel_agreement():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k=60)

    # a 与 b 都各被两个通道命中，分数应相同
    assert fused["a"] == fused["b"]


def test_rrf_fuse_single_channel_uses_rank():
    fused = rrf_fuse([["x", "y"]], k=60)

    assert fused["x"] > fused["y"]


# ---------- 混合检索 ----------

def test_hybrid_recovers_chunk_vector_channel_missed():
    """向量通道漏召时，BM25 应把关键词命中的片段补进来（这是混合检索的核心价值）。"""
    store = _FusedStore([
        _chunk("c0", "普通内容一"),
        _chunk("c1", "普通内容二"),
        _chunk("c2", "配置项 rrf_k 默认 60"),
    ])

    vector_only = HybridRetriever(store, hybrid=False)
    hybrid = HybridRetriever(store, hybrid=True)

    vec_ids = [r["chunk_id"] for r in vector_only.search("rrf_k", top_k=3)]
    hyb_ids = [r["chunk_id"] for r in hybrid.search("rrf_k", top_k=3)]

    assert "c2" not in vec_ids  # 向量通道看不到
    assert "c2" in hyb_ids  # 混合检索补回来了


def test_hybrid_respects_top_k():
    store = _FusedStore([_chunk(f"c{i}", f"内容{i} 关键词") for i in range(6)])
    hybrid = HybridRetriever(store, hybrid=True)

    assert len(hybrid.search("关键词", top_k=2)) == 2


def test_hybrid_rebuilds_index_on_revision_bump():
    """数据变更（版本号变化）后 BM25 索引必须重建，否则会检索到过期内容。"""
    store = _FusedStore([_chunk("c0", "旧内容")])
    hybrid = HybridRetriever(store, hybrid=True)

    assert "c1" not in [r["chunk_id"] for r in hybrid.search("新增关键词", top_k=5)]

    store.chunks.append(_chunk("c1", "新增关键词在这里"))
    store._revision += 1

    assert "c1" in [r["chunk_id"] for r in hybrid.search("新增关键词", top_k=5)]


def test_hybrid_bm25_only_hit_has_no_vector_score():
    """仅 BM25 命中的片段没有向量相似度，score 应为 None（前端已做类型保护）。"""
    store = _FusedStore([
        _chunk("c0", "普通内容一"),
        _chunk("c1", "普通内容二"),
        _chunk("c2", "关键词 zebra 出现"),  # 排在前 2 之外，向量通道看不到
    ])
    hybrid = HybridRetriever(store, hybrid=True)

    by_id = {r["chunk_id"]: r for r in hybrid.search("zebra", top_k=3)}

    assert by_id["c2"].get("score") is None
    assert by_id["c0"].get("score") == 0.9
