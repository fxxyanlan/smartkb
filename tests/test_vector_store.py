"""向量库集成测试：用真实 Chroma + 临时目录 + 确定性假嵌入。"""
import app.core.vector_store as vs
from app.config import get_settings
from tests.fakes import FakeEmbedder


def _make_store(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(vs, "get_embedder", lambda: FakeEmbedder())
    return vs.VectorStore()


def test_add_count_and_find_by_hash(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)

    store.add("d1", "doc1.md", ["hello world", "foo bar"], "hash-1")

    assert store.count() == 2
    assert store.find_by_hash("hash-1")["doc_id"] == "d1"
    assert store.find_by_hash("missing") is None


def test_list_docs_counts_chunks(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store.add("d1", "doc1.md", ["a", "b", "c"], "h1")

    docs = store.list_docs()

    assert len(docs) == 1
    assert docs[0]["chunks"] == 3


def test_search_returns_most_similar(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store.add("d1", "doc1.md", ["hello hello hello", "zzz qqq"], "h1")

    results = store.search("hello", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_text"] == "hello hello hello"
    assert results[0]["score"] > 0


def test_delete_returns_count_and_is_idempotent(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store.add("d1", "doc1.md", ["a", "b"], "h1")

    assert store.delete("d1") == 2
    assert store.delete("d1") == 0
    assert store.count() == 0
