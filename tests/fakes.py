"""测试替身：不依赖真实模型 / 向量库 / LLM 的轻量假实现。"""
import string


class FakeStore:
    """内存向量库替身，接口与 VectorStore 保持一致。"""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}

    def add(self, doc_id, doc_name, chunks, content_hash=""):
        self.docs[doc_id] = {
            "doc_name": doc_name,
            "chunks": list(chunks),
            "hash": content_hash,
        }
        return [f"{doc_id}_{i}" for i in range(len(chunks))]

    def search(self, query, top_k=None):
        results = []
        for did, d in self.docs.items():
            for c in d["chunks"]:
                score = float(len(set(query) & set(c)))
                results.append(
                    {
                        "doc_id": did,
                        "doc_name": d["doc_name"],
                        "chunk_text": c,
                        "score": score,
                    }
                )
        results.sort(key=lambda r: -r["score"])
        return results[: top_k or 4]

    def find_by_hash(self, content_hash):
        for did, d in self.docs.items():
            if d["hash"] and d["hash"] == content_hash:
                return {"doc_id": did, "doc_name": d["doc_name"]}
        return None

    def delete(self, doc_id):
        d = self.docs.pop(doc_id, None)
        return len(d["chunks"]) if d else 0

    def list_docs(self):
        return [
            {"doc_id": did, "doc_name": d["doc_name"], "chunks": len(d["chunks"])}
            for did, d in self.docs.items()
        ]

    def count(self):
        return sum(len(d["chunks"]) for d in self.docs.values())


class FakeLLM:
    """LLM 替身：记录收到的 messages，便于断言。"""

    def __init__(self, reply="这是基于文档的回答。[来源 1]", fail=False, error=None) -> None:
        self.reply = reply
        self.error = error or (RuntimeError("llm boom") if fail else None)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, temperature=0.3, max_tokens=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply

    async def stream_chat(self, messages, temperature=0.3):
        self.calls.append(messages)
        if self.error:
            raise self.error
        for ch in self.reply:
            yield ch


class FakeEmbedder:
    """确定性嵌入：用字母频次当向量，保证同文本相似度高。"""

    def embed_texts(self, texts):
        return [[float(t.lower().count(ch)) for ch in string.ascii_lowercase] for t in texts]

    def embed_query(self, query):
        return self.embed_texts([query])[0]
