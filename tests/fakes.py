"""测试替身：不依赖真实模型 / 向量库 / LLM 的轻量假实现。"""
import asyncio
import string


class FakeStore:
    """内存向量库替身，接口与 VectorStore 保持一致。"""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self._revision = 0

    @property
    def revision(self):
        """数据版本号，供混合检索判断索引是否需重建。"""
        return self._revision

    def add(self, doc_id, doc_name, chunks, content_hash=""):
        self.docs[doc_id] = {
            "doc_name": doc_name,
            "chunks": list(chunks),
            "hash": content_hash,
        }
        self._revision += 1
        return [f"{doc_id}_{i}" for i in range(len(chunks))]

    def search(self, query, top_k=None):
        results = []
        for did, d in self.docs.items():
            for i, c in enumerate(d["chunks"]):
                score = float(len(set(query) & set(c)))
                results.append(
                    {
                        "chunk_id": f"{did}_{i}",
                        "doc_id": did,
                        "doc_name": d["doc_name"],
                        "chunk_text": c,
                        "score": score,
                    }
                )
        results.sort(key=lambda r: -r["score"])
        return results[: top_k or 4]

    def get_all_chunks(self):
        """全量取块（供 BM25 建索引）。"""
        return [
            {
                "chunk_id": f"{did}_{i}",
                "doc_id": did,
                "doc_name": d["doc_name"],
                "chunk_text": c,
            }
            for did, d in self.docs.items()
            for i, c in enumerate(d["chunks"])
        ]

    def find_by_hash(self, content_hash):
        for did, d in self.docs.items():
            if d["hash"] and d["hash"] == content_hash:
                return {"doc_id": did, "doc_name": d["doc_name"]}
        return None

    def delete(self, doc_id):
        d = self.docs.pop(doc_id, None)
        if d:
            self._revision += 1
        return len(d["chunks"]) if d else 0

    def list_docs(self):
        return [
            {"doc_id": did, "doc_name": d["doc_name"], "chunks": len(d["chunks"])}
            for did, d in self.docs.items()
        ]

    def count(self):
        return sum(len(d["chunks"]) for d in self.docs.values())


class FakeLLM:
    """LLM 替身：记录收到的 messages，便于断言。

    流式场景的可控失败钩子：
        error:        一旦设置，所有调用都抛该异常
        fail_times:   前 N 次流式调用立即失败（用于验证"未输出即重试"）
        fail_after:   吐出 N 个字后中途失败（用于验证"已输出不重试"）
        stall_seconds: 首字前阻塞时长（用于验证首字超时）
    """

    def __init__(
        self,
        reply="这是基于文档的回答。[来源 1]",
        fail=False,
        error=None,
        fail_times=0,
        fail_after=None,
        stall_seconds=0.0,
    ) -> None:
        self.reply = reply
        self.error = error or (RuntimeError("llm boom") if fail else None)
        self.fail_times = fail_times
        self.fail_after = fail_after
        self.stall_seconds = stall_seconds
        self.calls: list[list[dict]] = []
        self._stream_calls = 0

    async def chat(self, messages, temperature=0.3, max_tokens=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply

    async def stream_chat(self, messages, temperature=0.3):
        self.calls.append(messages)
        self._stream_calls += 1

        if self.error:
            raise self.error
        if self._stream_calls <= self.fail_times:
            raise RuntimeError("stream boom")
        if self.stall_seconds:
            await asyncio.sleep(self.stall_seconds)

        for i, ch in enumerate(self.reply):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("stream broke midway")
            yield ch


class FakeEmbedder:
    """确定性嵌入：用字母频次当向量，保证同文本相似度高。"""

    def embed_texts(self, texts):
        return [[float(t.lower().count(ch)) for ch in string.ascii_lowercase] for t in texts]

    def embed_query(self, query):
        return self.embed_texts([query])[0]
