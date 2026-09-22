"""RAG 流水线：检索 + 生成。

完整流程：
    用户问题 → [改写] → Embedding → 向量检索 → 取 top-k 块 →
    拼装 Prompt → LLM 生成 → 返回答案 + 引用
"""
from app.config import get_settings
from app.core.llm import get_llm
from app.core.prompts import RAG_PROMPT
from app.core.vector_store import get_vector_store


class RAGPipeline:
    """RAG 核心管线。"""

    def __init__(self) -> None:
        self._llm = get_llm()
        self._store = get_vector_store()

    async def answer(self, question: str, top_k: int | None = None) -> dict:
        """回答一个独立问题（不维护多轮上下文）。

        Returns:
            {"answer": str, "citations": [{doc_id, doc_name, chunk_text, score}, ...]}
        """
        k = top_k or get_settings().top_k

        # 1. 检索相关 chunks
        retrieved = self._store.search(question, top_k=k)

        # 2. 拼装 Prompt
        context = "\n\n".join(
            f"[来源 {i + 1}: {r['doc_name']}]\n{r['chunk_text']}"
            for i, r in enumerate(retrieved)
        )
        messages = [
            {"role": "system", "content": RAG_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]

        # 3. LLM 生成
        answer = await self._llm.chat(messages, temperature=0.3)

        return {
            "answer": answer,
            "citations": [
                {
                    "doc_id": r["doc_id"],
                    "doc_name": r["doc_name"],
                    "chunk_text": r["chunk_text"],
                    "score": r["score"],
                }
                for r in retrieved
            ],
        }

    async def stream_answer(self, question: str, top_k: int | None = None):
        """流式版本：yield字典事件。

        Yields:
            {"type": "delta", "text": str}: 答案片段
            {"type": "done", "citations": list[dict]}: 流结束标记 + 引用
        """
        k = top_k or get_settings().top_k
        retrieved = self._store.search(question, top_k=k)

        context = "\n\n".join(
            f"[来源 {i + 1}: {r['doc_name']}]\n{r['chunk_text']}"
            for i, r in enumerate(retrieved)
        )
        messages = [
            {"role": "system", "content": RAG_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]

        async for delta in self._llm.stream_chat(messages, temperature=0.3):
            yield {"type": "delta", "text": delta}

        # 流结束事件，附带引用（前端用这个画"来源 1、来源 2..."卡片）
        yield {
            "type": "done",
            "citations": [
                {
                    "doc_id": r["doc_id"],
                    "doc_name": r["doc_name"],
                    "chunk_text": r["chunk_text"],
                    "score": r["score"],
                }
                for r in retrieved
            ],
        }


_pipeline: RAGPipeline | None = None


def get_rag() -> RAGPipeline:
    """全局单例。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline