"""RAG 流水线：检索 + 生成。

完整流程：
    用户问题 → Embedding → 向量检索 → 取 top-k 块 →
    （System Prompt + 多轮历史 + 当前问题）→ LLM 生成 → 返回答案 + 引用

设计要点：
    - 检索是同步 CPU/IO 操作，放在线程池里跑，避免阻塞事件循环。
    - 检索不到任何内容时直接返回固定话术，不调用 LLM（省成本 + 防幻觉）。
    - 多轮历史由 services 层传入，本层只负责拼装 messages。
"""
import asyncio
import logging
import time

from app.config import get_settings
from app.core.llm import get_llm
from app.core.prompts import RAG_PROMPT
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# 检索为空时的固定回答（不调用大模型，避免在无依据时编造）
NO_CONTEXT_ANSWER = (
    "知识库中暂时没有检索到与问题相关的内容。"
    "请先上传相关文档，或换一种更贴近文档内容的问法。"
)


class RAGPipeline:
    """RAG 核心管线。"""

    def __init__(self, llm=None, store=None) -> None:
        # 允许注入依赖（便于测试）；默认使用全局单例。
        self._llm = llm or get_llm()
        self._store = store or get_vector_store()

    def _retrieve(self, question: str, k: int) -> list[dict]:
        """同步检索（在线程池中调用）。"""
        return self._store.search(question, top_k=k)

    def _build_messages(
        self, question: str, retrieved: list[dict], history: list[dict] | None
    ) -> list[dict]:
        """拼装 messages：system(含参考内容) + 历史轮次 + 当前问题。"""
        context = "\n\n".join(
            f"[来源 {i + 1}: {r['doc_name']}]\n{r['chunk_text']}"
            for i, r in enumerate(retrieved)
        )
        messages: list[dict] = [
            {"role": "system", "content": RAG_PROMPT.format(context=context)}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _citations(retrieved: list[dict]) -> list[dict]:
        return [
            {
                "doc_id": r["doc_id"],
                "doc_name": r["doc_name"],
                "chunk_text": r["chunk_text"],
                "score": r["score"],
            }
            for r in retrieved
        ]

    async def answer(
        self, question: str, history: list[dict] | None = None, top_k: int | None = None
    ) -> dict:
        """回答一个独立问题（history 用于多轮上下文）。

        Returns:
            {"answer": str, "citations": [{doc_id, doc_name, chunk_text, score}, ...]}
        """
        k = top_k or get_settings().top_k

        # 1. 检索（线程池，避免阻塞事件循环）
        t0 = time.perf_counter()
        retrieved = await asyncio.to_thread(self._retrieve, question, k)
        logger.info("检索完成 | k=%d 命中=%d 耗时=%.0fms", k, len(retrieved),
                    (time.perf_counter() - t0) * 1000)

        # 2. 无检索结果：直接兜底，不调用 LLM
        if not retrieved:
            return {"answer": NO_CONTEXT_ANSWER, "citations": []}

        # 3. 拼装 Prompt（含多轮历史）并生成
        messages = self._build_messages(question, retrieved, history)
        t1 = time.perf_counter()
        answer = await self._llm.chat(messages, temperature=0.3)
        logger.info("生成完成 | 耗时=%.0fms", (time.perf_counter() - t1) * 1000)

        return {"answer": answer, "citations": self._citations(retrieved)}

    async def stream_answer(
        self, question: str, history: list[dict] | None = None, top_k: int | None = None
    ):
        """流式版本：yield 字典事件。

        Yields:
            {"type": "delta", "text": str}: 答案片段
            {"type": "done", "citations": list[dict]}: 流结束标记 + 引用
        """
        k = top_k or get_settings().top_k
        retrieved = await asyncio.to_thread(self._retrieve, question, k)
        logger.info("检索完成 | k=%d 命中=%d", k, len(retrieved))

        if not retrieved:
            yield {"type": "delta", "text": NO_CONTEXT_ANSWER}
            yield {"type": "done", "citations": []}
            return

        messages = self._build_messages(question, retrieved, history)
        async for delta in self._llm.stream_chat(messages, temperature=0.3):
            yield {"type": "delta", "text": delta}

        # 流结束事件，附带引用（前端用这个画"来源 1、来源 2..."卡片）
        yield {"type": "done", "citations": self._citations(retrieved)}


_pipeline: RAGPipeline | None = None


def get_rag() -> RAGPipeline:
    """全局单例。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
