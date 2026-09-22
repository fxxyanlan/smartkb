"""高层聊天服务：维护会话历史、调用 RAG。

MVP 用内存 dict 存历史；生产应换成 SQLite / Redis。
"""
from app.core.rag import get_rag


class ChatService:
    """封装完整的问答流程。"""

    def __init__(self) -> None:
        self._rag = get_rag()
        # 内存会话表。生产应换成持久化存储。
        self._conversations: dict[str, list[dict]] = {}

    async def answer(self, question: str, conversation_id: str | None) -> dict:
        """回答问题（非流式），支持多轮上下文。

        MVP 简化：只把历史附加进 messages，但 RAG 检索仍基于当前 question。
        进阶做法：先用 LLM 把多轮历史改写成独立 query，再去检索。
        """
        # 1. 获取 / 创建会话
        cid = conversation_id or self._new_id()
        history = self._conversations.setdefault(cid, [])

        # 2. 调用 RAG
        result = await self._rag.answer(question)

        # 3. 更新历史
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

        # 4. 裁剪历史，防止超出上下文窗口
        self._trim_history(history)

        # 5. 返回
        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "conversation_id": cid,
        }

    async def stream_answer(self, question: str, conversation_id: str | None):
        """流式回答：逐字 yield 字典事件。

        Yields:
            {"type": "delta", "text": str}: 答案片段
            {"type": "done", "citations": [...], "conversation_id": str}: 流结束标记
        """
        cid = conversation_id or self._new_id()
        history = self._conversations.setdefault(cid, [])

        full: list[str] = []
        async for event in self._rag.stream_answer(question):
            # 把 conversation_id 注入 done 事件，让前端知道自己的会话 ID
            if event["type"] == "done":
                event["conversation_id"] = cid
            if event["type"] == "delta":
                full.append(event["text"])
            yield event

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "".join(full)})
        self._trim_history(history)

    @staticmethod
    def _new_id() -> str:
        import uuid
        return f"conv_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _trim_history(history: list[dict], max_turns: int = 10) -> None:
        """保留最近 max_turns 轮对话。"""
        max_messages = max_turns * 2
        if len(history) > max_messages:
            history[:] = history[-max_messages:]
