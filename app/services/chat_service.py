"""高层聊天服务：维护会话历史、调用 RAG。

MVP 用内存 dict 存历史；生产应换成 SQLite / Redis。
"""
import logging
import uuid
from collections import OrderedDict

from app.config import get_settings
from app.core.rag import get_rag

logger = logging.getLogger(__name__)


class ChatService:
    """封装完整的问答流程。"""

    def __init__(self, rag=None) -> None:
        self._rag = rag or get_rag()
        # 内存会话表（按插入顺序，超量时淘汰最旧的，避免无限增长）。
        self._conversations: OrderedDict[str, list[dict]] = OrderedDict()
        self._max_conversations = 200

    async def answer(self, question: str, conversation_id: str | None) -> dict:
        """回答问题（非流式），支持多轮上下文。"""
        cid, history = self._get_conversation(conversation_id)

        # 把历史传给 RAG，使模型能理解追问（检索仍基于当前问题）
        result = await self._rag.answer(question, history=list(history))

        self._remember(history, question, result["answer"])

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
        cid, history = self._get_conversation(conversation_id)

        full: list[str] = []
        async for event in self._rag.stream_answer(question, history=list(history)):
            # 把 conversation_id 注入 done 事件，让前端知道自己的会话 ID
            if event["type"] == "done":
                event["conversation_id"] = cid
            elif event["type"] == "delta":
                full.append(event["text"])
            yield event

        self._remember(history, question, "".join(full))

    # ----- 内部辅助 -----

    def _get_conversation(self, conversation_id: str | None) -> tuple[str, list[dict]]:
        """取出或新建会话，并把最近使用的会话移到末尾（LRU 语义）。"""
        cid = conversation_id or self._new_id()
        if cid not in self._conversations:
            self._conversations[cid] = []
            if len(self._conversations) > self._max_conversations:
                evicted, _ = self._conversations.popitem(last=False)
                logger.info("会话数超限，淘汰最旧会话 | cid=%s", evicted)
        self._conversations.move_to_end(cid)
        return cid, self._conversations[cid]

    def _remember(self, history: list[dict], question: str, answer: str) -> None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        self._trim_history(history)

    @staticmethod
    def _new_id() -> str:
        return f"conv_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _trim_history(history: list[dict], max_turns: int | None = None) -> None:
        """保留最近 max_turns 轮对话。"""
        turns = max_turns or get_settings().max_history_turns
        max_messages = turns * 2
        if len(history) > max_messages:
            del history[:-max_messages]
