"""AgentService：LangGraph 修正型 RAG 的高层编排（会话历史 + 事件流）。

与 services/chat_service.py 并列：/chat 走线性链路，/agent/chat 走图编排。
历史污染防护与 ChatService 一致：会话绑定文档范围（检索过滤）+ 文档变更即重置历史。
"""
import logging

from app.agent.graph import get_agent_graph
from app.config import get_settings
from app.services.conversation import ConversationStore

logger = logging.getLogger(__name__)


class AgentService:
    """封装修正型 RAG 的问答流程。"""

    def __init__(self, graph=None, store=None) -> None:
        self._graph = graph or get_agent_graph()
        self._store = ConversationStore()
        self._max_conversations = self._store.max_conversations
        #: 向量库（用于读取数据版本号）；未注入时关闭「文档变更即重置历史」
        self._vector_store = store

    @property
    def conversations(self) -> dict:
        """仅供观测/测试：暴露底层会话表。"""
        return self._store.conversations

    async def answer(
        self, question: str, conversation_id: str | None, doc_ids: list[str] | None = None
    ) -> dict:
        """回答问题（非流式），支持多轮上下文与文档范围限定。"""
        cid, conversation = self._store.get_or_create(
            conversation_id, self._max_conversations
        )
        self._sync_scope(conversation, doc_ids)

        final = await self._graph.ainvoke(
            self._initial_state(question, conversation.history, conversation.doc_ids)
        )
        answer = final.get("answer", "")

        self._remember(conversation.history, question, answer)

        return {
            "answer": answer,
            "citations": final.get("citations", []),
            "conversation_id": cid,
            "rewrites": final.get("rewrites", 0),
        }

    async def stream_answer(
        self, question: str, conversation_id: str | None, doc_ids: list[str] | None = None
    ):
        """流式回答：逐字 yield 字典事件。

        Yields:
            {"type": "delta", "text": str}: 答案片段
            {"type": "done", "citations": [...], "conversation_id": str, "rewrites": int}
        """
        cid, conversation = self._store.get_or_create(
            conversation_id, self._max_conversations
        )
        self._sync_scope(conversation, doc_ids)

        final: dict = {}
        emitted = False
        # stream_mode 同时订阅：custom=generate 节点逐字写的 token；values=每一步后的完整状态
        async for mode, chunk in self._graph.astream(
            self._initial_state(question, conversation.history, conversation.doc_ids),
            stream_mode=["custom", "values"],
        ):
            if mode == "custom":
                emitted = True
                yield chunk
            elif mode == "values":
                final = chunk

        answer = final.get("answer", "")
        # 兜底（如检索为空走 no_context 时没有任何流式增量）：把整段答案作为单个增量发出
        if not emitted and answer:
            yield {"type": "delta", "text": answer}

        self._remember(conversation.history, question, answer)

        yield {
            "type": "done",
            "citations": final.get("citations", []),
            "conversation_id": cid,
            "rewrites": final.get("rewrites", 0),
        }

    # ----- 内部辅助 -----

    @staticmethod
    def _initial_state(
        question: str, history: list[dict], doc_ids: list[str] | None = None
    ) -> dict:
        s = get_settings()
        return {
            "question": question,
            "query": question,  # 初始检索词即原问题，后续可能被 rewrite 覆盖
            "history": list(history),
            "doc_ids": list(doc_ids) if doc_ids else None,
            "top_k": s.top_k,
            "rewrites": 0,
        }

    def _sync_scope(self, conversation, doc_ids: list[str] | None) -> None:
        """把会话对齐到当前文档范围与数据版本；文档变更时清空历史。"""
        revision = getattr(self._vector_store, "revision", None) if self._vector_store else None
        self._store.rebind(conversation, doc_ids, revision)

    def _remember(self, history: list[dict], question: str, answer: str) -> None:
        self._store.remember(history, question, answer, get_settings().max_history_turns)

