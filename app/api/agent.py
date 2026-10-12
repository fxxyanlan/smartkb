"""/agent 端点：LangGraph 修正型 RAG（混合 + Rerank 最优链路 + 评估/改写回环）。

与 /chat 的区别：/chat 是单向线性链路；/agent/chat 会在检索后评估上下文是否充分，
不充分则改写问题重新检索，因此更慢（多 1–2 次 LLM 调用），但召回不佳时更可靠。
"""
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agent.service import AgentService
from app.api.schemas import AgentChatResponse, ChatRequest
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

_service: AgentService | None = None


def get_agent_service() -> AgentService:
    """懒加载单例；测试时可通过 app.dependency_overrides 替换。"""
    global _service
    if _service is None:
        # 注入向量库：会话据此感知「中途上传/删除文档」，从而重置已失效的历史
        _service = AgentService(store=get_vector_store())
    return _service


@router.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    req: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentChatResponse:
    """智能体问答：检索 → 评估 →（必要时改写重检）→ 生成。"""
    result = await service.answer(req.question, req.conversation_id, req.doc_ids)
    return AgentChatResponse(**result)


@router.post("/agent/chat/stream")
async def agent_chat_stream(
    req: ChatRequest,
    service: AgentService = Depends(get_agent_service),
):
    """智能体流式问答（SSE 协议）：事件结构与 /chat/stream 一致。"""

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    async def gen():
        try:
            async for event in service.stream_answer(
                req.question, req.conversation_id, req.doc_ids
            ):
                yield _sse(event)
        except Exception:  # noqa: BLE001 - 流已开始，只能以事件形式回传错误
            logger.exception("Agent 流式问答失败")
            yield _sse({"type": "error", "message": "生成回答时发生错误，请稍后重试"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
