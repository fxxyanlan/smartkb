"""/chat 聊天端点。"""
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()

_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """懒加载单例；测试时可通过 app.dependency_overrides 替换。"""
    global _service
    if _service is None:
        _service = ChatService()
    return _service


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """问答接口：接收问题，返回基于知识库的答案。"""
    result = await service.answer(req.question, req.conversation_id)
    return ChatResponse(**result)


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    """流式问答（SSE 协议）：前端按 \\n\\n 切分，每个 data: 行 JSON.parse。"""

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    async def gen():
        try:
            async for event in service.stream_answer(req.question, req.conversation_id):
                yield _sse(event)
        except Exception:  # noqa: BLE001 - 流已开始，只能以事件形式回传错误
            logger.exception("流式问答失败")
            yield _sse({"type": "error", "message": "生成回答时发生错误，请稍后重试"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
