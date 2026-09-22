"""/chat 聊天端点。"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from app.api.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter()

# 模块级单例。生产建议用 FastAPI Depends 注入，便于测试替换。
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """问答接口：接收问题，返回基于知识库的答案。"""
    try:
        result = await chat_service.answer(req.question, req.conversation_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答（SSE 协议）：前端按 \\n\\n 切分，每个 data: 行 JSON.parse。"""
    async def gen():
        async for event in chat_service.stream_answer(req.question, req.conversation_id):
            # SSE 协议：每个事件是 data: <单行 JSON>\\n\\n
            # ensure_ascii=False 保留中文不转义（可读性 + 网络字节略多）
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream; charset=utf-8")

