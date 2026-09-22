"""API 数据契约：前后端共享的字段定义。"""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """问答请求体。"""

    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")


class Citation(BaseModel):
    """引用来源：哪篇文档的哪一段。"""

    doc_id: str
    doc_name: str
    chunk_text: str
    score: float


class ChatResponse(BaseModel):
    """问答响应。"""

    answer: str
    citations: list[Citation] = []
    conversation_id: str


class UploadResponse(BaseModel):
    """上传响应。"""

    doc_id: str
    filename: str
    status: str