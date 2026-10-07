"""API 数据契约：前后端共享的字段定义。"""

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """问答请求体。"""

    user_id: str | None = None
    conversation_id: str | None = None
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")

    @field_validator("question")
    @classmethod
    def _strip_and_require_text(cls, v: str) -> str:
        """去掉首尾空白，并拒绝纯空白问题。"""
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        return v


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
    """上传响应。

    status: ingested=新入库；exists=内容重复，复用已有文档（未重复入库）。
    """

    doc_id: str
    filename: str
    status: str
