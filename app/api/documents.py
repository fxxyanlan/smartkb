"""/documents 文档管理端点。

只做 HTTP 相关的事：解析请求、调用 service、返回结果。
业务错误（类型/大小/解析/未找到）由领域异常抛出，统一在 main.py 映射为状态码。
"""
from fastapi import APIRouter, Depends, File, UploadFile

from app.api.schemas import UploadResponse
from app.services.ingestion import IngestionService

router = APIRouter()

_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    """懒加载单例；测试时可通过 app.dependency_overrides 替换。"""
    global _service
    if _service is None:
        _service = IngestionService()
    return _service


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    """上传文档并入库：解析 → 分块 → 向量化 → 入向量库。"""
    result = await service.ingest(file)
    return UploadResponse(
        doc_id=result["doc_id"], filename=result["doc_name"], status=result["status"]
    )


@router.get("/documents")
async def list_documents(
    service: IngestionService = Depends(get_ingestion_service),
) -> dict:
    """列出已入库的所有文档。"""
    return await service.list_all()


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> dict:
    """删除指定文档及其向量；文档不存在返回 404。"""
    return await service.delete(doc_id)
