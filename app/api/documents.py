"""/documents 文档管理端点。"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import UploadResponse
from app.services.ingestion import IngestionService

router = APIRouter()
ingestion = IngestionService()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """上传文档并入库：解析 → 分块 → 向量化 → 入向量库。"""
    try:
        doc_id = await ingestion.ingest(file)
        return UploadResponse(doc_id=doc_id, filename=file.filename or "", status="ingested")
    except ValueError as e:
        # 文件类型不支持等客户端错误
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/documents")
async def list_documents() -> dict:
    """列出已入库的所有文档。"""
    return await ingestion.list_all()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """删除指定文档及其向量。"""
    try:
        return await ingestion.delete(doc_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e