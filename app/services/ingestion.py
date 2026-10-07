"""文档入库服务：解析 → 分块 → 向量化 → 入向量库。

完整流程：
    UploadFile → 校验(类型/大小/去重) → 保存到 upload_dir → 解析 →
    分块 → Embedding → VectorStore.add

错误一律抛领域异常（core/errors.py），由 api 层统一映射为 HTTP 状态码。
"""
import asyncio
import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core.errors import (
    DocumentNotFoundError,
    DocumentParseError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.core.vector_store import get_vector_store
from app.utils.loaders import SUPPORTED_TYPES, load_document

logger = logging.getLogger(__name__)


class IngestionService:
    """把上传的文档处理成可检索的向量。"""

    def __init__(self, store=None) -> None:
        self._store = store or get_vector_store()
        s = get_settings()
        self._upload_dir = s.upload_path
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = s.max_upload_bytes

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )

    async def ingest(self, file: UploadFile) -> dict:
        """入库一个文档。

        Returns:
            {"doc_id": str, "doc_name": str, "status": "ingested" | "exists"}
        """
        filename = file.filename or "unnamed"
        ext = Path(filename).suffix.lower()

        # 1. 校验类型
        if ext not in SUPPORTED_TYPES:
            raise UnsupportedFileTypeError(
                f"不支持的文件类型: {ext or '(无后缀)'}，仅支持 {sorted(SUPPORTED_TYPES)}"
            )

        # 2. 读取并校验大小 / 空文件
        data = await file.read()
        if not data:
            raise EmptyDocumentError("上传的文件为空")
        if len(data) > self._max_bytes:
            raise FileTooLargeError(
                f"文件大小 {len(data) / 1024 / 1024:.1f}MB 超过上限 "
                f"{self._max_bytes // 1024 // 1024}MB"
            )

        # 3. 内容去重：同一份文件重复上传时直接复用，避免检索结果重复
        content_hash = hashlib.sha256(data).hexdigest()
        existing = await asyncio.to_thread(self._store.find_by_hash, content_hash)
        if existing:
            logger.info("重复上传，复用已有文档 | doc_id=%s", existing["doc_id"])
            return {
                "doc_id": existing["doc_id"],
                "doc_name": existing["doc_name"],
                "status": "exists",
            }

        # 4. 落盘
        doc_id = str(uuid.uuid4())
        save_path = self._upload_dir / f"{doc_id}{ext}"
        save_path.write_bytes(data)

        # 5. 解析 + 分块（解析失败视为客户端文件问题，返回 400 而非 500）
        try:
            text = await asyncio.to_thread(load_document, save_path)
        except Exception as e:  # noqa: BLE001 - 统一转成领域异常
            self._safe_unlink(save_path)
            logger.warning("文档解析失败 | doc_id=%s err=%s", doc_id, e)
            raise DocumentParseError() from e

        if not text.strip():
            self._safe_unlink(save_path)
            raise EmptyDocumentError("未能从文档中提取到文本（可能是扫描件或空文档）")

        chunks = await asyncio.to_thread(self._chunk, text)
        if not chunks:
            self._safe_unlink(save_path)
            raise EmptyDocumentError("文档分块后为空")

        # 6. 入向量库
        await asyncio.to_thread(
            self._store.add, doc_id, filename, chunks, content_hash
        )
        logger.info("入库完成 | doc_id=%s chunks=%d", doc_id, len(chunks))

        return {"doc_id": doc_id, "doc_name": filename, "status": "ingested"}

    def _chunk(self, text: str) -> list[str]:
        """把长文本切成小块。"""
        return [c.strip() for c in self._splitter.split_text(text) if c.strip()]

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        """尽力删除文件。

        Windows 下解析库可能仍占用文件句柄（如 PyMuPDF 解析失败时），
        此时 unlink 会抛 PermissionError——必须吞掉，否则会把 400 变成 500。
        """
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("临时文件清理失败 | path=%s", path)

    async def list_all(self) -> dict:
        """列出已入库文档。"""
        docs = await asyncio.to_thread(self._store.list_docs)
        return {"documents": docs}

    async def delete(self, doc_id: str) -> dict:
        """删除文档及其向量；文档不存在时抛 DocumentNotFoundError（→ 404）。"""
        deleted = await asyncio.to_thread(self._store.delete, doc_id)
        if deleted == 0:
            raise DocumentNotFoundError(f"文档不存在: {doc_id}")

        # 顺带清理落盘的原始文件（尽力而为，失败不影响删除结果）
        for p in self._upload_dir.glob(f"{doc_id}.*"):
            try:
                p.unlink()
            except OSError:
                logger.warning("原始文件删除失败 | path=%s", p)

        logger.info("删除完成 | doc_id=%s chunks=%d", doc_id, deleted)
        return {"doc_id": doc_id, "deleted_chunks": deleted}
