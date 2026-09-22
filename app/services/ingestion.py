"""文档入库服务：解析 → 分块 → 向量化 → 入向量库。

完整流程：
    UploadFile → 保存到 data/documents/ → DocumentLoader 解析 →
    Chunker 分块 → Embedding → VectorStore.add
"""
import uuid
from pathlib import Path

from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core.vector_store import get_vector_store
from app.utils.loaders import SUPPORTED_TYPES, load_document


class IngestionService:
    """把上传的文档处理成可检索的向量。"""

    def __init__(self) -> None:
        self._store = get_vector_store()
        s = get_settings()
        self._upload_dir = s.project_root / "data" / "documents"
        self._upload_dir.mkdir(parents=True, exist_ok=True)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )

    async def ingest(self, file: UploadFile) -> str:
        """入库一个文档，返回 doc_id。"""
        # 1. 校验类型
        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED_TYPES:
            raise ValueError(f"不支持的文件类型: {ext}")

        # 2. 保存文件
        doc_id = str(uuid.uuid4())
        save_path = self._upload_dir / f"{doc_id}{ext}"
        save_path.write_bytes(await file.read())

        # 3. 解析
        text = load_document(save_path)

        # 4. 分块
        chunks = self._chunk(text)

        # 5. 入向量库
        self._store.add(doc_id=doc_id, doc_name=file.filename or doc_id, chunks=chunks)

        return doc_id

    def _chunk(self, text: str) -> list[str]:
        """把长文本切成小块。"""
        return [c.strip() for c in self._splitter.split_text(text) if c.strip()]

    async def list_all(self) -> dict:
        """列出已入库文档。"""

        return {"documents": self._store.list_docs()}

    async def delete(self, doc_id: str) -> dict:
        """删除文档及其向量。"""
        deleted = self._store.delete(doc_id)
        return {"doc_id": doc_id, "deleted_chunks": deleted}