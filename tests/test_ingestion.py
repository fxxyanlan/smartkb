"""入库服务单元测试：分块、去重、错误映射。"""
import io

import pytest
from starlette.datastructures import UploadFile

from app.core.errors import (
    DocumentNotFoundError,
    DocumentParseError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)


def make_file(name: str, data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=name)


def test_chunk_split_produces_multiple_chunks(ingestion):
    """超过 chunk_size 的长文本应被切成多块（回归：旧测试用例本身有误）。"""
    text = "这是一句用于测试的句子。" * 200  # 约 2400 字，远超 500
    chunks = ingestion._chunk(text)

    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


async def test_ingest_markdown(ingestion):
    result = await ingestion.ingest(make_file("guide.md", "文档正文内容。".encode()))

    assert result["status"] == "ingested"
    assert result["doc_name"] == "guide.md"


async def test_duplicate_content_is_reused(ingestion):
    data = "完全一样的内容。".encode()

    first = await ingestion.ingest(make_file("a.md", data))
    second = await ingestion.ingest(make_file("b.md", data))

    assert first["status"] == "ingested"
    assert second["status"] == "exists"
    assert second["doc_id"] == first["doc_id"]


async def test_unsupported_type_raises(ingestion):
    with pytest.raises(UnsupportedFileTypeError):
        await ingestion.ingest(make_file("malware.exe", b"MZbinary"))


async def test_empty_file_raises(ingestion):
    with pytest.raises(EmptyDocumentError):
        await ingestion.ingest(make_file("empty.md", b""))


async def test_corrupt_pdf_raises_parse_error(ingestion):
    """伪造的 PDF 内容应被识别为解析失败/空文档，而非 500。"""
    with pytest.raises((DocumentParseError, EmptyDocumentError)):
        await ingestion.ingest(make_file("fake.pdf", b"%PDF-1.4 not a real pdf body"))


async def test_delete_missing_raises(ingestion):
    with pytest.raises(DocumentNotFoundError):
        await ingestion.delete("no-such-doc")


async def test_delete_existing_removes_chunks(ingestion):
    doc_id = (await ingestion.ingest(make_file("x.md", "待删除。".encode())))["doc_id"]

    result = await ingestion.delete(doc_id)

    assert result["deleted_chunks"] >= 1
