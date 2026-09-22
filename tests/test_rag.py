"""RAG 流水线测试。"""
import pytest

from app.core.rag import RAGPipeline


@pytest.mark.asyncio
async def test_rag_pipeline_instantiates():
    """冒烟测试：RAG 流水线应该能实例化。"""
    rag = RAGPipeline()
    assert rag is not None


@pytest.mark.asyncio
async def test_rag_answer_returns_expected_shape():
    """RAG 答案应该包含 answer 和 citations 字段。

    依赖向量库和真实 LLM 调用，默认 skip。
    """
    pytest.skip("依赖向量库 + LLM，待接入测试向量库后启用")


def test_chunk_split_basic():
    """分块逻辑测试。"""
    from app.services.ingestion import IngestionService

    svc = IngestionService()
    text = "段落一。\n\n" + ("段落二" * 100)
    chunks = svc._chunk(text)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)