"""pytest 共享 fixture。

关键点：所有测试都不加载真实 Embedding 模型、不连真实向量库、不调用真实 LLM。
通过显式注入（依赖倒置）或 app.dependency_overrides 替换为内存替身。
"""
import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.api import documents as docs_api
from app.config import get_settings
from app.core.rag import RAGPipeline
from app.main import app
from app.services.chat_service import ChatService
from app.services.ingestion import IngestionService
from tests.fakes import FakeLLM, FakeStore


@pytest.fixture(autouse=True)
def _clean_overrides():
    """每个用例结束后清理依赖覆盖，避免相互污染。"""
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    """FastAPI 测试客户端（会触发 lifespan）。"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_store():
    return FakeStore()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def ingestion(tmp_path, fake_store, monkeypatch):
    """真实 IngestionService + 内存 store + 临时上传目录。"""
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "max_upload_size_mb", 20)
    service = IngestionService(store=fake_store)
    app.dependency_overrides[docs_api.get_ingestion_service] = lambda: service
    return service


@pytest.fixture
def chat_env(fake_store, fake_llm):
    """真实 ChatService/RAGPipeline + 内存 store + LLM 替身。"""
    rag = RAGPipeline(llm=fake_llm, store=fake_store)
    service = ChatService(rag=rag)
    app.dependency_overrides[chat_api.get_chat_service] = lambda: service
    return service, fake_store, fake_llm


@pytest.fixture
def settings():
    """全局配置。"""
    return get_settings()


@pytest.fixture
def sample_question() -> str:
    return "这份文档讲了什么?"
