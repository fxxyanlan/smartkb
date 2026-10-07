"""API 端点契约测试：状态码、响应结构、错误处理、SSE 协议。"""
import json

import pytest

from app.api import chat as chat_api
from app.core.rag import NO_CONTEXT_ANSWER, RAGPipeline
from app.main import app
from app.services.chat_service import ChatService
from tests.fakes import FakeLLM

# ---------- 健康检查 ----------

def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_reports_vector_store(client, fake_store, monkeypatch):
    from app.api import health as health_api

    monkeypatch.setattr(health_api, "get_vector_store", lambda: fake_store)
    fake_store.add("d1", "a.md", ["x", "y"])

    resp = client.get("/api/v1/health/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["vector_store"]["chunks"] == 2


# ---------- 请求校验 ----------

@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "   "}, {}],
)
def test_chat_validation_rejects_bad_question(client, payload):
    resp = client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 422


# ---------- 问答 ----------

def test_chat_success_returns_answer_and_citations(client, chat_env):
    _, store, llm = chat_env
    store.add("d1", "手册.md", ["这是文档中的一个片段"])

    resp = client.post("/api/v1/chat", json={"question": "这段讲了什么"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == llm.reply
    assert body["citations"][0]["doc_name"] == "手册.md"
    assert body["conversation_id"].startswith("conv_")


def test_chat_empty_knowledge_base_does_not_call_llm(client, chat_env):
    """知识库为空时应兜底回答，且不调用大模型（防幻觉 + 省成本）。"""
    _, _store, llm = chat_env

    resp = client.post("/api/v1/chat", json={"question": "任意问题"})

    assert resp.status_code == 200
    assert resp.json()["answer"] == NO_CONTEXT_ANSWER
    assert resp.json()["citations"] == []
    assert llm.calls == []


def test_chat_reuses_conversation_id(client, chat_env):
    _, store, _ = chat_env
    store.add("d1", "手册.md", ["片段"])

    first = client.post("/api/v1/chat", json={"question": "第一问"}).json()
    second = client.post(
        "/api/v1/chat",
        json={"question": "第二问", "conversation_id": first["conversation_id"]},
    ).json()

    assert second["conversation_id"] == first["conversation_id"]


def test_chat_llm_unavailable_returns_503(client, fake_store):
    """大模型不可用（未配置/鉴权失败）应返回 503，而不是 500。"""
    from app.core.errors import LLMUnavailableError

    service = ChatService(
        rag=RAGPipeline(llm=FakeLLM(error=LLMUnavailableError()), store=fake_store)
    )
    app.dependency_overrides[chat_api.get_chat_service] = lambda: service
    fake_store.add("d1", "手册.md", ["片段"])

    resp = client.post("/api/v1/chat", json={"question": "问题"})

    assert resp.status_code == 503


# ---------- 流式问答 ----------

def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[5:].strip())
        for line in text.split("\n")
        if line.startswith("data:")
    ]


def test_stream_emits_delta_then_done(client, chat_env):
    _, store, llm = chat_env
    store.add("d1", "手册.md", ["片段"])

    with client.stream("POST", "/api/v1/chat/stream", json={"question": "问题"}) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = "".join(resp.iter_text())

    events = _parse_sse(text)
    assert events[0]["type"] == "delta"
    assert events[-1]["type"] == "done"
    assert events[-1]["conversation_id"].startswith("conv_")
    answer = "".join(e["text"] for e in events if e["type"] == "delta")
    assert answer == llm.reply


def test_stream_returns_error_event_on_llm_failure(client, fake_store):
    """LLM 中途失败时，应以 error 事件回传而不是让连接静默中断。"""
    failing = FakeLLM(fail=True)
    service = ChatService(rag=RAGPipeline(llm=failing, store=fake_store))
    app.dependency_overrides[chat_api.get_chat_service] = lambda: service
    fake_store.add("d1", "手册.md", ["片段"])

    with client.stream("POST", "/api/v1/chat/stream", json={"question": "问题"}) as resp:
        text = "".join(resp.iter_text())

    events = _parse_sse(text)
    assert any(e["type"] == "error" for e in events)


# ---------- 文档上传 ----------

def _upload(client, name: str, data: bytes, content_type="application/octet-stream"):
    return client.post(
        "/api/v1/documents/upload",
        files={"file": (name, data, content_type)},
    )


def test_upload_rejects_unsupported_type(client, ingestion):
    resp = _upload(client, "virus.exe", b"MZ\x90\x00binary")
    assert resp.status_code == 400
    assert "不支持的文件类型" in resp.json()["detail"]


def test_upload_rejects_empty_file(client, ingestion):
    resp = _upload(client, "empty.md", b"", "text/markdown")
    assert resp.status_code == 400


def test_upload_rejects_corrupt_pdf(client, ingestion):
    """伪造扩展名（内容不是真 PDF）应返回 400，而不是 500。"""
    resp = _upload(client, "fake.pdf", b"%PDF-1.4 this is not a real pdf", "application/pdf")
    assert resp.status_code == 400


def test_upload_rejects_oversized_file(client, ingestion):
    ingestion._max_bytes = 0
    resp = _upload(client, "big.md", "内容".encode(), "text/markdown")
    assert resp.status_code == 413


def test_upload_success_then_list(client, ingestion):
    resp = _upload(client, "guide.md", "这是文档的正文内容。".encode(), "text/markdown")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ingested"
    assert body["filename"] == "guide.md"

    listed = client.get("/api/v1/documents").json()["documents"]
    assert any(d["doc_id"] == body["doc_id"] for d in listed)


def test_duplicate_upload_is_deduplicated(client, ingestion):
    """同一内容重复上传应复用已有文档，不产生重复数据。"""
    data = "重复内容测试。".encode()
    first = _upload(client, "a.md", data, "text/markdown").json()
    second = _upload(client, "b.md", data, "text/markdown").json()

    assert first["status"] == "ingested"
    assert second["status"] == "exists"
    assert second["doc_id"] == first["doc_id"]
    assert len(client.get("/api/v1/documents").json()["documents"]) == 1


# ---------- 文档删除 ----------

def test_delete_missing_document_returns_404(client, ingestion):
    resp = client.delete("/api/v1/documents/not-exist-id")
    assert resp.status_code == 404
    assert "文档不存在" in resp.json()["detail"]


def test_delete_existing_document(client, ingestion):
    doc_id = _upload(client, "x.md", "待删除的文档。".encode(), "text/markdown").json()["doc_id"]

    resp = client.delete(f"/api/v1/documents/{doc_id}")

    assert resp.status_code == 200
    assert resp.json()["deleted_chunks"] >= 1
    assert client.get("/api/v1/documents").json()["documents"] == []
