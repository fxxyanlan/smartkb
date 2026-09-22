"""API 端点测试。"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    """/health 应返回 200 + status ok。"""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_requires_non_empty_question():
    """空问题应被 Pydantic 校验拦截。"""
    resp = client.post("/api/v1/chat", json={"question": ""})
    assert resp.status_code == 422


def test_chat_request_schema_validation():
    """缺字段应被拦截。"""
    resp = client.post("/api/v1/chat", json={})
    assert resp.status_code == 422