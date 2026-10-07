"""LLM 客户端测试：密钥缺失与上游异常的统一翻译。"""
import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

from app.config import get_settings
from app.core.errors import LLMUnavailableError
from app.core.llm import LLMClient, _wrap_upstream_error


def _request():
    return httpx.Request("POST", "https://api.deepseek.com/chat/completions")


async def test_chat_without_api_key_raises(monkeypatch):
    monkeypatch.setattr(get_settings(), "deepseek_api_key", "")
    client = LLMClient()

    with pytest.raises(LLMUnavailableError):
        await client.chat([{"role": "user", "content": "hi"}])


async def test_stream_without_api_key_raises(monkeypatch):
    monkeypatch.setattr(get_settings(), "deepseek_api_key", "")
    client = LLMClient()

    with pytest.raises(LLMUnavailableError):
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass


def test_authentication_error_maps_to_unavailable():
    err = AuthenticationError(
        "invalid key",
        response=httpx.Response(401, request=_request()),
        body=None,
    )
    assert isinstance(_wrap_upstream_error(err), LLMUnavailableError)


def test_timeout_error_maps_to_unavailable():
    err = APITimeoutError(request=_request())
    assert isinstance(_wrap_upstream_error(err), LLMUnavailableError)


def test_connection_error_maps_to_unavailable():
    err = APIConnectionError(request=_request())
    assert isinstance(_wrap_upstream_error(err), LLMUnavailableError)


def test_status_error_maps_to_unavailable():
    err = APIStatusError(
        "server error",
        response=httpx.Response(500, request=_request()),
        body=None,
    )
    assert isinstance(_wrap_upstream_error(err), LLMUnavailableError)
