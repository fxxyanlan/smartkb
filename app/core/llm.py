"""LLM 客户端封装：统一调用 DeepSeek（OpenAI 兼容协议）。

任何模块要用 LLM 都通过 get_llm() 获取单例，不要直接 new。
上游错误（鉴权失败/超时/连接失败/5xx）统一转成 LLMUnavailableError（→ HTTP 503），
避免把第三方错误原样漏成 500。
"""
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
)

from app.config import get_settings
from app.core.errors import LLMUnavailableError


def _wrap_upstream_error(exc: Exception) -> LLMUnavailableError:
    """把 OpenAI SDK 的异常翻译成领域异常。"""
    if isinstance(exc, AuthenticationError):
        return LLMUnavailableError("大模型鉴权失败，请检查 DEEPSEEK_API_KEY 是否有效")
    if isinstance(exc, APITimeoutError):
        return LLMUnavailableError("调用大模型超时")
    if isinstance(exc, APIConnectionError):
        return LLMUnavailableError("无法连接大模型服务")
    if isinstance(exc, APIStatusError):
        return LLMUnavailableError(f"大模型服务返回错误（HTTP {exc.status_code}）")
    return LLMUnavailableError()


class LLMClient:
    """异步 LLM 客户端，支持普通调用和流式调用。"""

    def __init__(self) -> None:
        s = get_settings()
        self._model = s.deepseek_model
        self._configured = s.llm_configured
        self._client = AsyncOpenAI(
            api_key=s.deepseek_api_key or "not-configured",
            base_url=s.deepseek_base_url,
            timeout=s.llm_timeout,
            max_retries=s.llm_max_retries,
        )

    def _ensure_configured(self) -> None:
        if not self._configured:
            raise LLMUnavailableError("未配置 DEEPSEEK_API_KEY，无法调用大模型")

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """普通对话：返回完整文本。"""
        self._ensure_configured()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001 - 统一翻译上游异常
            raise _wrap_upstream_error(e) from e
        return resp.choices[0].message.content or ""

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ):
        """流式对话：逐 chunk yield。"""
        self._ensure_configured()
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
        except Exception as e:  # noqa: BLE001 - 统一翻译上游异常
            raise _wrap_upstream_error(e) from e

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:  # noqa: BLE001 - 流中途出错
            raise _wrap_upstream_error(e) from e


_llm_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """全局单例。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
