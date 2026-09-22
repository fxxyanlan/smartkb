"""LLM 客户端封装：统一调用 DeepSeek（OpenAI 兼容协议）。

任何模块要用 LLM 都通过 get_llm() 获取单例，不要直接 new。
"""
from openai import AsyncOpenAI

from app.config import get_settings


class LLMClient:
    """异步 LLM 客户端，支持普通调用和流式调用。"""

    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
        )
        self._model = s.deepseek_model

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """普通对话：返回完整文本。"""
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ):
        """流式对话：逐 chunk yield。"""
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


_llm_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """全局单例。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client