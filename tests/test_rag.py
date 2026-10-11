"""RAG 流水线与会话逻辑单元测试。"""

import asyncio

import pytest

from app.core.rag import NO_CONTEXT_ANSWER, RAGPipeline
from app.services.chat_service import ChatService
from tests.fakes import FakeLLM, FakeStore


def _rag_with(store, reply="基于文档的回答。[来源 1]"):
    llm = FakeLLM(reply=reply)
    return RAGPipeline(llm=llm, store=store), llm


async def test_answer_no_context_short_circuits():
    """检索为空：返回兜底话术，且不调用 LLM。"""
    rag, llm = _rag_with(FakeStore())

    result = await rag.answer("任意问题")

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["citations"] == []
    assert llm.calls == []


async def test_answer_returns_citations_shape():
    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    rag, _ = _rag_with(store)

    result = await rag.answer("内容")

    assert result["answer"].startswith("基于文档")
    assert len(result["citations"]) == 1
    cite = result["citations"][0]
    assert set(cite) == {"doc_id", "doc_name", "chunk_text", "score"}


async def test_answer_injects_history_into_messages():
    """多轮历史必须真的进入送给 LLM 的 messages（回归：修复前历史是死代码）。"""
    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    rag, llm = _rag_with(store)
    history = [
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]

    await rag.answer("追问", history=history)

    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert {"role": "user", "content": "上一轮问题"} in messages
    assert {"role": "assistant", "content": "上一轮回答"} in messages
    assert messages[-1] == {"role": "user", "content": "追问"}


async def test_stream_answer_yields_delta_then_done():
    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    rag, _ = _rag_with(store, reply="你好")

    events = [e async for e in rag.stream_answer("内容")]

    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "你好"
    assert events[-1]["citations"][0]["doc_name"] == "手册.md"


async def test_stream_answer_no_context_short_circuits():
    rag, llm = _rag_with(FakeStore())

    events = [e async for e in rag.stream_answer("任意问题")]

    assert events[0]["type"] == "delta"
    assert events[0]["text"] == NO_CONTEXT_ANSWER
    assert events[-1] == {"type": "done", "citations": []}
    assert llm.calls == []


async def test_stream_retries_when_failure_happens_before_any_output():
    """首字之前失败：可安全重试，最终正常吐完并给 done。"""
    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    llm = FakeLLM(reply="你好", fail_times=1)
    rag = RAGPipeline(llm=llm, store=store)

    events = [e async for e in rag.stream_answer("内容")]

    assert len(llm.calls) == 2  # 第一次失败后重试了一次
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "你好"
    assert events[-1]["type"] == "done"


async def test_stream_does_not_retry_after_partial_output():
    """已经吐出内容后失败：不重试（否则会出现重复文本），异常上抛。"""
    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    llm = FakeLLM(reply="你好世界", fail_after=2)
    rag = RAGPipeline(llm=llm, store=store)

    collected = []
    with pytest.raises(RuntimeError):
        async for event in rag.stream_answer("内容"):
            collected.append(event)

    assert len(llm.calls) == 1  # 没有重试
    assert "".join(e["text"] for e in collected) == "你好"


async def test_stream_retries_on_first_token_timeout(monkeypatch, settings):
    """首字超时：按 llm_max_retries 重试，最终仍失败则抛出。"""
    monkeypatch.setattr(settings, "llm_first_token_timeout", 0.05)
    monkeypatch.setattr(settings, "llm_max_retries", 1)

    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    llm = FakeLLM(reply="你好", stall_seconds=1.0)
    rag = RAGPipeline(llm=llm, store=store)

    with pytest.raises(asyncio.TimeoutError):
        async for _ in rag.stream_answer("内容"):
            pass

    assert len(llm.calls) == 2  # 首次 + 1 次重试


async def test_stream_no_timeout_when_disabled(monkeypatch, settings):
    """first_token_timeout <= 0 时不施加超时限制。"""
    monkeypatch.setattr(settings, "llm_first_token_timeout", 0.0)

    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    llm = FakeLLM(reply="你好", stall_seconds=0.05)
    rag = RAGPipeline(llm=llm, store=store)

    events = [e async for e in rag.stream_answer("内容")]

    assert "".join(e["text"] for e in events if e["type"] == "delta") == "你好"
    assert events[-1]["type"] == "done"


async def test_chat_service_accumulates_history_across_turns():
    """第二轮提问时，第一轮问答应作为历史传给 LLM。"""
    store = FakeStore()
    store.add("d1", "手册.md", ["相关内容片段"])
    llm = FakeLLM()
    service = ChatService(rag=RAGPipeline(llm=llm, store=store))

    first = await service.answer("第一问", None)
    second = await service.answer("第二问", first["conversation_id"])

    assert second["conversation_id"] == first["conversation_id"]
    messages = llm.calls[-1]
    assert {"role": "user", "content": "第一问"} in messages
    assert {"role": "assistant", "content": llm.reply} in messages


async def test_history_is_trimmed():
    """历史超过上限时应裁剪，避免无限增长。"""
    store = FakeStore()
    store.add("d1", "手册.md", ["片段"])
    service = ChatService(rag=RAGPipeline(llm=FakeLLM(), store=store))

    history = [{"role": "user", "content": str(i)} for i in range(100)]
    service._trim_history(history, max_turns=3)

    assert len(history) == 6


async def test_conversations_are_bounded(monkeypatch):
    """会话数超过上限时应淘汰最旧的会话，避免内存泄漏。"""
    store = FakeStore()
    store.add("d1", "手册.md", ["片段"])
    service = ChatService(rag=RAGPipeline(llm=FakeLLM(), store=store))
    service._max_conversations = 3

    for i in range(5):
        await service.answer(f"问题{i}", None)

    assert len(service._conversations) == 3
    assert "conv_" in next(iter(service._conversations))
