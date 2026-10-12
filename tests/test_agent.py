"""LangGraph 修正型 RAG 编排的测试。

全程不加载真实 Embedding / 向量库 / LLM：检索用 ScriptedRetriever，
LLM 用 ScriptedLLM（按提示词角色区分评估 / 改写 / 生成三类调用）。
"""
import json

from app.agent.graph import build_agent_graph
from app.agent.service import AgentService
from app.api import agent as agent_api
from app.core.rag import NO_CONTEXT_ANSWER
from app.main import app
from tests.fakes import FakeStore, ScriptedLLM, ScriptedRetriever, make_doc


def _graph(grade=None, rewrite=None, reply="答案", results=None):
    """组装 (graph, llm, retriever) 三元组。"""
    llm = ScriptedLLM(grade_replies=grade, rewrite_replies=rewrite, reply=reply)
    retriever = ScriptedRetriever(results if results is not None else [[make_doc()]])
    return build_agent_graph(retriever=retriever, llm=llm), llm, retriever


def _state(question="原始问题", **extra):
    return {"question": question, "query": question, "top_k": 4, "rewrites": 0, **extra}


# ---------- 图编排 ----------


async def test_sufficient_first_pass_generates_without_rewrite():
    graph, llm, retriever = _graph(grade=["充分"], reply="答案A")

    final = await graph.ainvoke(_state())

    assert final["answer"] == "答案A"
    assert final["rewrites"] == 0
    assert final["citations"][0]["doc_name"] == "doc.md"
    assert llm.grade_calls == 1
    assert llm.rewrite_calls == 0
    assert llm.generate_calls == 1
    assert retriever.queries == ["原始问题"]  # 未改写，只检索一次


async def test_insufficient_triggers_rewrite_and_second_retrieval():
    graph, llm, retriever = _graph(
        grade=["不充分", "充分"],
        rewrite=["改写后的问题"],
        reply="答案B",
        results=[[make_doc(text="低相关片段")], [make_doc(text="高相关片段")]],
    )

    final = await graph.ainvoke(_state())

    assert final["rewrites"] == 1
    assert llm.rewrite_calls == 1
    assert llm.grade_calls == 2  # 改写后重新评估
    assert llm.generate_calls == 1
    # 回环生效：第二次用改写后的查询重新检索
    assert retriever.queries == ["原始问题", "改写后的问题"]
    # 生成用的是第二次（改写后）检索到的片段
    assert final["citations"][0]["chunk_text"] == "高相关片段"
    assert final["answer"] == "答案B"


async def test_empty_retrieval_short_circuits_without_llm():
    graph, llm, retriever = _graph(results=[[]])

    final = await graph.ainvoke(_state())

    assert final["answer"] == NO_CONTEXT_ANSWER
    assert final["citations"] == []
    assert llm.calls == []  # 防幻觉：检索为空时一次 LLM 都不调用


async def test_rewrite_is_bounded_and_never_loops():
    """评估持续判"不充分"时，改写次数达上限即强制生成。"""
    graph, llm, retriever = _graph(grade=["不充分"], rewrite=["改写后的问题"], reply="答案C")

    final = await graph.ainvoke(_state())

    assert final["rewrites"] == 1  # agent_max_rewrites 默认 1
    assert llm.rewrite_calls == 1
    assert llm.generate_calls == 1
    assert len(retriever.queries) == 2
    assert final["answer"] == "答案C"


async def test_grade_disabled_skips_evaluation(monkeypatch, settings):
    monkeypatch.setattr(settings, "agent_grade_enabled", False)
    graph, llm, retriever = _graph(reply="答案D")

    final = await graph.ainvoke(_state())

    assert final["answer"] == "答案D"
    assert llm.grade_calls == 0  # 评估节点被关闭
    assert llm.rewrite_calls == 0
    assert llm.generate_calls == 1
    assert len(retriever.queries) == 1


async def test_grade_failure_fails_open(monkeypatch, settings):
    """评估调用抛错时应 fail-open 判"充分"，直接生成而不是报错。"""
    graph, llm, retriever = _graph(reply="答案E")
    monkeypatch.setattr(llm, "chat", _raise_on_grade(llm))

    final = await graph.ainvoke(_state())

    assert final["sufficient"] is True
    assert final["answer"] == "答案E"
    assert llm.rewrite_calls == 0


def _raise_on_grade(llm):
    """包装 chat：评估调用抛异常，其余照常。"""
    original = llm.chat

    async def wrapper(messages, temperature=0.3, max_tokens=None):
        if ScriptedLLM._role(messages) == "grade":
            raise RuntimeError("grade boom")
        return await original(messages, temperature=temperature, max_tokens=max_tokens)

    return wrapper


async def test_rewrite_zero_disables_loop(monkeypatch, settings):
    monkeypatch.setattr(settings, "agent_max_rewrites", 0)
    graph, llm, retriever = _graph(grade=["不充分"], reply="答案F")

    final = await graph.ainvoke(_state())

    assert final["rewrites"] == 0
    assert llm.rewrite_calls == 0
    assert llm.generate_calls == 1


# ---------- AgentService ----------


async def test_service_answer_returns_rewrites_and_conversation_id():
    graph, _llm, _retriever = _graph(grade=["充分"], reply="答案G")
    service = AgentService(graph=graph)

    result = await service.answer("问题", None)

    assert result["answer"] == "答案G"
    assert result["conversation_id"].startswith("conv_")
    assert result["rewrites"] == 0
    assert result["citations"]


async def test_service_passes_history_across_turns():
    graph, llm, _retriever = _graph(grade=["充分"], reply="答案H")
    service = AgentService(graph=graph)

    first = await service.answer("第一问", None)
    await service.answer("第二问", first["conversation_id"])

    # 第二次生成时，messages 里应带上第一轮的历史
    contents = [m["content"] for m in llm.calls[-1]]
    assert "第一问" in contents
    assert "答案H" in contents
    assert service.conversations[first["conversation_id"]]  # 会话被记录


async def test_service_stream_yields_deltas_then_done():
    graph, _llm, _retriever = _graph(grade=["充分"], reply="流式答案")
    service = AgentService(graph=graph)

    events = [e async for e in service.stream_answer("问题", None)]

    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas == "流式答案"
    done = events[-1]
    assert done["type"] == "done"
    assert done["conversation_id"].startswith("conv_")
    assert len(done["citations"]) == 1


async def test_service_stream_no_context_emits_single_delta():
    """检索为空时没有流式增量，服务应把整段兜底话术作为单个 delta 发出。"""
    graph, llm, _retriever = _graph(results=[[]])
    service = AgentService(graph=graph)

    events = [e async for e in service.stream_answer("问题", None)]

    deltas = [e for e in events if e["type"] == "delta"]
    assert len(deltas) == 1
    assert deltas[0]["text"] == NO_CONTEXT_ANSWER
    assert events[-1]["type"] == "done"
    assert events[-1]["citations"] == []
    assert llm.calls == []


# ---------- API ----------


def _override(service):
    app.dependency_overrides[agent_api.get_agent_service] = lambda: service


def test_agent_chat_endpoint(client):
    graph, _llm, _retriever = _graph(grade=["充分"], reply="接口答案")
    _override(AgentService(graph=graph))

    resp = client.post("/api/v1/agent/chat", json={"question": "问题"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "接口答案"
    assert body["rewrites"] == 0
    assert body["conversation_id"].startswith("conv_")
    assert body["citations"][0]["doc_name"] == "doc.md"


def test_agent_chat_stream_endpoint(client):
    graph, _llm, _retriever = _graph(grade=["充分"], reply="流式接口答案")
    _override(AgentService(graph=graph))

    resp = client.post("/api/v1/agent/chat/stream", json={"question": "问题"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    events = [
        json.loads(line[len("data: "):])
        for line in resp.text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "流式接口答案"
    assert events[-1]["type"] == "done"
    assert events[-1]["conversation_id"].startswith("conv_")


def test_agent_chat_rejects_empty_question(client):
    graph, _llm, _retriever = _graph()
    _override(AgentService(graph=graph))

    resp = client.post("/api/v1/agent/chat", json={"question": "   "})

    assert resp.status_code == 422


# ---------- 历史污染防护（与 /chat 链路同样适用）----------


async def test_agent_retrieval_is_limited_to_doc_ids():
    """图编排的检索必须把文档范围透传到检索器，旧文档才不会被召回。"""
    graph, _llm, retriever = _graph(grade=["充分"], reply="答案X")

    await graph.ainvoke(_state(doc_ids=["d1"]))

    assert retriever.scopes == [["d1"]]


async def test_agent_service_resets_history_when_document_changes():
    """回归：会话中途上传新文档（版本号变化）后，旧问答不得再进入生成上下文。"""
    store = FakeStore()
    store.add("old", "旧文档.md", ["旧文档的核心内容"])
    graph, llm, _retriever = _graph(grade=["充分"], reply="答案Y")
    service = AgentService(graph=graph, store=store)

    first = await service.answer("旧问题", None)
    store.add("new", "新文档.md", ["新文档的核心内容"])  # 上传新文档 → 版本号自增

    await service.answer("新问题", first["conversation_id"])

    contents = [m["content"] for m in llm.calls[-1]]
    assert "旧问题" not in contents
