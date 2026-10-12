"""LangGraph 修正型 RAG（Corrective RAG）图编排。

与 app/core/rag.py 的线性链路相比，这里多了一条"自我纠偏"闭环：

    检索 → LLM 评估上下文是否充分 →（不充分）改写问题 → 重新检索 → 生成

设计要点：
    - 检索复用 build_retriever()：即「混合(向量+BM25, RRF) → 交叉编码器重排 → 缓存」最优链路；
    - 检索为空时**不调用 LLM**，直接走 no_context（沿用项目防幻觉约定）；
    - 评估节点 fail-open：LLM 异常或输出无法解析时一律判"充分"，绝不让评估把链路拖垮；
    - 改写次数受 agent_max_rewrites 限制，最多回环 N 次后强制生成，不会死循环；
    - 流式：generate 节点经 get_stream_writer() 逐字吐 token；ainvoke 场景无 writer，
      自动退化为一次性 chat（无需分支两套代码）。
"""
import asyncio
import logging

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.config import get_settings
from app.core.llm import get_llm
from app.core.prompts import GRADE_PROMPT, REWRITE_PROMPT
from app.core.rag import (
    NO_CONTEXT_ANSWER,
    build_rag_messages,
    format_context,
    stream_llm,
    to_citations,
)
from app.core.retriever import build_retriever
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)

#: 评估节点判"充分"的输出词；模型输出以它开头即视为充分。
SUFFICIENT_TOKEN = "充分"


def _stream_writer():
    """取得 LangGraph 的流式写入器；非流式上下文（ainvoke）返回 None。"""
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 - ainvoke 场景本就没有流式上下文，属预期
        return None


def build_agent_graph(retriever=None, llm=None):
    """构建并编译修正型 RAG 图。

    Args:
        retriever: 检索器，默认 build_retriever(get_vector_store())（混合 + 重排 + 缓存）。
        llm: LLM 客户端，默认 get_llm()。两者均可注入以便测试。
    """
    settings = get_settings()
    _retriever = retriever or build_retriever(get_vector_store())
    _llm = llm or get_llm()

    async def retrieve(state: AgentState) -> dict:
        """粗召回 + 精排：走 build_retriever 组装好的最优链路（可按文档范围过滤）。"""
        k = state.get("top_k") or settings.top_k
        query = state.get("query") or state["question"]
        documents = await asyncio.to_thread(
            _retriever.search, query, k, state.get("doc_ids")
        )
        logger.info("Agent 检索 | query=%r 命中=%d", query, len(documents))
        return {"documents": documents}

    async def grade(state: AgentState) -> dict:
        """评估候选是否足以回答问题。异常/空输出一律 fail-open 判"充分"。"""
        documents = state.get("documents") or []
        if not documents:
            return {"sufficient": False}
        if not settings.agent_grade_enabled:
            return {"sufficient": True}

        prompt = GRADE_PROMPT.format(
            question=state["question"], context=format_context(documents)
        )
        try:
            verdict = await _llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
        except Exception:  # noqa: BLE001 - 评估失败不能拖垮链路
            logger.warning("上下文评估调用失败，按“充分”处理", exc_info=True)
            return {"sufficient": True}

        text = (verdict or "").strip()
        sufficient = True if not text else text.startswith(SUFFICIENT_TOKEN)
        logger.info("Agent 评估 | sufficient=%s verdict=%r", sufficient, text[:20])
        return {"sufficient": sufficient}

    async def rewrite(state: AgentState) -> dict:
        """把问题改写成更容易命中的检索查询；失败则沿用原查询，次数照计避免死循环。"""
        prompt = REWRITE_PROMPT.format(question=state["question"])
        try:
            rewritten = await _llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
        except Exception:  # noqa: BLE001 - 改写失败退化为原查询
            logger.warning("查询改写失败，沿用原查询", exc_info=True)
            rewritten = ""
        query = (rewritten or "").strip() or state.get("query") or state["question"]
        count = (state.get("rewrites") or 0) + 1
        logger.info("Agent 改写 | 第%d次 query=%r", count, query)
        return {"query": query, "rewrites": count}

    async def generate(state: AgentState) -> dict:
        """基于候选片段生成答案。流式上下文逐字写 token，否则一次性生成。"""
        documents = state.get("documents") or []
        messages = build_rag_messages(state["question"], documents, state.get("history"))
        writer = _stream_writer()

        if writer is None:
            answer = await _llm.chat(messages, temperature=0.3)
        else:
            chunks: list[str] = []
            async for delta in stream_llm(_llm, messages, settings.llm_first_token_timeout):
                chunks.append(delta)
                writer({"type": "delta", "text": delta})
            answer = "".join(chunks)

        return {"answer": answer, "citations": to_citations(documents)}

    async def no_context(state: AgentState) -> dict:
        """检索为空：直接兜底，不调用 LLM。"""
        return {"answer": NO_CONTEXT_ANSWER, "citations": []}

    def route_after_retrieve(state: AgentState) -> str:
        """候选为空 → 兜底；否则进入评估。"""
        return "grade" if state.get("documents") else "no_context"

    def route_after_grade(state: AgentState) -> str:
        """评估充分或改写次数用尽 → 生成；否则改写后重检。"""
        if state.get("sufficient"):
            return "generate"
        if (state.get("rewrites") or 0) >= settings.agent_max_rewrites:
            return "generate"
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("no_context", no_context)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"grade": "grade", "no_context": "no_context"}
    )
    graph.add_conditional_edges(
        "grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite", "retrieve")  # 回环：改写完重新检索
    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)

    return graph.compile()


_graph = None


def get_agent_graph():
    """全局单例：默认使用真实检索器与 LLM。"""
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph
