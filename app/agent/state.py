"""LangGraph 修正型 RAG 的图状态定义。"""
from typing import TypedDict


class AgentState(TypedDict, total=False):
    """在图节点之间流转的状态。

    total=False：每个节点只需返回自己更新的字段，LangGraph 会做增量合并。
    """

    question: str  # 用户原始问题（始终不变）
    query: str  # 当前用于检索的查询词（可能被 rewrite 节点改写）
    history: list[dict]  # 多轮对话历史
    doc_ids: list[str] | None  # 限定检索的文档范围；None 表示全库
    top_k: int  # 本次检索返回的片段数
    documents: list[dict]  # 检索命中的片段
    sufficient: bool  # grade 节点的评估结论
    rewrites: int  # 已执行的查询改写次数
    answer: str  # 最终答案
    citations: list[dict]  # 引用来源
