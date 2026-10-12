"""LangGraph 修正型 RAG 编排层。

与 app/core/rag.py 的线性链路并列：这里用 StateGraph 编排「检索 → 评估 → 改写 → 重检 → 生成」
的自我纠偏闭环，检索环节复用现有「混合 + 重排 + 缓存」最优链路。
"""
