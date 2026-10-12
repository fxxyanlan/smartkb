"""BM25 + 向量 + Rerank 三路检索评测脚本。

设计目标：与 `eval/` 下三组既有基线（bm25_demo.py / embedding_eval.py / hybrid_eval.py）
保持一致 —— 同一份语料（corpus/*.md）、同一份数据集（dataset.jsonl）、同一套指标口径
（K=1 的 Recall@1 与 MRR）与同一种输出格式，**唯一区别**是在混合检索结果之上再接入
交叉编码器（bge-reranker-base）做精排。因此四组结果可直接横向对比。

关键修复（详见 docs/troubleshooting-hybrid-rerank.md）：
    1. 原稿依赖 `../test_questions.json`（不存在）与 `../docs` 下的 .txt（实际是 .md），
       改为与其他脚本一致地从 `eval/corpus/*.md` 与 `eval/dataset.jsonl` 读取。
    2. 原稿向量通道是 `np.zeros_like` 桩，等价于纯 BM25；这里接入真实向量检索，
       模型与融合权重（alpha=0.5）与 hybrid_eval.py 完全一致，保证精排前的候选
       排序可复现 hybrid 基线。
    3. 原稿输出 `../baseline/hybrid_rerank.json`，与既有命名规范不符；改为写入
       `eval/baseline/hybrid_rerank_baseline.json` 与 `hybrid_rerank_report.json`。
    4. 默认离线加载：两个模型均已本地缓存，规避镜像超时（WinError 10060）。
"""
import json
import os

# ---- 离线优先：all-MiniLM-L6-v2 与 bge-reranker-base 均已本地缓存 ----
# 不设这两项时，sentence-transformers 会先联网校验，在受限网络下会等待到
# WinError 10060 超时。用 setdefault 保留外部显式覆盖的能力。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "corpus")
BASELINE_DIR = os.path.join(BASE_DIR, "baseline")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.jsonl")
RERANK_MODEL_DIR = os.path.normpath(
    os.path.join(BASE_DIR, "..", "model", "bge-reranker-base")
)

EMBED_MODEL = "all-MiniLM-L6-v2"  # 与 embedding_eval / hybrid_eval 一致，保证可比性
ALPHA = 0.5                       # 与 hybrid_eval 一致
TOP_K_RETRIEVE = 10               # 粗召回候选数，交给 rerank 精排
K = 1                             # 与三组基线一致：Recall@1 / MRR


def load_corpus():
    """读取 corpus 下的 md 文档，构建【文件名 → 下标】映射（与其他脚本同序）。"""
    corpus, doc_names, name2idx = [], [], {}
    files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith(".md"))
    for idx, fname in enumerate(files):
        with open(os.path.join(CORPUS_DIR, fname), "r", encoding="utf-8") as f:
            corpus.append(f.read().strip())
        doc_names.append(fname)
        name2idx[fname] = idx
    return corpus, doc_names, name2idx


def load_dataset():
    query_list = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                query_list.append(json.loads(line))
    return query_list


def minmax(scores: np.ndarray) -> np.ndarray:
    """min-max 归一化；全相等时返回全 0，避免除零得到 nan。"""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo == 0:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def recall_at_k(retrieved_ids, gold_ids, k):
    return 1.0 if len(set(retrieved_ids[:k]) & set(gold_ids)) > 0 else 0.0


def mrr_score(retrieved_ids, gold_ids):
    for rank, did in enumerate(retrieved_ids):
        if did in gold_ids:
            return 1.0 / (rank + 1)
    return 0.0


def main():
    os.makedirs(BASELINE_DIR, exist_ok=True)

    # ---------- 1. 语料 + 三路检索器初始化 ----------
    corpus, doc_names, name2idx = load_corpus()
    print(f"✅ 加载文档总数：{len(corpus)}")

    bm25 = BM25Okapi([jieba.lcut(doc) for doc in corpus])
    embedder = SentenceTransformer(EMBED_MODEL)
    doc_embeds = embedder.encode(corpus, convert_to_numpy=True)
    doc_norms = np.linalg.norm(doc_embeds, axis=1)

    if not os.path.isdir(RERANK_MODEL_DIR):
        raise FileNotFoundError(
            f"未找到本地重排模型：{RERANK_MODEL_DIR}\n"
            f"请先运行 eval/download_reranker.py 下载，或改用 RERANK_MODEL_DIR。"
        )
    reranker = CrossEncoder(RERANK_MODEL_DIR, local_files_only=True)
    print(f"✅ 重排模型已加载：{RERANK_MODEL_DIR}")

    # ---------- 2. 数据集 ----------
    query_list = load_dataset()
    print(f"✅ 读取评测问题数量：{len(query_list)}")

    # ---------- 3. 逐题：混合检索 → 精排 ----------
    recall_total = mrr_total = 0.0
    valid_count = 0
    detail_items = []

    for qid, item in enumerate(query_list, start=1):
        q = item["question"]
        gold_doc_name = item.get("doc")

        # 3.1 混合检索（与 hybrid_eval.py 完全一致）
        q_emb = embedder.encode(q, convert_to_numpy=True)
        sims = doc_embeds @ q_emb / (doc_norms * np.linalg.norm(q_emb))
        final_score = ALPHA * minmax(np.array(bm25.get_scores(jieba.lcut(q)))) \
            + (1 - ALPHA) * minmax(sims)
        ranked_idx = np.argsort(-final_score).tolist()

        # 3.2 粗召回 top-10 → 交叉编码器精排
        candidates = ranked_idx[:TOP_K_RETRIEVE]
        pairs = [[q, corpus[did]] for did in candidates]
        rerank_scores = reranker.predict(pairs)
        order = np.argsort(-np.asarray(rerank_scores)).tolist()
        reranked_ids = [candidates[i] for i in order]
        retrieved_ids = reranked_ids[:K]

        # 3.3 指标（仅统计 answerable 样本）
        if gold_doc_name is None:
            rk = mrr = None
            gold_ids = []
        else:
            gold_ids = [name2idx[gold_doc_name]]
            rk = recall_at_k(retrieved_ids, gold_ids, K)
            mrr = mrr_score(retrieved_ids, gold_ids)
            recall_total += rk
            mrr_total += mrr
            valid_count += 1

        detail_items.append({
            "qid": qid,
            "question": q,
            "gold_doc_name": gold_doc_name,
            "gold_doc_ids": gold_ids,
            "retrieved_doc_ids": retrieved_ids,
            "recall@1": rk,
            "mrr": mrr,
        })

    avg_recall = recall_total / valid_count if valid_count else 0
    avg_mrr = mrr_total / valid_count if valid_count else 0

    print("\n===== 混合检索 + Rerank 评测汇总 =====")
    print(f"有效样本数 = {valid_count}")
    print(f"Recall@{K} = {avg_recall:.4f}")
    print(f"MRR        = {avg_mrr:.4f}")

    # ---------- 4. 输出（格式与三组基线完全一致）----------
    summary = {
        "method": "Hybrid(BM25+Embedding)+Rerank",
        "alpha": ALPHA,
        "rerank_model": os.path.basename(RERANK_MODEL_DIR),
        "top_k_retrieve": TOP_K_RETRIEVE,
        "K": K,
        "valid_answerable_count": valid_count,
        "avg_recall@1": avg_recall,
        "avg_mrr": avg_mrr,
    }
    baseline_path = os.path.join(BASELINE_DIR, "hybrid_rerank_baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(BASELINE_DIR, "hybrid_rerank_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": detail_items},
                  f, ensure_ascii=False, indent=2)

    print(f"\n✅ 基线指标保存至 {os.path.relpath(baseline_path, BASE_DIR)}")
    print(f"✅ 详细查询记录保存至 {os.path.relpath(report_path, BASE_DIR)}")


if __name__ == "__main__":
    main()
