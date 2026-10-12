"""四方案多维度检索对比基准（统一埋点）。

在同一次进程内、同一份语料与同一份问题集上依次跑通四种检索方案：

    1. BM25                        —— rank_bm25.BM25Okapi + jieba 分词
    2. 纯向量（Embedding）           —— all-MiniLM-L6-v2 句向量 + 余弦相似度
    3. BM25 + 向量（混合）           —— 两路 min-max 归一化后加权融合，alpha=0.5
    4. BM25 + 向量 + Rerank         —— 混合粗召回 Top10 → bge-reranker-base 交叉编码器精排

并测量五类指标：

    - 召回率：Recall@1 / Recall@3 / Recall@5
    - 准确率：Precision@1 / Precision@3 / Precision@5（单金标场景 = Recall@k / k）
    - top-k 匹配精度：MRR@10
    - 响应耗时：单次查询延迟（不含索引构建与模型加载）的中位数 / P95
    - 内存占用：检索索引常驻内存 + 该方案所需模型权重常驻内存

设计约束（与既有三组基线可比）：
    - 语料读取顺序、分词方式、相似度口径、融合权重 alpha 与 eval/*_eval.py 完全一致，
      因此同一问题集上得到的 Recall@1 / MRR 应与 baseline/*_baseline.json 对得上；
    - 本脚本不写入任何既有 baseline 文件，只输出独立的 multi_metric_<tag>.json。

用法：
    python eval/compare_metrics.py
    python eval/compare_metrics.py --corpus eval/corpus_v2 --dataset eval/dataset_v2.jsonl --tag v2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import tracemalloc

# ---- 离线优先：两个模型均已本地缓存，规避镜像 WinError 10060 超时 ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import jieba  # noqa: E402
import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "baseline")
RERANK_MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "model", "bge-reranker-base"))

# eval/ 现有脚本的历史取值（英文模型），可用 --embed-model 覆盖
EMBED_MODEL = "all-MiniLM-L6-v2"
# 生产链路 app/config.py 的取值（中文模型）
ZH_EMBED_MODEL = "./data/models/models/BAAI--bge-small-zh-v1.5/snapshots/master"
ALPHA = 0.5
TOP_K_RETRIEVE = 10
KS = (1, 3, 5)
MRR_K = 10
WARMUP_ROUNDS = 1
MEASURE_ROUNDS = 2


def load_corpus(corpus_dir: str):
    """读取语料目录下的 md 文档，按文件名排序构建【文件名 → 下标】映射。"""
    corpus, doc_names, name2idx = [], [], {}
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".md"))
    for idx, fname in enumerate(files):
        with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
            corpus.append(f.read().strip())
        doc_names.append(fname)
        name2idx[fname] = idx
    return corpus, doc_names, name2idx


def load_dataset(dataset_path: str):
    items = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def gold_ids_of(item: dict, name2idx: dict) -> list[int]:
    """兼容单金标 `doc` 与多金标 `docs` 两种写法，去重并保持顺序。"""
    names = []
    if item.get("doc"):
        names.append(item["doc"])
    for name in item.get("docs") or []:
        names.append(name)
    ids = []
    for name in names:
        did = name2idx.get(name)
        if did is not None and did not in ids:
            ids.append(did)
    return ids


def minmax(scores: np.ndarray) -> np.ndarray:
    """min-max 归一化；全相等时返回全 0，避免除零得到 nan。"""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo == 0:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def model_param_bytes(model) -> int:
    """统计模型权重常驻字节数（仅参数张量，不含激活与框架开销）。"""
    params = None
    for attr in ("parameters",):
        fn = getattr(model, attr, None)
        if callable(fn):
            try:
                params = list(fn())
                break
            except Exception:
                params = None
    if not params:
        # CrossEncoder: 真正的模型在 .model 上
        inner = getattr(model, "model", None)
        if inner is not None and callable(getattr(inner, "parameters", None)):
            params = list(inner.parameters())
    if not params:
        return 0
    return int(sum(int(p.numel()) * int(p.element_size()) for p in params))


def score_metrics(ranked: list[int], gold_ids: list[int]) -> dict:
    """按单/多金标计算 Recall@k、Precision@k、MRR。

    - `mrr@10`：仅在前 10 位内找首个命中（top-k 匹配精度口径）
    - `mrr`   ：在全量候选内找首个命中，与 baseline/*_baseline.json 的 avg_mrr 口径一致，
                用于交叉校验本次实测与冻结基线是否可复现
    """
    out = {}
    gold_set = set(gold_ids)
    for k in KS:
        hits = len(set(ranked[:k]) & gold_set)
        out[f"recall@{k}"] = 1.0 if hits > 0 else 0.0
        out[f"precision@{k}"] = hits / k
    mrr_capped = 0.0
    for rank, did in enumerate(ranked[:MRR_K]):
        if did in gold_set:
            mrr_capped = 1.0 / (rank + 1)
            break
    out[f"mrr@{MRR_K}"] = mrr_capped
    mrr_full = 0.0
    for rank, did in enumerate(ranked):
        if did in gold_set:
            mrr_full = 1.0 / (rank + 1)
            break
    out["mrr"] = mrr_full
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return ordered[idx]


def build_rankers(corpus, name2idx, embed_model: str):
    """构建四种检索方案的打分函数，并测量索引构建的内存峰值。"""
    # ---------- BM25 索引 ----------
    tracemalloc.start()
    tokenized = [jieba.lcut(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    _, bm25_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ---------- 向量索引 ----------
    embedder = SentenceTransformer(embed_model)
    doc_embeds = embedder.encode(corpus, convert_to_numpy=True)
    doc_norms = np.linalg.norm(doc_embeds, axis=1)
    embed_model_bytes = model_param_bytes(embedder)
    vector_index_bytes = int(doc_embeds.nbytes + doc_norms.nbytes)
    print(f"✅ 向量模型：{embed_model}（{doc_embeds.shape[1]} 维）")

    # ---------- Rerank 模型 ----------
    if not os.path.isdir(RERANK_MODEL_DIR):
        raise FileNotFoundError(
            f"未找到本地重排模型：{RERANK_MODEL_DIR}\n"
            f"请先运行 eval/download_reranker.py，或改用 RERANK_MODEL_DIR。"
        )
    reranker = CrossEncoder(RERANK_MODEL_DIR, local_files_only=True)
    rerank_model_bytes = model_param_bytes(reranker)

    def bm25_rank(q: str) -> list[int]:
        scores = bm25.get_scores(jieba.lcut(q))
        return np.argsort(-np.asarray(scores)).tolist()

    def vector_rank(q: str) -> list[int]:
        q_emb = embedder.encode(q, convert_to_numpy=True)
        sims = doc_embeds @ q_emb / (doc_norms * np.linalg.norm(q_emb))
        return np.argsort(-sims).tolist()

    def hybrid_rank(q: str) -> list[int]:
        q_emb = embedder.encode(q, convert_to_numpy=True)
        sims = doc_embeds @ q_emb / (doc_norms * np.linalg.norm(q_emb))
        bm25_scores = minmax(np.asarray(bm25.get_scores(jieba.lcut(q))))
        fused = ALPHA * bm25_scores + (1 - ALPHA) * minmax(sims)
        return np.argsort(-fused).tolist()

    def rerank_rank(q: str) -> list[int]:
        coarse = hybrid_rank(q)
        candidates = coarse[:TOP_K_RETRIEVE]
        pairs = [[q, corpus[did]] for did in candidates]
        scores = np.asarray(reranker.predict(pairs))
        order = np.argsort(-scores).tolist()
        return [candidates[i] for i in order] + coarse[TOP_K_RETRIEVE:]

    schemes = [
        {
            "name": "BM25",
            "rank": bm25_rank,
            "index_bytes": int(bm25_peak),
            "model_bytes": 0,
        },
        {
            "name": "Pure Embedding",
            "rank": vector_rank,
            "index_bytes": vector_index_bytes,
            "model_bytes": embed_model_bytes,
        },
        {
            "name": "Hybrid(BM25+Embedding)",
            "rank": hybrid_rank,
            "index_bytes": int(bm25_peak) + vector_index_bytes,
            "model_bytes": embed_model_bytes,
        },
        {
            "name": "Hybrid(BM25+Embedding)+Rerank",
            "rank": rerank_rank,
            "index_bytes": int(bm25_peak) + vector_index_bytes,
            "model_bytes": embed_model_bytes + rerank_model_bytes,
        },
    ]
    return schemes


def aggregate(score_rows: dict[int, dict]) -> dict:
    """把逐题得分聚合成方案级指标。"""
    agg = {}
    for k in KS:
        agg[f"recall@{k}"] = mean([row[f"recall@{k}"] for row in score_rows.values()])
        agg[f"precision@{k}"] = mean([row[f"precision@{k}"] for row in score_rows.values()])
    agg[f"mrr@{MRR_K}"] = mean([row[f"mrr@{MRR_K}"] for row in score_rows.values()])
    agg["mrr"] = mean([row["mrr"] for row in score_rows.values()])
    return agg


def breakdown(answerable, score_rows, dim_key: str) -> dict:
    """按某一维度（difficulty / semantic / qtype）切分后重新聚合，用于定位各方案的差异来源。"""
    groups: dict[str, list] = {}
    for it in answerable:
        groups.setdefault(it.get(dim_key) or "未标注", []).append(it)
    out = {}
    for value, group in sorted(groups.items()):
        sub = {it["id"]: score_rows[it["id"]] for it in group}
        entry = {"count": len(group)}
        entry.update(aggregate(sub))
        out[value] = entry
    return out


def benchmark(schemes, items, name2idx, doc_names):
    answerable = [it for it in items if gold_ids_of(it, name2idx)]
    print(f"✅ 有效可回答样本：{len(answerable)} / {len(items)}")

    results = []
    for scheme in schemes:
        rank = scheme["rank"]

        # 预热（不计入耗时，同时触发模型图构建/缓存）
        for _ in range(WARMUP_ROUNDS):
            for it in answerable:
                rank(it["question"])

        # 计时 + 记录排名
        latencies, rankings = [], {}
        for _ in range(MEASURE_ROUNDS):
            for it in answerable:
                t0 = time.perf_counter()
                ranked = rank(it["question"])
                latencies.append((time.perf_counter() - t0) * 1000.0)
                rankings[it["id"]] = ranked

        # 逐题得分
        score_rows = {
            it["id"]: score_metrics(rankings[it["id"]], gold_ids_of(it, name2idx))
            for it in answerable
        }

        agg = aggregate(score_rows)
        agg["latency_median_ms"] = round(statistics.median(latencies), 2)
        agg["latency_p95_ms"] = round(p95(latencies), 2)
        agg["latency_mean_ms"] = round(mean(latencies), 2)
        agg["index_bytes"] = scheme["index_bytes"]
        agg["model_bytes"] = scheme["model_bytes"]
        agg["total_bytes"] = scheme["index_bytes"] + scheme["model_bytes"]

        # 维度切分（v1 无维度标注时统一落在「未标注」，等价于总体指标）
        by_dimension = {
            dim: breakdown(answerable, score_rows, dim)
            for dim in ("difficulty", "semantic", "qtype")
        }

        # 明细（供追溯：每题的金标与 Top5 命中）
        details = []
        for it in answerable:
            gids = gold_ids_of(it, name2idx)
            ranked = rankings[it["id"]]
            row = dict(score_rows[it["id"]])
            row.update({
                "qid": it["id"],
                "question": it["question"],
                "gold_doc_ids": gids,
                "gold_doc_names": [doc_names[i] for i in gids],
                "retrieved_top5": ranked[:5],
                "retrieved_top5_names": [doc_names[i] for i in ranked[:5]],
            })
            details.append(row)

        results.append({
            "method": scheme["name"],
            "metrics": agg,
            "by_dimension": by_dimension,
            "details": details,
        })

        print(f"\n===== {scheme['name']} =====")
        r = agg["recall@1"], agg["recall@3"], agg["recall@5"]
        p = agg["precision@1"], agg["precision@3"], agg["precision@5"]
        print(f"  Recall@1/3/5   = {r[0]:.4f} / {r[1]:.4f} / {r[2]:.4f}")
        print(f"  Precision@1/3/5= {p[0]:.4f} / {p[1]:.4f} / {p[2]:.4f}")
        print(f"  MRR@10         = {agg['mrr@10']:.4f}")
        print(f"  延迟 中位/P95   = {agg['latency_median_ms']} ms / {agg['latency_p95_ms']} ms")
        print(f"  索引/模型内存   = {agg['index_bytes'] / 1024:.1f} KB / "
              f"{agg['model_bytes'] / 1048576:.1f} MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="四方案多维度检索对比基准")
    parser.add_argument("--corpus", default=os.path.join(BASE_DIR, "corpus"))
    parser.add_argument("--dataset", default=os.path.join(BASE_DIR, "dataset.jsonl"))
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                        help=f"向量模型，默认 {EMBED_MODEL}；生产链路为 {ZH_EMBED_MODEL}")
    args = parser.parse_args()

    os.makedirs(BASELINE_DIR, exist_ok=True)

    corpus, doc_names, name2idx = load_corpus(args.corpus)
    print(f"✅ 加载文档总数：{len(corpus)}（{os.path.relpath(args.corpus, BASE_DIR)}）")
    items = load_dataset(args.dataset)
    print(f"✅ 读取评测问题数量：{len(items)}（{os.path.relpath(args.dataset, BASE_DIR)}）")

    schemes = build_rankers(corpus, name2idx, args.embed_model)
    results = benchmark(schemes, items, name2idx, doc_names)

    payload = {
        "tag": args.tag,
        "corpus_dir": os.path.relpath(args.corpus, BASE_DIR),
        "dataset": os.path.relpath(args.dataset, BASE_DIR),
        "doc_count": len(corpus),
        "doc_names": doc_names,
        "query_count": len(items),
        "config": {
            "embed_model": args.embed_model,
            "alpha": ALPHA,
            "top_k_retrieve": TOP_K_RETRIEVE,
            "ks": list(KS),
            "mrr_k": MRR_K,
            "warmup_rounds": WARMUP_ROUNDS,
            "measure_rounds": MEASURE_ROUNDS,
            "rerank_model": os.path.basename(RERANK_MODEL_DIR),
        },
        "results": results,
    }
    out_path = os.path.join(BASELINE_DIR, f"multi_metric_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 多维度对比结果已保存：{os.path.relpath(out_path, BASE_DIR)}")


if __name__ == "__main__":
    main()
