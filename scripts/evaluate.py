"""检索评估：同一语料下对比【纯向量】与【混合检索(向量+BM25/RRF)】的召回表现。

用法：
    python scripts/evaluate.py                  # 出对比表 + 写 eval/report.json
    python scripts/evaluate.py --rerank         # 额外测"混合检索 + 交叉编码器精排"
    python scripts/evaluate.py --check-baseline # 与 eval/baseline.json 比对，回退则以非 0 退出

输出：
    - 控制台对比表
    - eval/report.json（机器可读，供报告引用）

说明：本脚本只评估"检索"环节（不调用 LLM），因此可离线、可复现、零 API 成本。
"""
import asyncio
import io
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在导入 app 之前指定隔离的数据目录，避免污染开发数据
_TMP = tempfile.mkdtemp(prefix="smartkb_eval_")
os.environ["CHROMA_PERSIST_DIR"] = str(Path(_TMP) / "chroma")
os.environ["UPLOAD_DIR"] = str(Path(_TMP) / "documents")

from fastapi import UploadFile  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.core.cache import RetrievalCache  # noqa: E402
from app.core.reranker import CrossEncoderReranker  # noqa: E402
from app.core.retriever import (  # noqa: E402
    CachedRetriever,
    HybridRetriever,
    RerankedRetriever,
)
from app.core.vector_store import get_vector_store  # noqa: E402
from app.services.ingestion import IngestionService  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "corpus"
DATASET = ROOT / "eval" / "dataset.jsonl"
REPORT = ROOT / "eval" / "report.json"
BASELINE = ROOT / "eval" / "baseline.json"
TOP_K = 4
WARMUP = "预热"  # 首个查询含模型加载 / 索引构建开销，不计入延迟统计


def norm(text: str) -> str:
    return text.lower().replace(" ", "").replace("\u3000", "")


def load_dataset() -> list[dict]:
    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def ingest_corpus() -> list[dict]:
    service = IngestionService()
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        data = path.read_bytes()
        upload = UploadFile(file=io.BytesIO(data), filename=path.name)
        result = await service.ingest(upload)
        docs.append(result)
    return docs


def evaluate(retriever, rows: list[dict], with_latency: bool = False) -> dict:
    """对 answerable 题算 recall@k；对 oos 题记录最高相似度。"""
    hits = {1: 0, TOP_K: 0}
    total = 0
    latencies: list[float] = []
    misses: list[dict] = []
    oos_top_scores: list[float] = []

    for row in rows:
        t0 = time.perf_counter()
        results = retriever.search(row["question"], TOP_K)
        if with_latency:
            latencies.append((time.perf_counter() - t0) * 1000)

        if row["type"] == "answerable":
            total += 1
            top_text = norm("".join(r["chunk_text"] for r in results))
            top1_text = norm(results[0]["chunk_text"]) if results else ""
            hit_k = all(norm(k) in top_text for k in row["must_contain"])
            hit_1 = all(norm(k) in top1_text for k in row["must_contain"])
            hits[TOP_K] += int(hit_k)
            hits[1] += int(hit_1)
            if not hit_k:
                misses.append({"id": row["id"], "question": row["question"]})
        elif row["type"] == "oos" and results:
            s = results[0].get("score")
            if isinstance(s, (int, float)):
                oos_top_scores.append(round(float(s), 3))

    out = {
        "answerable_total": total,
        "recall@1": round(hits[1] / total, 4) if total else 0.0,
        "recall@4": round(hits[TOP_K] / total, 4) if total else 0.0,
        "hits@1": hits[1],
        "hits@4": hits[TOP_K],
        "misses": misses,
    }
    if with_latency and latencies:
        out["latency_ms_median"] = round(statistics.median(latencies), 1)
    if oos_top_scores:
        out["oos_top1_sim_avg"] = round(statistics.mean(oos_top_scores), 3)
    return out


def rerank_variant(hybrid: HybridRetriever, rows: list[dict]) -> dict:
    """在混合检索之上加一层交叉编码器精排，量化重排带来的首位提升。

    重排默认不参与评估：它会加载外部模型（约 1GB），在网络受限时会先卡一次超时。
    需要时用 `--rerank` 或 RERANK_ENABLED=true 显式开启。
    模型不可用时如实标注 unavailable——绝不用"未重排"的结果冒充"已重排"。
    """
    s = get_settings()
    if not (s.rerank_enabled or "--rerank" in sys.argv):
        return {
            "status": "skipped",
            "model": s.rerank_model,
            "hint": "加 --rerank 或设 RERANK_ENABLED=true 后重跑",
        }

    reranker = CrossEncoderReranker()
    if not reranker.available:
        return {"status": "unavailable", "model": s.rerank_model}

    retriever = RerankedRetriever(hybrid, reranker, s.rerank_candidates)
    retriever.search(WARMUP, TOP_K)  # 预热：首次 predict 的初始化开销不计入延迟
    result = evaluate(retriever, rows, with_latency=True)
    result.update({"status": "ok", "model": s.rerank_model, "candidates": s.rerank_candidates})
    return result


def cache_benchmark(rows: list[dict], store) -> dict:
    """用同一批问题连跑两遍，量化检索缓存命中后的延迟收益。

    第一遍全部未命中（真实检索），第二遍全部命中（只查缓存）。
    实际线上收益取决于重复提问率，这里给的是"命中时"的上界。
    """
    cache = RetrievalCache(maxsize=1024, ttl=300.0)
    inner = HybridRetriever(store, hybrid=True)
    inner.search("预热", TOP_K)  # 预热不走缓存，避免污染命中率统计
    retriever = CachedRetriever(inner, cache, revision_provider=lambda: store.revision)

    cold: list[float] = []
    warm: list[float] = []
    for bucket in (cold, warm):
        for row in rows:
            t0 = time.perf_counter()
            retriever.search(row["question"], TOP_K)
            bucket.append((time.perf_counter() - t0) * 1000)

    cold_med = statistics.median(cold)
    warm_med = statistics.median(warm)
    return {
        "repeats": len(rows),
        "hit_rate": cache.stats["hit_rate"],
        "cold_median_ms": round(cold_med, 2),
        "warm_median_ms": round(warm_med, 3),
        "speedup": round(cold_med / max(warm_med, 1e-6), 1),
    }


def check_baseline(report: dict, path: Path = BASELINE) -> list[str]:
    """与冻结基线比对，返回不达标项清单（空列表表示通过）。"""
    if not path.exists():
        return [f"未找到基线文件 {path}，请先运行一次评估并据此生成基线"]
    base = json.loads(path.read_text(encoding="utf-8"))

    problems: list[str] = []
    for variant in ("vector_only", "hybrid"):
        for metric in ("recall@1", "recall@4"):
            floor = base.get(variant, {}).get(metric)
            current = report[variant][metric]
            if floor is not None and current + 1e-9 < floor:
                problems.append(f"{variant}.{metric} 由 {floor:.4f} 回退到 {current:.4f}")

    # 语料/题量变了，历史基线就不再可比，必须显式重新核定
    if base.get("corpus_chunks") != report["corpus_chunks"]:
        problems.append(
            f"语料片段数 {base.get('corpus_chunks')} → {report['corpus_chunks']}，"
            "基线需重新核定（更新 eval/baseline.json）"
        )
    return problems


async def main() -> None:
    docs = await ingest_corpus()
    store = get_vector_store()
    rows = load_dataset()

    vector_only = HybridRetriever(store, hybrid=False)
    hybrid = HybridRetriever(store, hybrid=True)

    # 预热（首个查询包含嵌入/索引构建开销，不计入延迟统计）
    vector_only.search(WARMUP, TOP_K)
    hybrid.search(WARMUP, TOP_K)

    res_vector = evaluate(vector_only, rows, with_latency=True)
    res_hybrid = evaluate(hybrid, rows, with_latency=True)
    res_rerank = rerank_variant(hybrid, rows)
    res_cache = cache_benchmark(rows, store)

    report = {
        "corpus_docs": len(docs),
        "corpus_chunks": store.count(),
        "dataset_total": len(rows),
        "answerable": res_vector["answerable_total"],
        "top_k": TOP_K,
        "vector_only": res_vector,
        "hybrid": res_hybrid,
        "rerank": res_rerank,
        "cache": res_cache,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n语料：{len(docs)} 份文档 / {store.count()} 个片段  题目：{len(rows)} 题")
    print(f"{'指标':<22}{'纯向量':>12}{'混合检索':>12}")
    print("-" * 46)
    print(f"{'recall@1':<20}{res_vector['recall@1']:>12.1%}{res_hybrid['recall@1']:>12.1%}")
    print(f"{'recall@4':<20}{res_vector['recall@4']:>12.1%}{res_hybrid['recall@4']:>12.1%}")
    print(f"{'命中@4（题）':<19}{res_vector['hits@4']:>12}{res_hybrid['hits@4']:>12}")
    print(f"{'检索中位延迟(ms)':<16}{res_vector.get('latency_ms_median', 0):>12}"
          f"{res_hybrid.get('latency_ms_median', 0):>12}")
    print(f"{'超纲题top1相似度':<16}{res_vector.get('oos_top1_sim_avg', 0):>12}"
          f"{res_hybrid.get('oos_top1_sim_avg', 0):>12}")
    if res_rerank.get("status") == "ok":
        print(f"\n精排（{res_rerank['model']}，候选池 {res_rerank['candidates']}）："
              f"recall@1 {res_rerank['recall@1']:>6.1%} | "
              f"recall@4 {res_rerank['recall@4']:>6.1%} | "
              f"中位延迟 {res_rerank.get('latency_ms_median', 0)}ms")
    elif res_rerank.get("status") == "unavailable":
        print(f"\n精排未启用：模型 {res_rerank['model']} 不可用，已跳过对照"
              "（如实标注，不影响其它指标）")
    else:
        print(f"\n精排未参与评估（{res_rerank['model']}）："
              "如需测量请加 --rerank 或设 RERANK_ENABLED=true")

    print(f"\n检索缓存（同一批 {res_cache['repeats']} 题重复提问）："
          f"命中率 {res_cache['hit_rate']:.1%}，"
          f"中位延迟 {res_cache['cold_median_ms']}ms → {res_cache['warm_median_ms']}ms"
          f"（{res_cache['speedup']}x）")
    print(f"报告已写入：{REPORT}")

    fixed = {m['id'] for m in res_vector["misses"]} - {m['id'] for m in res_hybrid["misses"]}
    broke = {m['id'] for m in res_hybrid["misses"]} - {m['id'] for m in res_vector["misses"]}
    print(f"混合检索相较纯向量：修复 {len(fixed)} 题 {sorted(fixed)}；"
          f"新增漏检 {len(broke)} 题 {sorted(broke)}")

    if "--check-baseline" in sys.argv:
        problems = check_baseline(report)
        if problems:
            print("\n[门禁] 基线校验未通过：")
            for p in problems:
                print(f"  - {p}")
            sys.exit(1)
        print("\n[门禁] 基线校验通过。")


if __name__ == "__main__":
    asyncio.run(main())
