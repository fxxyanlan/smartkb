from rank_bm25 import BM25Okapi
import jieba
import os
import json

# 获取当前脚本文件所在目录（eval文件夹）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
corpus_dir = os.path.join(BASE_DIR, "corpus")

# ===================== 1. 读取corpus下12份md文档，构建【文件名→下标】映射 =====================
corpus = []
doc_name_list = []
name2idx = dict()  # key: md文件名，value: doc下标0~11

# sorted保持顺序稳定
file_list = sorted([f for f in os.listdir(corpus_dir) if f.endswith(".md")])

for idx, fname in enumerate(file_list):
    file_path = os.path.join(corpus_dir, fname)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    corpus.append(content)
    doc_name_list.append(fname)
    name2idx[fname] = idx

print(f"✅ 成功加载文档总数：{len(corpus)}")
for name, idx in name2idx.items():
    print(f"{name} → doc[{idx}]")

# 构建BM25索引
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# ===================== 2. 读取 dataset.jsonl 50条问题 =====================
dataset_path = os.path.join(BASE_DIR, "dataset.jsonl")
query_list = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        query_list.append(obj)
print(f"\n✅ 读取评测问题数量：{len(query_list)}")

# ===================== 3. 评测指标函数 Recall@K, MRR =====================
def recall_at_k(retrieved_doc_ids: list[int], gold_ids: list[int], k: int):
    topk_set = set(retrieved_doc_ids[:k])
    hit = len(set(gold_ids) & topk_set)
    return 1.0 if hit > 0 else 0.0

def mrr_score(retrieved_doc_ids: list[int], gold_ids: list[int]):
    for rank, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in gold_ids:
            return 1.0 / (rank + 1)
    return 0.0

# ===================== 4. 批量跑评测 =====================
K = 1
recall_total = 0.0
mrr_total = 0.0
valid_count = 0   # 只统计answerable样本
result_items = []

for qid, item in enumerate(query_list, start=1):
    query = item["question"]
    gold_doc_name = item.get("doc")

    tokens = jieba.lcut(query)
    # 获取所有文档分数，直接排序拿到下标
    scores = bm25.get_scores(tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    retrieved_ids = ranked_idx[:K]

    # 处理不可回答样本(doc为None)
    if gold_doc_name is None:
        rk = None
        mrr = None
        gold_doc_ids = []
    else:
        gold_doc_ids = [name2idx[gold_doc_name]]
        rk = recall_at_k(retrieved_ids, gold_doc_ids, K)
        mrr = mrr_score(retrieved_ids, gold_doc_ids)
        recall_total += rk
        mrr_total += mrr
        valid_count += 1

    result_items.append({
        "qid": qid,
        "question": query,
        "gold_doc_name": gold_doc_name,
        "gold_doc_ids": gold_doc_ids,
        "retrieved_doc_ids": retrieved_ids,
        "recall@1": rk,
        "mrr": mrr
    })

# 汇总指标（只用可回答样本计算均值）
if valid_count > 0:
    avg_recall = recall_total / valid_count
    avg_mrr = mrr_total / valid_count
else:
    avg_recall = 0
    avg_mrr = 0

print("\n===== BM25评测汇总（仅answerable样本） =====")
print(f"有效可回答样本数 = {valid_count}")
print(f"Recall@{K} = {avg_recall:.4f}")
print(f"MRR        = {avg_mrr:.4f}")

# 写入report.json
report_path = os.path.join(BASE_DIR, "report.json")
report = {
    "summary": {
        "K": K,
        "valid_answerable_count": valid_count,
        "avg_recall@1": avg_recall,
        "avg_mrr": avg_mrr,
        "total_query": len(query_list)
    },
    "details": result_items
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("\n✅ 评测结果已保存至 report.json")

# 同时输出baseline.json，存放BM25基线结果
baseline_path = os.path.join(BASE_DIR, "baseline.json")
baseline = {
    "method": "BM25Okapi+jieba",
    "K": K,
    "valid_answerable_count": valid_count,
    "avg_recall@1": avg_recall,
    "avg_mrr": avg_mrr
}
with open(baseline_path, "w", encoding="utf-8") as f:
    json.dump(baseline, f, ensure_ascii=False, indent=2)
print("✅ baseline基线指标保存至 baseline.json")
