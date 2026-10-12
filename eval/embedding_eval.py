import os
# 【必须放在最前面！】国内镜像加速，解决10060超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
corpus_dir = os.path.join(BASE_DIR, "corpus")
baseline_dir = os.path.join(BASE_DIR, "baseline")
os.makedirs(baseline_dir, exist_ok=True)

# 加载知识库md文档
corpus = []
doc_names = []
name2idx = {}
files = sorted([f for f in os.listdir(corpus_dir) if f.endswith(".md")])
for idx, fname in enumerate(files):
    fp = os.path.join(corpus_dir, fname)
    with open(fp, "r", encoding="utf-8") as f:
        text = f.read().strip()
    corpus.append(text)
    doc_names.append(fname)
    name2idx[fname] = idx
print(f"✅ 加载文档总数：{len(corpus)}")

# 向量模型（现在走国内镜像下载）
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeds = model.encode(corpus, convert_to_numpy=True)

# 加载评测数据集
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

# 指标计算函数
def recall_at_k(retrieved_ids, gold_ids, k):
    topk = set(retrieved_ids[:k])
    return 1.0 if len(topk & set(gold_ids)) > 0 else 0.0

def mrr_score(retrieved_ids, gold_ids):
    for rank, did in enumerate(retrieved_ids):
        if did in gold_ids:
            return 1.0 / (rank + 1)
    return 0.0

K = 1
recall_total = 0.0
mrr_total = 0.0
valid_count = 0
result_items = []

for qid, item in enumerate(query_list, start=1):
    q = item["question"]
    gold_doc_name = item.get("doc")

    # 向量相似度打分
    q_emb = model.encode(q, convert_to_numpy=True)
    sims = doc_embeds @ q_emb / (np.linalg.norm(doc_embeds, axis=1) * np.linalg.norm(q_emb))
    ranked_idx = np.argsort(-sims).tolist()

    if gold_doc_name is None:
        rk = None
        mrr = None
        gold_ids = []
    else:
        gold_ids = [name2idx[gold_doc_name]]
        rk = recall_at_k(ranked_idx, gold_ids, K)
        mrr = mrr_score(ranked_idx, gold_ids)
        recall_total += rk
        mrr_total += mrr
        valid_count += 1

    result_items.append({
        "qid": qid,
        "question": q,
        "gold_doc_name": gold_doc_name,
        "gold_doc_ids": gold_ids,
        "retrieved_doc_ids": ranked_idx[:K],
        "recall@1": rk,
        "mrr": mrr
    })

avg_recall = recall_total / valid_count if valid_count > 0 else 0
avg_mrr = mrr_total / valid_count if valid_count > 0 else 0

print("\n===== 纯向量 Embedding 评测汇总 =====")
print(f"有效样本数 = {valid_count}")
print(f"Recall@{K} = {avg_recall:.4f}")
print(f"MRR        = {avg_mrr:.4f}")

# 保存基线指标
base_path = os.path.join(baseline_dir, "embedding_baseline.json")
base_data = {
    "method": "Pure Embedding",
    "K": K,
    "valid_answerable_count": valid_count,
    "avg_recall@1": avg_recall,
    "avg_mrr": avg_mrr
}
with open(base_path, "w", encoding="utf-8") as f:
    json.dump(base_data, f, ensure_ascii=False, indent=2)

# 保存每条query详细记录
report_path = os.path.join(baseline_dir, "embedding_report.json")
report_data = {
    "summary": base_data,
    "details": result_items
}
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)

print("\n✅ 基线指标保存至 baseline/embedding_baseline.json")
print("✅ 详细查询记录保存至 baseline/embedding_report.json")
