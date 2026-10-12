import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
corpus_dir = os.path.join(BASE_DIR, "corpus")
baseline_dir = os.path.join(BASE_DIR, "baseline")
os.makedirs(baseline_dir, exist_ok=True)

# 加载文档
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

# 向量模型
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeds = model.encode(corpus, convert_to_numpy=True)

# BM25 中文分词索引
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 加载评测集
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

# 指标函数
def recall_at_k(retrieved_ids, gold_ids, k):
    topk = set(retrieved_ids[:k])
    return 1.0 if len(topk & set(gold_ids))>0 else 0.0

def mrr_score(retrieved_ids, gold_ids):
    for rank, did in enumerate(retrieved_ids):
        if did in gold_ids:
            return 1.0/(rank+1)
    return 0.0

alpha = 0.5
K = 1
recall_total = 0.0
mrr_total =0.0
valid_count=0
result_items = []

for qid, item in enumerate(query_list, start=1):
    q = item["question"]
    gold_doc_name = item.get("doc")

    # 向量打分 + 归一化
    q_emb = model.encode(q, convert_to_numpy=True)
    sims = doc_embeds @ q_emb / (np.linalg.norm(doc_embeds, axis=1)*np.linalg.norm(q_emb))
    emb_norm = (sims - np.min(sims)) / (np.max(sims) - np.min(sims))

    # BM25打分 + 归一化
    query_tokens = jieba.lcut(q)
    bm25_scores = np.array(bm25.get_scores(query_tokens))
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    # 分数融合
    final_score = alpha * bm25_norm + (1-alpha)*emb_norm
    ranked_idx = np.argsort(-final_score).tolist()

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
        valid_count +=1

    result_items.append({
        "qid":qid,
        "question":q,
        "gold_doc_name":gold_doc_name,
        "gold_doc_ids":gold_ids,
        "retrieved_doc_ids": ranked_idx[:K],
        "recall@1":rk,
        "mrr":mrr
    })

avg_recall = recall_total / valid_count if valid_count>0 else 0
avg_mrr = mrr_total / valid_count if valid_count>0 else 0

print("\n===== 混合检索 BM25+Embedding 评测汇总 =====")
print(f"alpha={alpha}, 有效样本数 = {valid_count}")
print(f"Recall@{K} = {avg_recall:.4f}")
print(f"MRR        = {avg_mrr:.4f}")

# 保存基线
base_path = os.path.join(baseline_dir,"hybrid_baseline.json")
base_data = {
    "method":"Hybrid(BM25+Embedding)",
    "alpha":alpha,
    "K":K,
    "valid_answerable_count":valid_count,
    "avg_recall@1":avg_recall,
    "avg_mrr":avg_mrr
}
with open(base_path,"w",encoding="utf-8") as f:
    json.dump(base_data,f,ensure_ascii=False,indent=2)

report_path = os.path.join(baseline_dir,"hybrid_report.json")
report_data = {"summary":base_data, "details":result_items}
with open(report_path,"w",encoding="utf-8") as f:
    json.dump(report_data,f,ensure_ascii=False,indent=2)

print("\n✅ 基线指标保存至 baseline/hybrid_baseline.json")
print("✅ 详细查询记录保存至 baseline/hybrid_report.json")
