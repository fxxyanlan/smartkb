"""测试集难度画像（纯 jieba，无需加载向量/重排模型，毫秒级完成）。

用于量化判断一个问题集是否「过于简单、类型过于集中」：

    - 类型分布：answerable / oos / vague（v2 另有 difficulty / semantic / qtype）
    - 词面重合度：问题分词与金标文档分词的交集占问题词数的比例
      比例越高，说明越依赖字面精确匹配（这类问题无法区分 BM25 与语义检索）
    - 「近字面题」占比：重合度 >= 0.8 的问题比例
    - must_contain 平均词数：越少越接近单点事实题
    - 文档覆盖：每个语料文档被多少道题命中

用法：
    python eval/dataset_profile.py
    python eval/dataset_profile.py --corpus eval/corpus_v2 --dataset eval/dataset_v2.jsonl --tag v2
"""

from __future__ import annotations

import argparse
import json
import os

import jieba

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "baseline")

# 中文高频虚词（对语义贡献低，但会虚高字面重合度）
STOPWORDS = set(
    "的 是 了 吗 呢 吧 啊 在 有 和 与 及 以及 或 对 为 被 把 从 到 中 上 下 内 外 "
    "请问 什么 怎么 怎样 如何 多少 哪 哪些 哪个 多久 多长 多大 需要 可以 会 要 能 "
    "个 一 二 三 四 五 六 七 八 九 十 一般 大约 分别 以及 我们 你们 他们 它".split()
)


def load_corpus(corpus_dir: str):
    corpus, names, name2idx = [], [], {}
    for idx, fname in enumerate(sorted(f for f in os.listdir(corpus_dir) if f.endswith(".md"))):
        with open(os.path.join(corpus_dir, fname), encoding="utf-8") as f:
            corpus.append(f.read().strip())
        names.append(fname)
        name2idx[fname] = idx
    return corpus, names, name2idx


def load_dataset(path: str):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def content_tokens(text: str) -> set[str]:
    return {t for t in jieba.lcut(text) if t.strip() and t not in STOPWORDS}


def main():
    parser = argparse.ArgumentParser(description="测试集难度画像")
    parser.add_argument("--corpus", default=os.path.join(BASE_DIR, "corpus"))
    parser.add_argument("--dataset", default=os.path.join(BASE_DIR, "dataset.jsonl"))
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()

    os.makedirs(BASELINE_DIR, exist_ok=True)

    corpus, names, name2idx = load_corpus(args.corpus)
    items = load_dataset(args.dataset)

    type_counter, difficulty_counter, semantic_counter, qtype_counter = {}, {}, {}, {}
    coverage = {name: 0 for name in names}
    overlaps, must_counts, per_query = [], [], []

    for item in items:
        qtype = item.get("type", "unknown")
        type_counter[qtype] = type_counter.get(qtype, 0) + 1
        difficulty_counter[item.get("difficulty", "未标注")] = \
            difficulty_counter.get(item.get("difficulty", "未标注"), 0) + 1
        semantic_counter[item.get("semantic", "未标注")] = \
            semantic_counter.get(item.get("semantic", "未标注"), 0) + 1
        qtype_counter[item.get("qtype", "未标注")] = \
            qtype_counter.get(item.get("qtype", "未标注"), 0) + 1

        gold_names = [n for n in ([item["doc"]] if item.get("doc") else [])
                      + (item.get("docs") or [])]
        if not gold_names:
            continue
        for n in set(gold_names):
            if n in coverage:
                coverage[n] += 1

        q_tokens = content_tokens(item["question"])
        doc_tokens = set()
        for n in gold_names:
            if n in name2idx:
                doc_tokens |= content_tokens(corpus[name2idx[n]])
        ratio = len(q_tokens & doc_tokens) / len(q_tokens) if q_tokens else 0.0
        overlaps.append(ratio)
        must_counts.append(len(item.get("must_contain") or []))
        per_query.append({
            "id": item["id"],
            "question": item["question"],
            "gold_docs": gold_names,
            "lexical_overlap": round(ratio, 4),
            "difficulty": item.get("difficulty"),
            "semantic": item.get("semantic"),
            "qtype": item.get("qtype"),
        })

    answerable = len(overlaps)
    near_literal = sum(1 for r in overlaps if r >= 0.8)

    def mean(v):
        return round(sum(v) / len(v), 4) if v else 0.0

    profile = {
        "tag": args.tag,
        "corpus_dir": os.path.relpath(args.corpus, BASE_DIR),
        "dataset": os.path.relpath(args.dataset, BASE_DIR),
        "doc_count": len(corpus),
        "query_count": len(items),
        "type_distribution": type_counter,
        "difficulty_distribution": difficulty_counter,
        "semantic_distribution": semantic_counter,
        "qtype_distribution": qtype_counter,
        "answerable_count": answerable,
        "lexical_overlap_mean": mean(overlaps),
        "lexical_overlap_max": round(max(overlaps), 4) if overlaps else 0.0,
        "near_literal_count": near_literal,
        "near_literal_ratio": round(near_literal / answerable, 4) if answerable else 0.0,
        "must_contain_mean": mean(must_counts),
        "doc_coverage": coverage,
        "per_query": per_query,
    }

    out_path = os.path.join(BASELINE_DIR, f"dataset_profile_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"===== 测试集难度画像（{args.tag}）=====")
    print(f"语料文档数        : {len(corpus)}")
    print(f"问题总数          : {len(items)}")
    print(f"类型分布          : {type_counter}")
    print(f"难度分布          : {difficulty_counter}")
    print(f"语义复杂度分布     : {semantic_counter}")
    print(f"查询类型分布       : {qtype_counter}")
    print(f"可回答样本数       : {answerable}")
    print(f"词面重合度 均值/最大: {mean(overlaps)} / {profile['lexical_overlap_max']}")
    print(f"近字面题(>=0.8)    : {near_literal} / {answerable} "
          f"= {profile['near_literal_ratio']:.2%}")
    print(f"must_contain 均词数: {mean(must_counts)}")
    print(f"\n✅ 画像已保存：{os.path.relpath(out_path, BASE_DIR)}")


if __name__ == "__main__":
    main()
