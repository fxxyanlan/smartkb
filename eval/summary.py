import os
import json

# 当前文件在 eval/ 目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
baseline_folder = os.path.join(BASE_DIR, "baseline")
# 生成的报告放到【项目根目录】
out_md_path = os.path.join(BASE_DIR, "../report.md")

rows = []
# 遍历baseline文件夹下所有json
for fname in os.listdir(baseline_folder):
    if not fname.endswith("baseline.json"):
        continue
    file_path = os.path.join(baseline_folder, fname)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    method_name = data["method"]
    alpha_val = data.get("alpha", "-")
    valid_sample = data["valid_answerable_count"]
    recall1 = f'{data["avg_recall@1"]:.4f}'
    mrr_val = f'{data["avg_mrr"]:.4f}'
    rows.append([method_name, alpha_val, valid_sample, recall1, mrr_val])

# 构造markdown文本
md_content = """# RAG检索方案消融实验报告
> 评测任务：文档级检索召回评估
> 评测指标：Recall@1（Top1召回率）、MRR（平均倒数排名）
> 测试数据集：dataset.jsonl
> 知识库文档：eval/corpus目录下md文档

| 检索方案 | Alpha权重(BM25占比) | 有效可回答样本 | Recall@1 | MRR |
| ---- | ---- | ---- | ---- | ---- |
"""
# 填充表格每一行
for row in rows:
    md_content += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"

# 下面是实验分析部分，你可以按需修改文字
md_content += """
## 各检索方案简介
1. **纯BM25**：基于jieba中文分词的关键词检索，擅长精确匹配专业名词、编号。缺点：同义改写问题召回效果差。
2. **纯Embedding向量检索**：all-MiniLM-L6-v2语义向量，适合语义相同、用词不同的自然语言提问；缺点：专有名词容易召回错误。
3. **BM25+Embedding混合检索**：两路分数归一化后加权融合，兼顾关键词精确匹配和语义理解能力。
4. **混合检索 + Rerank精排**：混合粗召回捞出Top10候选文档，使用BGE-Reranker交叉编码器做精细重排，最后取Top1。

## 实验结论与项目局限性
1. 本数据集关键词特征强，因此纯BM25取得很高指标；在真实业务用户自然语言提问场景，混合检索与Rerank的优势会进一步体现。
2. 当前实验为**文档级召回**，工业落地时需要将长文档切分为Chunk片段，在片段层面检索，减少送入LLM的上下文长度。
3. 本次离线评测**仅评估检索环节召回质量**，不评估大模型最终生成答案的准确性、幻觉情况。
4. 离线测试集存在分布偏差，离线指标无法完全等价线上真实用户效果。

## 复现方式
1. 进入eval目录，依次执行4个评测脚本，生成baseline指标json
2. 运行 `python summary.py`，自动更新report.md
"""

# 写入report.md
with open(out_md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print("==== 自动生成的实验表格 ====")
print(md_content)
print(f"\n✅ 生成完成！文件输出路径：{out_md_path}")
