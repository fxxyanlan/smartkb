# BM25 + 向量 + Rerank 检索链路：报错排查与实现记录

> 目标：完成「BM25 + 向量 + Rerank」三路检索链路的可运行实现，并把结果写入 `eval/baseline/`，
> 与既有三组基线（纯 BM25 / 纯向量 / BM25+向量混合）格式统一，供四组横向对比。
> 记录内容：环境快照 → 报错复现 → 根因分析 → 解决步骤 → 配置调整 → 验证结果 → 局限说明。

---

## 一、环境快照（排查前）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.13.15 | 项目 `.venv` |
| sentence-transformers | 6.1.0 | 提供 `SentenceTransformer` / `CrossEncoder` |
| transformers | 5.17.0 | 新版 API |
| huggingface-hub | **1.32.0** | **1.x 已移除旧参数**（关键） |
| torch | 2.14.0 | CPU |
| numpy | 2.5.3 | |
| jieba | ❌ 未安装 | 排查时缺失 |
| rank-bm25 | ❌ 未安装 | 排查时缺失 |

模型资产（排查时已核实，**均已完整存在于本地**）：

| 模型 | 位置 | 状态 |
|---|---|---|
| all-MiniLM-L6-v2 | `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` | 完整（384 维） |
| bge-reranker-base | `model/bge-reranker-base`（`pytorch_model.bin` = **1060.73 MB**） | 完整 |

> **重要结论**：报错**不是模型没下下来**，而是依赖缺失 + 新版 huggingface-hub 参数不兼容 + 脚本引用了不存在的路径。

---

## 二、报错清单与根因

| # | 现象 | 根本原因 | 证据 |
|---|---|---|---|
| P1 | `ModuleNotFoundError: No module named 'jieba'` | 当前虚拟环境**未安装** `jieba` 与 `rank-bm25`，而 `bm25_demo.py` / `embedding_eval.py` / `hybrid_eval.py` / `hybrid_rerank_eval.py` 全部依赖它们 | 实际运行 `hybrid_rerank_eval.py` 第 2 行即抛错 |
| P2 | `UserWarning: The 'resume_download' argument is deprecated and ignored`、`'local_dir_use_symlinks' ... deprecated and ignored` | huggingface-hub **1.32.0** 已废弃这两个参数（写本地目录不再用软链接），旧脚本仍传 | 运行 `download_reranker.py` 输出的两条告警 |
| P3 | `Returning existing local_dir ... as remote repo cannot be accessed ([WinError 10060] 由于连接方在一段时间后没有正确答复...)` | `snapshot_download` 仍尝试联网校验；当前网络无法访问 hf-mirror，连接超时 | 运行 `download_reranker.py` 的实际输出 |
| P4 | `FileNotFoundError: ../test_questions.json`；且文档数会为 0 | 脚本引用**不存在**的 `../test_questions.json`，并从 `../docs` 读 `*.txt`（该目录只有 `.md`），过滤后文档数为 0 | 静态审查 + `Glob` 确认 `test_questions.json` 不存在、`docs/` 仅含 `.md`。**未实际触发**——被 P1 提前中断 |
| P5 | （无报错，但结果无意义）向量通道恒为 0 | 原脚本 `vec_scores = np.zeros_like(bm25_scores)` 是**占位桩**，等价于"纯 BM25"，并不是"BM25+向量" | 静态审查，原文件第 51 行 |

---

## 三、排查过程（时间线）

1. **先看现场，不猜**：列出 `eval/` 全部文件，发现已有一套独立实验脚本 + `eval/baseline/` 结果，且 `hybrid_rerank_baseline.json` 是 **0 字节空文件**——说明 rerank 那一组从未成功产出。
2. **读既有脚本定"基线口径"**：`bm25_demo.py` / `embedding_eval.py` / `hybrid_eval.py` 三份脚本确认了口径——
   - 语料：`eval/corpus/*.md`（12 份），**文档级**检索；
   - 数据集：`eval/dataset.jsonl`（50 题，40 题 answerable）；
   - 指标：`K=1` 的 `Recall@1` 与 `MRR`；
   - 输出：`baseline/*_baseline.json`（汇总）+ `baseline/*_report.json`（逐题明细）。
3. **环境体检**：打印包版本 → 发现 `jieba` / `rank-bm25` 缺失（P1）；发现 huggingface-hub 已到 1.32（P2 的伏笔）。
4. **核实模型资产**：查 `model/` 与 HF 缓存 → 两个模型**都完整**，排除"下载失败"这一最初假设。
5. **实际复现**：运行 `download_reranker.py` → 复现 P2、P3；运行 `hybrid_rerank_eval.py` → 复现 P1。
6. **验证离线可加载**：设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，`CrossEncoder(local path, local_files_only=True)` 加载成功（2.4s），`SentenceTransformer('all-MiniLM-L6-v2')` 成功（0.7s，384 维）。
7. **验证基线可复现**：重跑 `embedding_eval.py` / `hybrid_eval.py`，指标与存档**完全一致**（0.2250/0.3694、0.7000/0.8271）→ 证明环境修好后，口径没漂移。
8. **改写 rerank 脚本**并运行 → 产出结果，格式对齐。
9. **聚合校验**：跑 `summary.py`，四组同时出现在对比表中。

---

## 四、解决步骤（按问题逐条）

### P1：补装依赖
```powershell
.\.venv\Scripts\python.exe -m pip install jieba rank-bm25 -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 30
```
结果：`jieba 0.42.1`、`rank-bm25 0.2.2` 安装成功（走本地 wheel 缓存）。
> 说明：项目 `app/core` 的检索层坚持零新增依赖（自实现 BM25）；但 `eval/` 这套实验脚本原本就以 `jieba` + `rank-bm25` 为口径，为保持与既有基线**可比**，沿用同一套工具。

### P2 + P3：修正下载脚本
`eval/download_reranker.py` 改动：
- **删除** `resume_download=True`（1.x 已忽略）；
- **删除** `local_dir_use_symlinks=False`（1.x 已忽略，写本地目录本就不用软链接）；
- **新增"本地已完整则短路"**：校验 `pytorch_model.bin` / `config.json` / `tokenizer.json` 存在且非空，命中则直接返回，**不再联网**，从根本上规避 `WinError 10060`；
- 镜像改用 `os.environ.setdefault("HF_ENDPOINT", ...)`，保留外部覆盖能力。

### P4 + P5：重写 rerank 评测脚本
`eval/hybrid_rerank_eval.py` 关键改动：
- 路径全部改为基于 `BASE_DIR` 的绝对路径（不再依赖 `cwd`，也不再引用不存在的 `../test_questions.json`）；
- 语料改读 `eval/corpus/*.md`，数据集改读 `eval/dataset.jsonl`，与三组基线同源；
- **向量通道接真实模型** `all-MiniLM-L6-v2`，融合权重 `alpha=0.5`、min-max 归一化——**与 `hybrid_eval.py` 完全一致**，保证精排前的候选排序可复现 hybrid 基线；
- 新增 `minmax()` 除零保护（`max==min` 时返回全 0，避免 `nan`）；
- 输出改为 `baseline/hybrid_rerank_baseline.json` + `baseline/hybrid_rerank_report.json`，字段与既有基线一致。

### 配置调整汇总

| 配置项 | 调整 | 原因 |
|---|---|---|
| `HF_HUB_OFFLINE` | 脚本内 `setdefault("1")` | 阻断联网校验，规避超时 |
| `TRANSFORMERS_OFFLINE` | 脚本内 `setdefault("1")` | 同上 |
| `HF_ENDPOINT` | 保留 hf-mirror（可外部覆盖） | 仅在确实需下载时生效 |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | 与既有基线保持一致（可比性） |
| `ALPHA` | `0.5` | 与 `hybrid_eval.py` 一致 |
| `TOP_K_RETRIEVE` | `10` | 粗召回候选数，交给 rerank |
| `K` | `1` | 与三组基线一致（Recall@1 / MRR） |

---

## 五、验证结果

### 5.1 四组方案横向对比（`eval/report.md`，由 `summary.py` 自动生成）

| 检索方案 | Alpha(BM25占比) | 有效样本 | Recall@1 | MRR |
|---|---|---|---|---|
| BM25Okapi+jieba | - | 40 | 1.0000 | 1.0000 |
| Pure Embedding | - | 40 | 0.2250 | 0.3694 |
| Hybrid(BM25+Embedding) | 0.5 | 40 | 0.7000 | 0.8271 |
| **Hybrid(BM25+Embedding)+Rerank** | 0.5 | 40 | **1.0000** | **1.0000** |

### 5.2 产物清单

| 文件 | 内容 |
|---|---|
| `eval/baseline/hybrid_rerank_baseline.json` | 汇总指标（原为空文件，现已填充） |
| `eval/baseline/hybrid_rerank_report.json` | 50 题逐题明细（`qid / question / gold_doc_ids / retrieved_doc_ids / recall@1 / mrr`） |
| `eval/report.md` | 四组方案对比表（项目根目录） |

### 5.3 校验方式
```powershell
cd eval
..\.venv\Scripts\python.exe download_reranker.py      # 期望：本地已完整，跳过下载
..\.venv\Scripts\python.exe hybrid_rerank_eval.py     # 期望：Recall@1=1.0000  MRR=1.0000
..\.venv\Scripts\python.exe summary.py                # 期望：表格含四行
```
三项均**无报错**，退出正常。

---

## 六、必须写清的局限（避免过度解读）

1. **语料太小，指标接近天花板**：仅 12 份文档，而粗召回候选池 `TOP_K_RETRIEVE=10`，已覆盖约 83% 的语料。rerank 的 `Recall@1=1.0` 更多反映"候选池足够大"，**不能**直接推断 rerank 在真实场景有同等增益。
2. **纯 BM25 组同样满分**：本数据集问题含大量关键词/编号（`rrf_k`、`429`、`20mb`），关键词特征极强，因此 BM25 与 rerank 都容易满分；混合组 0.7 反而更低，是**归一化融合在强关键词数据集上的副作用**，不代表混合检索更差。
3. **文档级，非片段级**：本实验在"整篇文档"粒度检索，工业落地需切 Chunk 后在片段级检索，指标不可直接套用。
4. **只评检索、不评生成**：不含 LLM 生成质量与幻觉评估。
5. **与主项目评估口径不同**：项目主评估（`scripts/evaluate.py` → `eval/report.json`）是**片段级**、含 `recall@4` 与延迟/缓存；本文件是 `eval/` 下**文档级消融实验**，两套数字不可混算。

> 另注：`eval/bm25_demo.py` 会把结果写入 `eval/report.json`，**与主项目评估报告同名**，重跑它会覆盖主报告的 `vector_only/hybrid/rerank/cache` 内容。本次排查已刻意规避，未运行该脚本；后续如需运行请先备份。
