# SmartKB 测试全流程实操手册

> 适用对象：需要独立复现本项目全部测试与指标的同学。
> 覆盖：测试准备 → 用例执行 → 结果校验 → 问题记录；并把「成果产出」里每一个数据指标的**含义 / 采集命令 / 预期结果 / 校验标准 / 异常处理**逐项拆开。
> 平台：Windows PowerShell（命令均按此给出；Linux/macOS 把 `.\.venv\Scripts\python.exe` 换成 `python` 即可）。
> 原则：**所有指标必须能被你自己重跑出来**。跑不出来或对不上的，一律按"问题"记录，不得直接抄报告数字。

---

## 0. 全景：测试分四层，门禁有四道

| 层 | 跑什么 | 命令 | 耗时量级 | 依赖 |
|---|---|---|---|---|
| L1 静态检查 | 代码规范 | `ruff check .` | 秒级 | 无 |
| L2 单元 + 接口测试 | 85 条用例 + 覆盖率 | `pytest --cov=app ...` | 十秒级 | 无网络（全用桩件） |
| L3 离线检索评估 | recall@k / 延迟 / 缓存 | `python scripts/evaluate.py --check-baseline` | 分钟级 | 需 Embedding 模型（可缓存） |
| L4 全链路人工验收 | 生成准确率 / 拒答率 / 首字时延 | 见 §6.7（手工） | 半小时级 | **需真实 DeepSeek API Key** |

CI 四道门禁（`.github/workflows/ci.yml`）：`ruff` → `pytest`（覆盖率 ≥ 85%）→ `evaluate.py --check-baseline` → `docker-build` 镜像冒烟。**任一失败即整条 CI 失败。**

关键设计：L2 全部使用内存替身（`tests/fakes.py`），不加载真实模型、不连真实向量库、不调真实 LLM，所以**离线、快、可复现**；L3 的评估数据写在临时目录（`tempfile.mkdtemp`），**不污染开发数据**。

---

## 1. 阶段一：测试准备

### 1.1 环境准备

| 步骤 | 操作 | 预期结果 | 异常处理 |
|---|---|---|---|
| 1 | 进入项目根目录 `cd <项目根>` | 能 `dir` 看到 `app/ tests/ eval/ scripts/` | 目录不对则后续 `pythonpath` 失效，全是大红叉 |
| 2 | 激活虚拟环境 `.\.venv\Scripts\Activate.ps1` | 提示符前出现 `(.venv)` | 提示"禁止运行脚本"→ 用 `.\.venv\Scripts\python.exe` 全路径调用，不激活也能跑 |
| 3 | 装依赖 `pip install -e ".[dev]"` | 无报错 | 网络慢→加国内镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

> **注意事项**：Windows 上直接敲 `python` 可能命中 WindowsApps 的应用执行别名存根，报 `The system cannot find the path specified`。**统一改用 `.\.venv\Scripts\python.exe -m ...`**，这是本项目最常踩的环境坑。

### 1.2 确认测试资产齐全

```powershell
dir tests            # 应有 11 个 test_*.py + conftest.py + fakes.py
dir eval\corpus      # 应有 12 个 .md
dir eval             # 应有 dataset.jsonl / baseline.json / report.json
```

| 资产 | 作用 | 缺失后果 |
|---|---|---|
| `tests/conftest.py` | 共享 fixture（client / fake_store / fake_llm / ingestion / chat_env / settings） | 所有用例报 fixture 未找到 |
| `tests/fakes.py` | `FakeStore` / `FakeLLM` / `FakeEmbedder` 内存替身 | L2 会尝试加载真实模型，极慢甚至失败 |
| `eval/corpus/*.md` | 12 份评估语料，与业务库隔离 | L3 报 `corpus_chunks` 为 0，基线必然失败 |
| `eval/dataset.jsonl` | 50 道题（可答 40 / 超纲 10） | L3 无题可算，分母为 0 |
| `eval/baseline.json` | 冻结基线，门禁比对基准 | L3 `--check-baseline` 直接报"未找到基线文件" |

### 1.3 准备配置文件

```powershell
copy .env.example .env
```

| 项 | 测试期建议值 | 原因 |
|---|---|---|
| `DEEPSEEK_API_KEY` | L1–L3 可留空 | 无 Key 不影响这四层 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 首次会联网下载（约 100 MB），之后走缓存 |
| `RERANK_ENABLED` | `false` | 重排模型约 1 GB；评估默认不参与，保持确定性 |
| `HYBRID_ENABLED` | `true` | 混合检索需开启，否则评估对比无意义 |

> **注意事项**：`.env` 不要提交到 Git。配置项大小写不敏感（`case_sensitive=False`），但环境变量名按字段大写写。

### 1.4 准备顺序（推荐）

先 L1 → 再 L2 → 再 L3 → 最后 L4。理由：前两层秒级反馈、零外部依赖，能先把"代码有没有坏"排除掉，再花时间在模型和网络上。

---

## 2. 阶段二：测试用例执行

### 2.1 L1 代码规范

```powershell
ruff check .
```

- **操作要点**：项目根目录执行，`pyproject.toml` 已配置 `line-length=100`、规则集 `E,F,I,N,W,B,UP`、忽略 `B008`（FastAPI 依赖注入惯例）。
- **预期结果**：`All checks passed!`
- **异常处理**：报 `E501` 超长行 → 按 100 列换行；报 `I001` 导入顺序 → `ruff check --fix .` 自动修；报 `B008` → 说明你动了 ignore 配置，恢复即可。

### 2.2 L2 单元 + 接口测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

带覆盖率（与 CI 完全一致）：

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=app --cov-report=term-missing --cov-fail-under=85
```

- **操作要点**：`asyncio_mode="auto"` 已配好，协程用例无需加 `@pytest.mark.asyncio`。
- **预期结果**：`85 passed`；覆盖率 `TOTAL ... 93%`（≥ 85% 门槛）。
- **异常处理**：
  - 用例间互相污染 → `conftest.py` 的 `_clean_overrides` 是 autouse，每例后清依赖覆盖；若自己新加了 `app.dependency_overrides`，记得同样清理。
  - 想单跑某一类：`pytest tests/test_api.py -q`；只看失败：`pytest -q -x --tb=short`。

**用例分层与代表用例**（便于你按层验收）：

| 层 | 文件 | 代表用例 |
|---|---|---|
| 解析器 | `test_loaders.py` | `test_load_unsupported_type_raises`（不支持的扩展名） |
| 入库 | `test_ingestion.py` | `test_duplicate_content_is_reused`（SHA-256 去重）、`test_corrupt_pdf_raises_parse_error` |
| 检索 | `test_retriever.py` | `test_tokenize_chinese_emits_unigram_and_bigram`、`test_rrf_fuse_rewards_multichannel_agreement`、`test_hybrid_recovers_chunk_vector_channel_missed` |
| 重排 | `test_reranker.py` | `test_cross_encoder_degrades_when_model_load_fails`（降级）、`test_build_retriever_cache_is_outermost`（链路顺序） |
| 缓存 | `test_cache.py` | `test_cache_invalidated_after_document_added`、`test_cached_retriever_key_includes_top_k` |
| RAG 流水线 | `test_rag.py` | `test_answer_no_context_short_circuits`、`test_stream_retries_when_failure_happens_before_any_output`、`test_stream_does_not_retry_after_partial_output` |
| 接口 | `test_api.py` | `test_chat_llm_unavailable_returns_503`、`test_upload_rejects_oversized_file`、`test_delete_missing_document_returns_404` |
| LLM 异常归一 | `test_llm.py` | `test_authentication_error_maps_to_unavailable`（鉴权失败→503） |

### 2.3 L3 离线检索评估

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py
.\.venv\Scripts\python.exe scripts\evaluate.py --check-baseline   # 加门禁
```

- **操作要点**：
  1. 脚本**先设置** `CHROMA_PERSIST_DIR` / `UPLOAD_DIR` 到临时目录，**再**导入 `app`（顺序不能改，否则污染开发数据）。
  2. 会把 `eval/corpus/*.md` 全部入库，再对 50 题跑检索。
  3. 每个变体先跑一次 `WARMUP="预热"` 查询，把模型加载/索引构建开销排除在延迟统计外。
- **预期结果**：控制台输出对比表 + `报告已写入：.../eval/report.json`。
- **异常处理**：
  - 卡在模型下载 → 首次联网下载 bge-small-zh-v1.5；网络差时设 `$env:HF_ENDPOINT="https://hf-mirror.com"` 后重跑。
  - 报 `corpus_chunks` 与基线不符 → 说明你改了 `eval/corpus/` 或分块参数，**必须显式重新核定基线**（见 §3.4）。

### 2.4 L4 全链路人工验收（需真实 Key）

无自动化脚本，按 §6.7 的手工步骤执行。

---

## 3. 阶段三：结果校验

### 3.1 三层校验的判据

| 层 | 通过判据 | 不通过的常见原因 |
|---|---|---|
| L1 | 退出码 0 / `All checks passed!` | 新代码引入未用导入、超长行 |
| L2 | `85 passed` 且覆盖率 ≥ 85% | 改了接口签名没同步用例 |
| L3 | `[门禁] 基线校验通过。` | recall 回退、或语料变更未重定基线 |

### 3.2 读懂 `eval/report.json`

| 字段 | 含义 |
|---|---|
| `corpus_docs` / `corpus_chunks` | 入库文档数 / 片段数（用于和基线核对可比性） |
| `dataset_total` / `answerable` | 总题数 / 可答题数（分母） |
| `vector_only` / `hybrid` | 两个变体的 `recall@1`、`recall@4`、`hits@*`、`latency_ms_median`、`misses` |
| `rerank` | 精排对照，三态：`ok` / `unavailable` / `skipped` |
| `cache` | `hit_rate`、`cold_median_ms`、`warm_median_ms`、`speedup` |

### 3.3 `misses` 是校验的抓手

`misses` 列出未命中的题号。**校验方法**：对比 `vector_only.misses` 与 `hybrid.misses`，脚本末尾会打印「修复 N 题 / 新增漏检 N 题」。预期为「修复 3 题、新增漏检 0 题」。**只要出现"新增漏检"，就必须按缺陷记录并排查**——这是检索质量回退的硬信号。

### 3.4 基线失效与重定流程

`check_baseline` 有两条规则：①`vector_only` / `hybrid` 的 `recall@1`、`recall@4` 不得低于基线；②`corpus_chunks` 必须与基线一致。

**当且仅当**你确实有意改了语料或分块参数时，才重定基线，步骤：

1. 跑一次评估，确认新指标是你**有意接受**的；
2. 用新 `report.json` 的值更新 `eval/baseline.json` 的 `corpus_chunks` 与两档 recall；
3. 在提交信息里写明"重新核定基线 + 原因"。

> ⚠️ **禁止**为了让 CI 变绿而直接下调基线数字。基线只能因"有意变更语料/算法"而重定，不能因"指标掉了"而重定。

---

## 4. 阶段四：问题记录

### 4.1 记录模板（每条缺陷一行不漏）

| 字段 | 说明 |
|---|---|
| 编号 | 如 `D-01` |
| 发现环节 | L1 / L2 / L3 / L4 |
| 现象 | 可观察到的结果（含报错原文关键行） |
| 复现命令 | 精确到可直接粘贴的一条命令 |
| 期望 | 应当是什么 |
| 根因 | 定位到的代码位置（文件 + 函数） |
| 严重度 | 高（功能不可用/数据错误）/ 中（边界错误）/ 低（体验/规范） |
| 状态 | 待修 / 已修 / 不修（附理由） |
| 回归方式 | 用哪条用例/哪条命令证明修好了 |

### 4.2 定级基准（沿用本项目评估报告口径）

- **高**：功能不可用、结果错误、数据丢失。例：多轮上下文失效（死代码）、缺失依赖声明。
- **中**：边界/异常路径错误。例：删除不存在文档返回 200、伪造扩展名触发 500、async 阻塞事件循环。
- **低**：规范/体验。例：CORS 无效组合、未用依赖。

### 4.3 定位流程（三步）

1. **缩范围**：先跑 L1 → L2 → L3，判断是"代码坏了"还是"模型/数据问题"。
2. **抓单点**：`pytest tests/<file>::<test_name> -q -x --tb=long` 单点复现。
3. **对比基线**：L3 问题先看 `misses` 差集与 `corpus_chunks` 是否变化，再决定是"改算法"还是"重定基线"。

---

## 5. 成果指标逐项实操（手把手）

> 每一项按 **含义 → 采集 → 预期 → 校验 → 异常** 五步走。

### 5.1 用例数与覆盖率：`85 条 / 93%`

- **含义**：85 = `pytest` 收集到的用例总数；93% = `app/` 目录行覆盖率。
- **采集**：
  ```powershell
  .\.venv\Scripts\python.exe -m pytest --cov=app --cov-report=term-missing
  ```
- **预期**：`85 passed`；末行 `TOTAL ... 93%`。
- **校验**：覆盖率 ≥ 85%（CI 门槛）；`term-missing` 会列出未覆盖行号，`llm.py` 覆盖率偏低属已知（真实上游调用无法在无 Key 下测），其错误翻译分支已用 mock 覆盖。
- **异常**：覆盖率骤降 → 大概率是新增了未覆盖代码，用 `term-missing` 定位到具体行。

### 5.2 代码规范零告警

- **含义**：ruff 所有规则集通过。
- **采集**：`ruff check .`
- **预期**：`All checks passed!`
- **校验**：退出码 0。
- **异常**：见 §2.1。

### 5.3 召回率：`recall@1 75.0% → 87.5%`、`recall@4 92.5% → 100.0%`

- **含义**：`recall@k` = 在 top-k 结果里命中"标准相关片段"的题目占比。分母是 40 道可答题。`@1` 只看第一条，`@4` 看前四条。
- **采集**：`scripts\evaluate.py`，结果在控制台表格与 `report.json` 的 `vector_only` / `hybrid`。
- **命中判定逻辑**（务必理解，否则会误判）：
  ```python
  top_text = norm("".join(r["chunk_text"] for r in results))   # top-k 拼接
  hit_k    = all(norm(k) in top_text for k in row["must_contain"])  # 关键词全部出现才算命中
  ```
  `norm()` 会去空格、转小写、去全角空格。**是"关键词全含"的严格判定，不是语义相似度。**
- **预期**：`recall@1` 0.75 → 0.875；`recall@4` 0.925 → 1.0；修复 3 题、新增漏检 0 题。
- **校验**：与 `eval/baseline.json` 比对，`--check-baseline` 打印 `[门禁] 基线校验通过。`
- **异常**：
  - 数值对不上（例如 hybrid 拿到 0.75）→ 检查 `HYBRID_ENABLED` 是否为 true。
  - 分母不是 40 → 检查 `dataset.jsonl` 是否被改动。
  - **口径陷阱**：`75→87.5` 与 `92.5→100` 是**两个不同的 k**，不要写成"一条曲线从 75 涨到 100"。

### 5.4 检索中位延迟：`纯向量 ≈12ms / 混合 ≈14ms`

- **含义**：`statistics.median` 统计的每问耗时的中位数（毫秒），已排除首个 `预热` 查询。
- **采集**：`report.json` 的 `vector_only.latency_ms_median` / `hybrid.latency_ms_median`。
- **预期**：两者处于同一毫秒量级，混合略高。
- **校验**：量级一致即可（受机器影响，绝对值不必卡死）；**若混合比纯向量慢一个数量级，说明 BM25 索引在每次查询都重建，需按缺陷记录**。
- **异常**：首次运行偏大 → 忘了预热；确认脚本里的 `WARMUP` 调用存在。

### 5.5 缓存收益：`命中率 50.0%，13ms → 0.002ms`

- **含义**：同一批 50 题连问两遍，第二遍全部命中；`hit_rate` = 命中数 / 总请求数 ≈ 50%。
- **采集**：`report.json` 的 `cache`：`hit_rate`、`cold_median_ms`、`warm_median_ms`、`speedup`。
- **预期**：`hit_rate≈0.5`；`warm_median_ms` 在毫秒以下（约 0.002ms）。
- **校验**：预热查询不走缓存（`inner.search("预热")` 在包缓存之前），所以命中率是干净的 50%。**若命中率明显高于 50%，说明预热被计入了统计**。
- **异常**：`warm_median_ms` 与 `cold` 接近 → 缓存没生效，检查 `revision_provider` 是否返回了每次都变的版本号。

### 5.6 并发非阻塞：`3 路并发 29.8ms`

- **含义**：3 个请求同时发，总耗时远小于 3 倍单请求耗时，证明事件循环没被阻塞。
- **采集**：手工（无专用脚本）：
  ```powershell
  # 先确保服务已起：uvicorn app.main:app
  # 用 PowerShell 并发打 3 个请求并计时
  $sw=[Diagnostics.Stopwatch]::StartNew()
  1..3 | ForEach-Object -Parallel { Invoke-RestMethod -Uri "http://localhost:8000/api/v1/chat" -Method Post -ContentType "application/json" -Body '{"question":"这份文档讲了什么?"}' } -ThrottleLimit 3
  $sw.Stop(); $sw.ElapsedMilliseconds
  ```
- **预期**：总耗时明显 < 3× 单请求；日志中请求时间戳相互交叠而非串行。
- **校验**：用桩 LLM 时最干净（避免真实上游波动干扰）。
- **异常**：总耗时 ≈ 3 倍 → 有阻塞调用没走 `asyncio.to_thread`，按缺陷记录并检查 `IngestionService` / `RAGPipeline` 的阻塞点。

### 5.7 生成质量：`18/18`、`6/6`、`100% 引用`

- **含义**：4 份文档、27 题（可答 18 + 超纲 6 + 其余）下的人工核对结果——可答题答案正确 18/18；超纲题拒答 6/6；每条回答都带来源引用。
- **采集**：**无自动化脚本，需真实 API Key 人工验收**（步骤见 §6.7）。
- **预期**：见上。
- **校验**：逐题对照原文核对"答案是否出自原文"；超纲题必须拒答而非编造。
- **异常**：超纲题被作答 → 检查 system prompt 约束与空检索短路是否被改坏。
- **诚实说明**：这部分是人工/半人工核对，不属 `evaluate.py` 的自动指标，报告里单独列出、不与 recall 混算。

### 5.8 首字时延：`约 0.5s`

- **含义**：从发出流式请求到收到**第一个 token** 的时间。
- **采集**：手工，用带时间戳的流式读取（§6.7 步骤 5）。
- **预期**：中位约 510 ms（随网络与模型负载浮动）。
- **校验**：与 `llm_first_token_timeout`（默认 20s）比对，远小于超时阈值。
- **异常**：长时间无首字 → 触发首字超时重试；仍失败则 SSE 回传 `{"type":"error"}` 事件。

### 5.9 缺陷修复：`12 类`

- **含义**：评估报告中记录并修复的 12 类真实缺陷（A–L）。
- **采集**：查阅 `docs/evaluation-report.md` 第三节「缺陷清单与修复状态汇总」。
- **校验**：每条缺陷都应有对应回归证据（用例或命令）。缺证据的条目视为未闭环。
- **异常**：若你重跑发现某条"已修"其实仍复现 → 按新缺陷记录，编号续接。

---

## 6. 端到端实操清单（可直接照着做）

**步骤 1｜静态检查**
```powershell
ruff check .                                  # 期望 All checks passed!
```

**步骤 2｜单元 + 接口 + 覆盖率**
```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest --cov=app --cov-report=term-missing --cov-fail-under=85
```

**步骤 3｜离线检索评估 + 门禁**
```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --check-baseline
```
期望末尾：`[门禁] 基线校验通过。`

**步骤 4｜查看机器可读报告**
```powershell
Get-Content eval\report.json -Raw | ConvertFrom-Json
```

**步骤 5｜（可选）测精排对照**
```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --rerank
```
- 有本地重排模型 → 输出精排 `recall@1/@4` 与延迟；
- 模型不可用 → 明确标注 `unavailable`；
- 不开启 → 标注 `skipped`。
**三种状态都不允许用"未重排"的结果冒充"已重排"。**

**步骤 6｜手工验证接口语义（起服务后另开终端）**
```powershell
uvicorn app.main:app --reload
# 另开终端：
curl.exe -X POST http://localhost:8000/api/v1/documents/upload -F "file=@某文档.pdf"
curl.exe -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" -d "{\"question\":\"这份文档讲了什么?\"}"
curl.exe -X DELETE http://localhost:8000/api/v1/documents/不存在的id   # 期望 404
curl.exe http://localhost:8000/api/v1/health/ready                     # 期望含缓存命中率
```

**步骤 7｜全链路人工验收（需真实 Key）**
1. 在 `.env` 填好 `DEEPSEEK_API_KEY`，起服务；
2. 上传 4 份不同主题文档，确认列表 4 条、无重复；
3. 按 27 题清单逐题提问，记录：答案是否正确（对照原文）、是否附引用、超纲题是否拒答；
4. 用 `curl.exe -N -X POST .../chat/stream` 观察首字到达时间（`-N` 关闭缓冲）；
5. 汇总：可答 `正确数/18`、拒答 `正确数/6`、引用覆盖率 —— 与报告比对。

---

## 7. 异常处理速查表

| 症状 | 可能原因 | 处理 |
|---|---|---|
| `python.exe ... cannot find the path` | Windows Apps 执行别名存根 | 用 `.\.venv\Scripts\python.exe` |
| `Activate.ps1 无法加载，禁止运行脚本` | 执行策略限制 | 不激活，直接用 venv 全路径 |
| 模型下载卡住/超时 | 网络受限 | `$env:HF_ENDPOINT="https://hf-mirror.com"` |
| 门禁报"基线需重新核定" | `corpus_chunks` 变化 | 确认是有意改语料 → 更新 `baseline.json` |
| hybrid 指标与纯向量相同 | `HYBRID_ENABLED` 未开 | 置为 true 后重跑 |
| 出现"新增漏检" | 检索质量回退 | 按缺陷记录，查 BM25 分词/融合逻辑 |
| 缓存命中率异常高 | 预热被计入统计 | 确认预热查询在包缓存之前 |
| 并发耗时为单请求的 N 倍 | 阻塞调用未卸载 | 检查 `asyncio.to_thread` 覆盖点 |
| 超纲题被作答 | prompt 约束/短路被改坏 | 回归 `test_answer_no_context_short_circuits` |
| 报表 `rerank: unavailable` | 重排模型不可达 | 属预期降级，非缺陷；需要指标则本地备好模型 |

---

## 8. 一句话记住校验底线

**能重跑 → 才敢引用；对不上 → 记缺陷；改了语料 → 才动基线；模型没跑 → 标 `skipped`，不标成绩。**
