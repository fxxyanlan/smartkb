# AskDocs 系统评估与优化报告

> 评估对象：AskDocs —— 基于 RAG 的文档问答系统
> 评估日期：2026-10-05 ｜ 版本：v1.0.0
> 评估方法：静态代码审查 + 真实栈端到端实测（真实 Embedding + 真实 Chroma + 真实 HTTP 链路）+ 自动化测试
> 配套文档：[功能测试报告](./test-report.md) ｜ [性能评估文档](./performance-report.md)

---

## 一、结论摘要

系统在 7 个维度上完成了一轮"体检 → 修复 → 加固 → 验证"的闭环。共定位并修复 **12 类真实缺陷**（含 3 类可复现的功能性 Bug），补全自动化测试至 **50 用例 / 92% 覆盖率**，且实测数据全部来自真实运行链路（含真实 DeepSeek 全链路复测）。

| 维度 | 优化前评级 | 优化后评级 | 关键动作 |
|---|---|---|---|
| 功能完整性 | B | **A** | 修复重复上传去重、删除语义、多轮上下文失效 |
| 交互体验流畅度 | B- | **A-** | 流式错误可见、上传状态反馈、IME 回车、打字机光标 |
| 响应准确性 | B | **A-** | 空库短路防幻觉、引用结构校验、检索命中 5/5 |
| 代码结构合理性 | B+ | **A** | 统一领域异常层、依赖注入、分层清晰、ruff 零告警 |
| 错误处理机制 | C | **A-** | 统一异常映射、客户端错 4xx/服务端错 5xx 严格区分 |
| 性能表现 | 未知 | **A-** | 事件循环去阻塞、检索中位 10.7ms、并发非阻塞 |
| 可扩展性 | B | **A-** | 依赖注入可替换、配置集中、演进路径明确 |

> 评级为相对本项目定位（实习求职展示 / 单机演示级）的相对评价，非生产集群级标准。

---

## 二、七维度详细评估

### 1. 功能完整性 —— 优化前 B → 优化后 A

**优化前问题**

| 编号 | 问题 | 现象 | 根因 |
|---|---|---|---|
| A | 删除不存在的文档返回 200 | `DELETE /documents/{id}` 对无效 id 返回 `deleted_chunks: 0` | 服务层从不抛 `KeyError`，接口层 404 分支是死代码 |
| C | 重复上传无去重 | 同一 PDF 连传 2 次，文档列表出现多条，检索引用重复 | 入库前无内容指纹比对（实测发现线上库曾有 6 份重复文档） |
| D | 多轮上下文失效 | 追问时模型不记得上一轮内容 | `history` 参数全程未传给 LLM，是死代码 |

**改进方案与实施步骤**

1. **删除语义修正**：`IngestionService.delete()` 中，当删除分块数为 0 时抛 `DocumentNotFoundError` → 全局处理器映射为 **404**。
2. **内容去重**：入库时计算文件字节的 **SHA-256**；`VectorStore.find_by_hash()` 命中则直接复用已有 `doc_id`，返回 `status="exists"`，不再重复入库。
3. **多轮上下文打通**：`RAGPipeline.answer()/stream_answer()` 新增 `history` 参数，`_build_messages()` 将历史消息插入 system 之后、当前问题之前，真正送入 LLM。

**验证结果**（真实链路实测）

- 重复上传第二次返回 `{"status": "exists", "doc_id": <同一 id>}`，文档列表保持 1 条 ✓
- 删除不存在文档 → **404** ✓
- 多轮追问 `conversation_id` 保持一致、历史进入消息体 ✓

---

### 2. 交互体验流畅度 —— 优化前 B- → 优化后 A-

**优化前问题**

- 上传失败时前端只显示"上传成功"式文案，用户不知道错在哪；
- 流式过程中后端异常时，前端静默卡住，无任何提示；
- 中文输入法按回车会误发送（组合输入未结束）；
- 引用卡片缺少数值判断，`score` 缺失时渲染 `undefined`。

**改进方案与实施步骤**

1. 前端 `showUploadStatus(text, isError)`：区分成功/失败样式（`.status.error`），并根据后端返回展示"已存在（去重）"。
2. 流式循环识别 `event.type === "error"`，渲染 `.message.error` 气泡，不再静默。
3. 回车发送增加 `!e.isComposing` 判断，兼容中文输入法；并加 `sending` 锁防重复提交。
4. `renderCitations()` 增加 `typeof c.score === "number"` 守卫。
5. 流式消息追加 `.streaming` 类，CSS 绘制闪烁光标，结束移除。

**验证结果**：上传失败有明确红色提示；流式中断显示错误气泡；中文输入回车不误发；引用渲染无 `undefined`。

---

### 3. 响应准确性 —— 优化前 B → 优化后 A-

**优化前问题**

- 空知识库提问仍调用 LLM，模型无资料可依，易产生幻觉；
- 引用缺少稳定结构，前端渲染依赖后端字段拼写。

**改进方案与实施步骤**

1. **空检索短路**：`RAGPipeline` 中检索结果为空时直接返回常量 `NO_CONTEXT_ANSWER`（"知识库中暂时没有检索到与问题相关的内容……"），**不调用 LLM**，从机制上杜绝空库幻觉，同时节省一次模型调用。
2. **Prompt 强约束**：system prompt 明确要求"仅依据提供的参考内容回答，资料中没有就说明无法回答"，并将检索片段按 `[来源 i: 文档名]` 注入。
3. **引用结构标准化**：统一返回 `{doc_name, chunk_text, score}`，`score = 1 - distance`。

**验证结果**（真实检索，5 道可答题）

| 指标 | 结果 |
|---|---|
| 可答题检索命中率（hit@4） | **5/5 = 100%** |
| 超纲题（"北京天气""写快排"） | 不做 LLM 臆答，交由 prompt 约束拒答 |
| 引用条数 | 与 top_k 一致，字段完整 |

> **诚实的局限**：实测发现超纲问题"请用 Python 写一个快速排序函数"仍拿到 **0.499** 的相似度分数（近可答题的 0.5 量级）。这说明**纯向量相似度对"域外但表述相近"的问题区分力有限**，单靠阈值无法可靠区分「相关」与「不相关」。本项目通过 prompt 约束 + 引用溯源缓解，但根因需要在可扩展性章节的 Rerank 方案中解决。

---

### 4. 代码结构合理性 —— 优化前 B+ → 优化后 A

**优化前问题**

- 各服务自行 `try/except` 拼 HTTP 状态码，错误语义分散；
- 服务单例在模块顶层实例化，测试无法注入替身；
- 存在未使用的依赖声明（`markdown`、`tiktoken`），以及**已 import 却未声明的关键依赖** `sentence-transformers`。

**改进方案与实施步骤**

1. **统一领域异常层** `app/core/errors.py`：`AppError` 基类携带 `status_code`，子类 `UnsupportedFileTypeError(400)`、`DocumentParseError(400)`、`FileTooLargeError(413)`、`DocumentNotFoundError(404)`、`LLMUnavailableError(503)` 等；接口层不再关心状态码。
2. **依赖注入**：`get_chat_service()` / `get_ingestion_service()` 改为惰性 getter + `Depends`，构造函数支持传入 `store`/`rag` 替身，测试可 `app.dependency_overrides` 覆盖。
3. **依赖清单修正**：`pyproject.toml` 补 `sentence-transformers>=2.7.0`，删除未用依赖；`ruff` 忽略 FastAPI `Depends` 惯用的 `B008`。
4. 修复 `zip(..., strict=True)`、超长行等 lint 问题，实现 **ruff 零告警**。

**验证结果**：分层依赖方向清晰（api → services → core → utils），领域异常 100% 覆盖，ruff 通过。

---

### 5. 错误处理机制 —— 优化前 C → 优化后 A-

**优化前问题**

- 伪造扩展名文件（如 exe 改名 pdf）解析失败 → **500**（把客户端错误当服务端错误）；
- 上游 LLM 鉴权失败（401）→ 直接 500，且可能泄露上游细节；
- CORS 配置为 `["*"] + allow_credentials=True`，是**浏览器规范禁止的无效组合**；
- 未捕获异常可能把堆栈暴露给前端。

**改进方案与实施步骤**

1. **解析失败归类**：`ingest()` 中解析异常统一转 `DocumentParseError`（400），并配合 `_safe_unlink()`（Windows 下 PyMuPDF 解析失败时仍持有文件句柄，`unlink` 会抛 `PermissionError`——必须吞掉，否则 400 会变 500）。
2. **上游错误翻译** `_wrap_upstream_error()`：`AuthenticationError → "鉴权失败"`、`APITimeoutError / APIConnectionError / APIStatusError` 全部映射为 `LLMUnavailableError`（**503 而非 500**）。
3. **全局异常处理器**：注册 `AppError` 处理器（按 `exc.status_code` 返回），并注册兜底 `Exception` 处理器返回 500 且**不泄露堆栈**。
4. **CORS 修正**：改为读取 `settings.cors_origins`，并设 `allow_credentials=False`。
5. **流式错误可见**：SSE 流内部异常时，yield 一条 `{"type":"error","message":...}` 事件，前端可展示。

**验证结果**（真实 HTTP 实测）

| 场景 | 期望 | 实测 |
|---|---|---|
| 删除不存在文档 | 404 | **404** ✓ |
| 上传 .exe | 400 | **400** ✓ |
| 上传损坏 PDF | 400 | **400** ✓ |
| 空问题（纯空格） | 422 | **422** ✓ |
| 无效 API Key 提问 | 503 | **503** ✓（提示"大模型鉴权失败，请检查 DEEPSEEK_API_KEY 是否有效"） |

---

### 6. 性能表现 —— 优化后 A-（详见[性能评估文档](./performance-report.md)）

**优化前问题**：解析、分块、嵌入、检索等 CPU/IO 阻塞调用直接写在 `async` 函数里，会**阻塞事件循环**，并发时相互拖累。

**改进方案与实施步骤**：所有阻塞调用统一经 `asyncio.to_thread(...)` 卸载到线程池（解析 `load_document`、`_chunk`、`store.add`、`store.search`、`store.find_by_hash`、`store.delete`），保持事件循环不被占用。

**验证结果**（真实栈实测）

| 指标 | 实测值 |
|---|---|
| 首次入库耗时（含模型预热） | 1036.9 ms |
| 检索延迟（中位/最小/最大） | **10.7 / 10.5 / 12.4 ms** |
| 批量嵌入 20 段 | 50.3 ms（≈2.5 ms/段） |
| 流式首字时延（桩 LLM） | 14.0 ms |
| 3 路并发问答总耗时 | 29.8 ms（验证非阻塞） |

---

### 7. 可扩展性 —— 优化前 B → 优化后 A-

**优化前问题**：单例硬编码、会话历史无上限增长（内存泄漏风险）、配置项分散。

**改进方案与实施步骤**

1. **会话内存有界**：`ChatService` 改用 `OrderedDict` + LRU 淘汰（上限 `_max_conversations=200`），防止长期运行内存无限增长。
2. **历史裁剪**：`_trim_history()` 依据 `settings.max_history_turns`（默认 6 轮）裁剪，控制 prompt 长度与成本。
3. **配置集中**：`config.py` 统一暴露 `llm_timeout`、`llm_max_retries`、`cors_origins`、`max_upload_size_mb`、`max_history_turns`、`upload_dir` 等，并派生 `llm_configured` / `max_upload_bytes` / `upload_path`。
4. **可替换存储**：`VectorStore` 通过构造函数注入，未来可平滑替换为 Milvus；`RAGPipeline`、`ChatService`、`IngestionService` 均支持注入替身。

**验证结果**：会话数量受控、历史可裁剪、组件可替换，演进路径（Rerank / 混合检索 / Docker）清晰。

---

## 三、缺陷清单与修复状态汇总

| 编号 | 缺陷 | 严重度 | 状态 |
|---|---|---|---|
| A | 删除不存在文档返回 200 | 中 | 已修复（404） |
| B | 伪造扩展名文件触发 500 | 中 | 已修复（400） |
| C | 重复上传无去重 | 中 | 已修复（SHA-256 去重） |
| D | 多轮上下文失效（死代码） | 高 | 已修复 |
| E | async 中阻塞调用 | 中 | 已修复（to_thread） |
| F | 空库仍调用 LLM 易幻觉 | 高 | 已修复（短路） |
| G | 无统一异常处理 | 中 | 已修复（异常层+处理器） |
| H | CORS 无效组合 | 低 | 已修复 |
| I | 流式异常静默 | 中 | 已修复（error 事件） |
| J | 会话内存无上限 | 中 | 已修复（LRU） |
| K | 缺失依赖声明 `sentence-transformers` | 高 | 已修复 |
| L | Windows 清理临时文件 PermissionError | 中 | 已修复（_safe_unlink） |

---

## 四、遗留事项与后续建议（明确不在本次范围）

1. **LLM 全链路复测已完成**：更新有效 API Key 后，已完成真实 DeepSeek 全链路复测——4 份文档、27 题评估集下作答准确率 **18/18**、超纲拒答率 **6/6**，真实首字时延中位 **510 ms**、问答中位延迟 **841 ms**（详见[测试报告](./test-report.md) 4.5–4.6）。
2. **Rerank / 混合检索**：针对 3. 节暴露的"域外高相似度"问题，后续可引入 `bge-reranker` 重排或 BM25 混合检索提升区分度（用户明确本次不做）。
3. **Docker 部署**：用户明确本次不做。
4. **LLM 层覆盖率**：`llm.py` 覆盖率 61%，因真实上游调用无法在无有效 Key 下测试；错误翻译分支已用 mock 覆盖。

---

## 五、交付物清单

| 交付物 | 路径 |
|---|---|
| 优化后系统代码 | `app/`、`frontend/`、`tests/` |
| 评估与优化报告（本文件） | [docs/evaluation-report.md](./evaluation-report.md) |
| 功能测试报告 | [docs/test-report.md](./test-report.md) |
| 性能评估文档 | [docs/performance-report.md](./performance-report.md) |
