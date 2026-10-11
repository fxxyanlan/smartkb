# SmartKB · 企业知识库智能问答系统

> 把企业散落的 PDF、Word、Markdown 文档变成"可对话的知识库"——上传即入库，提问即作答，答案带出处。

---

## 一、项目简介与价值定位

企业内部的知识往往散落在几十上百份文档里：新人手册、产品说明、API 文档、运维规范……找一句话要翻半天。SmartKB 解决的就是这个问题。

上传文档后，系统自动完成**解析 → 分块 → 向量化 → 入库**；用户用自然语言提问时，系统先做**语义检索**召回最相关的原文片段，再交给大模型**基于原文生成答案**，并逐字流式返回。每个回答都附带**来源引用**，可人工核验，从机制上抑制大模型"一本正经胡说"。

**核心价值：**

- **可信**——答案只依据检索到的原文生成，附引用可溯源；检索为空时直接拒答，不编造。
- **开箱即用**——`docker compose up -d --build` 一条命令拉起（模型内置镜像、数据落卷）；也可以单进程直接跑，无需外部数据库；Embedding 模型本地运行，数据不出内网。
- **检索可靠**——向量 + BM25 混合召回，可选交叉编码器精排；效果由 50 题离线评估集与 CI 基线门禁守护，指标回退即失败。
- **易扩展**——分层架构 + 依赖注入，模型、向量库、生成逻辑均可替换。

**适用场景：** 企业内部知识库问答、产品/技术文档助手、规章制度查询、客服知识支撑等。

---

## 二、核心功能与特性

| 能力 | 说明 |
|---|---|
| 多格式解析 | PDF / Word(docx) / Markdown / TXT，自动路由到对应解析器 |
| 智能分块 | 按 500 字切分、相邻块重叠 50 字，避免关键句被切断 |
| 语义检索 | 基于向量的余弦相似度召回相关片段 |
| 混合检索 | 向量语义 + BM25 关键词双通道，RRF 融合（自主实现，零新增依赖） |
| 重排序 | 交叉编码器对候选精排（两段式：粗召回 N → 精排取 top_k），模型缺失自动降级 |
| 检索缓存 | LRU + TTL 结果缓存，按向量库版本号自动失效（命中免去嵌入与检索全流程） |
| 可信生成 | 检索内容注入 Prompt，模型仅依据资料作答，答案标注来源 |
| 引用溯源 | 引用卡片可展开回看原文片段，并高亮问题中的命中词 |
| 流式输出 | 基于 SSE（`text/event-stream`），前端打字机逐字呈现 |
| 流式健壮性 | 首字超时自动重试、未输出前失败可安全重试，错误以事件收敛回传 |
| 多轮对话 | `conversation_id` 维护上下文，支持追问 |
| 内容去重 | SHA-256 内容指纹，重复上传自动识别，不重复入库 |
| 空库兜底 | 知识库无相关内容时不调用模型，直接返回提示话术 |
| 健壮性 | 统一异常层（4xx/5xx 语义正确）、请求校验、上传大小限制 |
| 可观测 | 存活探针 `/health` + 就绪探针 `/health/ready`（含检索缓存命中率） |
| 一键部署 | `docker compose up -d --build` 即起服务，模型内置镜像、数据落卷持久化 |
| 质量门禁 | CI 跑 ruff + pytest（覆盖率门槛）+ 离线评估基线校验 + 镜像构建冒烟 |

---

## 三、技术架构与实现原理

### 技术栈

| 层次 | 选型 |
|---|---|
| Web 框架 | FastAPI（异步 + 自动 OpenAPI 文档） |
| 大模型 | DeepSeek `deepseek-chat`（OpenAI 兼容协议） |
| Embedding | bge-small-zh-v1.5（512 维，本地离线推理） |
| 向量库 | Chroma（持久化，余弦相似度） |
| 检索策略 | 向量检索 + BM25 关键词检索，RRF 倒数排名融合（`app/core/bm25.py`、`retriever.py`） |
| 重排序 | 交叉编码器精排（sentence-transformers `CrossEncoder`，bge-reranker-base，可关可降级） |
| 流式协议 | SSE（Server-Sent Events） |
| 前端 | 原生 HTML / CSS / JavaScript（单页，零构建） |
| 文档解析 | PyMuPDF / python-docx / langchain-text-splitters |
| 配置管理 | pydantic-settings（`.env` 驱动，全局单例） |
| 部署 | Docker 多阶段构建 + docker compose（模型内置、命名卷持久化） |

### 分层架构

```
api/        HTTP 路由 + 数据契约（Pydantic 模型）
  │
services/   业务编排（ingestion 入库 / chat 问答）
  │
core/       领域逻辑（LLM / Embedding / RAG / 向量库 / 异常）
  │
utils/      通用工具（日志 / 文档解析器）
```

依赖方向单向向下：`api → services → core → utils`，上层依赖下层，绝不反向。层与层之间通过构造函数/依赖注入衔接，便于替换与测试。

### 两条核心数据流

**① 文档入库**

```
上传文件 → 校验(类型/大小) → SHA-256 去重 → 解析文本 → 分块(500/50)
        → 批量向量化(512维) → 写入 Chroma（含 doc_id / doc_name 元数据）
```

**② 问答生成**

```
问题 → 缓存查找 ─ 命中 → 直接返回 top-4
              │
              └ 未命中 → 向量化 ┐
                                ├─ 向量通道(语义) ┐
问题 → 分词                     ┘                ├─ RRF 融合 → 候选池 ─(开启时)→ 交叉编码器精排 → top-4
                     └─ BM25 通道(关键词) ──────┘                                    ↓ 命中为空？
                                                                        ├─ 是 → 直接返回拒答话术（不调用模型）
                                                                        └─ 否 → 拼装 prompt + 多轮历史
                                                                               → LLM 流式生成（首字超时，未输出前可重试）
                                                                               → SSE 逐字推送
                                                                               → 流末返回引用来源 + conversation_id
```

**关键实现点：**

- **防幻觉**：`core/prompts.py` 中的 system prompt 明确限定"仅根据以下资料回答"，检索为空时短路，不产生无依据输出。
- **混合检索**：向量检索擅长语义相近，但对专有名词、编号、缩写（如 `rrf_k`、`429`、`P0`）不敏感；因此并接一路 BM25 关键词检索，用 RRF（倒数排名融合）合并——只用名次、不依赖量纲，无需调权。BM25 为自主实现（含中文单字+双字切词），不引入 jieba 等外部依赖。
- **两段式重排**：向量（双塔）与 BM25 都只能保证"相关片段落在候选池里"，不保证排在最前；交叉编码器把 `(问题, 片段)` 成对送进模型、直接建模交互，排序更准但无法预建索引，所以先粗召回 `rerank_candidates`（默认 10）个候选再精排取 top_k。模型加载或打分失败一律**降级为不重排**，绝不因模型问题拖垮链路。
- **检索缓存**：`core/cache.py` 自实现线程安全的 LRU + TTL 缓存，键为 `(问题, top_k, 向量库版本号)`——文档增删后版本号变化，旧结果自动失配，无需遍历清理。缓存套在整条检索链最外层，命中即全链路免算。
- **不阻塞事件循环**：解析、分块、嵌入、检索等 CPU/IO 密集操作通过 `asyncio.to_thread` 卸载到线程池。
- **流式健壮性**：对**首个 token**施加超时（`llm_first_token_timeout`），超时按 `llm_max_retries` 重试；仅在"一个字都没吐出"时才重试，避免重复输出；一旦已输出则异常上抛，由 API 层转成 `{"type":"error"}` 事件。流结束显式 `aclose()` 防连接泄漏。
- **流式错误兜底**：SSE 流一旦开始无法改状态码，异常以 `{"type":"error"}` 事件回传，前端据此提示。
- **引用高亮**：前端在**原始文本**上按问题词切分匹配、对每段分别转义，既高亮命中词又不引入 XSS。
- **内存保护**：会话历史用 `OrderedDict` + LRU 上限（200 个会话），历史轮数上限 6；检索缓存同样有容量上限（256 条）。

---

## 四、快速开始

### 环境要求

- **Docker（推荐）**：只需 Docker + Docker Compose，见下方「Docker 一键部署」。
- **本地开发**：Python ≥ 3.10、约 500 MB 磁盘空间（含本地 Embedding 模型）。
- 一个 DeepSeek API Key（[获取地址](https://platform.deepseek.com/)）。

### 安装步骤

```bash
# 1. 克隆并进入项目
git clone <your-repo-url> smartkb && cd smartkb

# 2. 创建虚拟环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. 安装依赖（含开发依赖）
pip install -e ".[dev]"

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 启动与使用

```bash
uvicorn app.main:app --reload
```

启动后访问 **http://localhost:8000**：上传一份文档 → 提问 → 观察答案逐字流出与下方引用卡片。交互式 API 文档见 http://localhost:8000/docs。

### Docker 一键部署（推荐）

无需本地 Python 环境，模型在镜像构建期下载，容器起来即可离线运行：

```bash
# 1. 准备密钥（compose 会自动读取同目录的 .env 做变量替换）
cp .env.example .env    # 填入 DEEPSEEK_API_KEY

# 2. 构建并启动
docker compose up -d --build

# 3. 访问 http://localhost:8000
docker compose logs -f      # 查看日志
docker compose down         # 停止（数据保存在命名卷中，不会丢）
```

说明：

- **国内网络**建议先 `export HF_ENDPOINT=https://hf-mirror.com` 再构建，加速模型下载。
- 只想跑通、不需要重排：`INSTALL_RERANK_MODEL=false docker compose up -d --build`（镜像小约 1GB，重排会自动降级）。
- **数据持久化**：向量库与上传文件放在 `smartkb_vectors` / `smartkb_documents` 两个命名卷中，容器重建不影响数据。
- 容器内置 `HEALTHCHECK`，`docker ps` 可直接看到 healthy/unhealthy。

### API 速览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/documents/upload` | 上传文档并入库 |
| GET | `/api/v1/documents` | 列出已入库文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及其向量 |
| POST | `/api/v1/chat` | 一次性问答 |
| POST | `/api/v1/chat/stream` | 流式问答（SSE） |
| GET | `/api/v1/health` | 存活探针 |
| GET | `/api/v1/health/ready` | 就绪探针（向量库 + LLM 配置 + 缓存命中率） |

```bash
# 上传
curl -X POST http://localhost:8000/api/v1/documents/upload -F "file=@doc.pdf"

# 提问
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "这份文档讲了什么？"}'
```

### 质量验证

```bash
pytest                       # 运行测试（85 用例）
ruff check .                 # 代码规范检查
python scripts/evaluate.py   # 检索效果评估（纯向量 vs 混合检索 + 缓存收益）
python scripts/evaluate.py --rerank   # 额外测"混合检索 + 交叉编码器精排"（需本地有重排模型）
```

CI（`.github/workflows/ci.yml`）在每次推送时跑这些检查：ruff + pytest（覆盖率门槛 85%）+ `evaluate.py --check-baseline`（与冻结基线 `eval/baseline.json` 比对，任一检索指标回退即失败）+ 镜像构建冒烟测试。

### 检索效果实测

在 12 份文档 / 50 题评估集下，对比纯向量与混合检索（`eval/dataset.jsonl`，可复现）：

| 指标 | 纯向量 | 混合检索 |
|---|---|---|
| recall@1 | 75.0% | **87.5%** |
| recall@4 | 92.5% | **100.0%** |
| 检索中位延迟 | ≈ 12 ms | ≈ 14 ms |

混合检索修复了 3 道纯向量漏检的题（均为含专有标识的查询，如 `rrf_k`、`200 个会话`、`内测`），无新增漏检；延迟处于同一毫秒量级，收益来自召回提升而非速度提升。

**检索缓存收益**（50 题各重复提问一次，命中率 50%）：

| 指标 | 冷启动（未命中） | 缓存命中 |
|---|---|---|
| 检索中位延迟 | ≈ 15 ms | **≈ 0.002 ms** |

命中时直接返回结果，跳过嵌入与双通道检索；文档增删后版本号变化，缓存自动失效。

**重排（Rerank）的评估口径**：重排需要额外下载约 1GB 的交叉编码器模型，因此默认不参与评估（`eval/report.json` 中如实标记为 `skipped`），需要时用 `python scripts/evaluate.py --rerank` 或 `RERANK_ENABLED=true` 显式开启，报告会记录 `recall@1` / `recall@4` / 中位延迟；模型不可用时标记为 `unavailable`，**绝不用"未重排"的结果冒充"已重排"**。基线门禁只约束确定性的 `vector_only` / `hybrid` 两档。

---

## 五、主要模块说明

| 模块 | 职责 |
|---|---|
| `app/main.py` | 应用工厂、路由注册、前端挂载、全局异常映射 |
| `app/config.py` | `.env` 配置读取，全局 Settings 单例 |
| `app/api/chat.py` | `/chat`、`/chat/stream` 端点（含 SSE 封装） |
| `app/api/documents.py` | 文档上传 / 列表 / 删除端点 |
| `app/api/health.py` | 存活与就绪探针 |
| `app/api/schemas.py` | 请求/响应模型，含输入校验（空问题拦截） |
| `app/services/ingestion.py` | 入库编排：校验 → 去重 → 解析 → 分块 → 向量化 |
| `app/services/chat_service.py` | 问答编排：会话历史管理、调用 RAG |
| `app/core/rag.py` | RAG 流水线：检索 → 拼装 → 生成 / 流式生成（首字超时 + 未输出前重试） |
| `app/core/retriever.py` | 混合检索：向量 + BM25 双通道，RRF 融合，索引按版本号惰性重建；`RerankedRetriever` 精排、`CachedRetriever` 缓存两层装饰器 |
| `app/core/reranker.py` | 重排序：`Reranker` 接口 + `CrossEncoderReranker`（懒加载、失败降级）+ `NoopReranker` 兜底 |
| `app/core/cache.py` | 检索结果缓存：线程安全 LRU + TTL，按向量库版本号失效，暴露命中率统计 |
| `app/core/bm25.py` | BM25 关键词检索（自主实现，含中文单字+双字切词） |
| `app/core/llm.py` | LLM 客户端：普通 + 流式，上游错误归一化 |
| `app/core/embeddings.py` | Embedding 模型加载与批量向量化 |
| `app/core/vector_store.py` | 向量库读写：add / search / delete / 去重查询 |
| `app/core/prompts.py` | Prompt 模板集中管理 |
| `app/core/errors.py` | 领域异常定义，映射 HTTP 状态码 |
| `app/utils/loaders.py` | 多格式文档解析器（按扩展名路由） |
| `app/utils/logging.py` | 统一日志格式 |
| `frontend/` | 单页前端：上传、聊天、引用展开与命中词高亮、XSS 转义 |
| `tests/` | pytest 用例（含内存向量库 / 桩模型，85 用例） |
| `scripts/evaluate.py` | 离线检索评估：纯向量 vs 混合检索对比、缓存收益量化、可选精排对照、`--check-baseline` 门禁 |
| `Dockerfile` / `docker-compose.yml` | 一键部署：多阶段构建、模型内置、命名卷持久化、健康检查 |
| `.github/workflows/ci.yml` | CI 门禁：ruff + pytest 覆盖率 + 评估基线校验 + 镜像构建冒烟 |

---

## 六、常见问题解答（FAQ）

**Q1：没有 DeepSeek API Key 能跑起来吗？**
可以启动，上传、入库、检索链路均正常；但问答会返回 503。启动日志会提示"未配置 DEEPSEEK_API_KEY"。建议配置后使用。

**Q2：支持哪些文档格式？有大小限制吗？**
支持 PDF、Word(docx)、Markdown、TXT，单文件默认上限 20 MB（可在 `.env` 的 `MAX_UPLOAD_SIZE_MB` 调整）。不支持的格式返回 400。

**Q3：首次启动比较慢？**
首次启动需要加载本地 Embedding 模型（bge-small-zh-v1.5），之后会常驻内存。模型默认从 `./data/models/` 本地路径加载，无需联网。

**Q4：上传同一份文档两次会怎样？**
系统按 SHA-256 内容指纹判重，第二次会直接返回 `status: "exists"`，不会重复入库，检索结果也不会重复。

**Q5：为什么回答要带引用？**
引用是"防幻觉 + 可核验"的关键：答案限定只能依据检索到的原文生成，读者可展开引用卡片回看原文片段，命中的关键词还会高亮标注，便于快速定位依据。

**Q6：如何更换大模型或向量库？**
模型、向量库均通过构造函数注入。换 LLM 改 `.env`（`DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL`）；换向量库可实现同样的接口后注入，业务层无感。

**Q7：如何调优检索效果？**
`.env` 中 `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` 三个参数控制切分粒度与召回数量，可结合评估集做对照实验（`python scripts/evaluate.py`）。检索缓存由 `CACHE_ENABLED` / `CACHE_SIZE` / `CACHE_TTL` 控制，缓存命中率可在 `/api/v1/health/ready` 查看。

**Q8：端口 8000 被占用？**
`uvicorn app.main:app --reload --port 8001`，或在 `.env` 中修改 `APP_PORT`。

**Q9：重排序（Rerank）默认为什么是关的？怎么开？**
重排要额外下载约 1GB 的交叉编码器模型，默认关闭可让「clone 即跑」和 CI 保持轻量。开启方式：`.env` 设 `RERANK_ENABLED=true` 并配置 `RERANK_MODEL`（可填 HuggingFace 模型名或本地目录）；Docker 镜像已内置模型并默认开启。**模型缺失或加载失败时会自动降级为不重排**，只记一条日志，不影响服务可用性。

**Q10：Docker 部署时模型下载很慢/失败？**
构建期需要联网下载模型。国内网络建议设置 `HF_ENDPOINT=https://hf-mirror.com` 后再 `docker compose up -d --build`；若只想快速跑通，用 `INSTALL_RERANK_MODEL=false` 跳过重排模型（可省约 1GB）。

---

## 七、贡献指南

欢迎提交 Issue 与 Pull Request。

1. **Fork** 本仓库并创建特性分支：`git checkout -b feat/your-feature`
2. **编码规范**：遵循 `ruff` 配置（line-length 100）；新增函数补充类型提示。
3. **补充测试**：新功能需附 pytest 用例，`pytest` 全绿、`ruff check .` 无告警后再提交。
4. **提交信息**：使用清晰的动词开头（如 `feat: 支持表格解析`、`fix: 修正空库拒答逻辑`）。
5. **发起 PR**：说明改动动机、实现方式与验证结果。

**开发路线图**

- [x] 多格式解析与分块
- [x] 本地 Embedding + Chroma 持久化检索
- [x] 端到端 RAG 流水线（上传 → 检索 → 生成 → 引用）
- [x] SSE 流式输出
- [x] 多轮对话与会话管理
- [x] 内容去重 / 统一异常层 / 健康探针
- [x] 测试补全（pytest，85 用例，覆盖率 93%）
- [x] 混合检索（向量 + BM25，RRF 融合）与 50 题检索评估集
- [x] 检索结果缓存（LRU + TTL + 版本号失效，重复提问延迟 <1ms）
- [x] 流式健壮性（首字超时 + 未输出前重试 + 错误收敛）
- [x] 引用原文展开与命中词高亮
- [x] CI 门禁（ruff + pytest 覆盖率 + 评估基线校验）
- [x] Docker 一键部署（多阶段构建 + 模型内置 + 命名卷持久化）
- [x] 重排序（Rerank，交叉编码器，可关可降级）
- [ ] Agent 工具调用
- [ ] 多租户与权限

---

## License

MIT
