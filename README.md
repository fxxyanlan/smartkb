# SmartKB · 企业知识库智能问答系统

> 把企业散落的 PDF、Word、Markdown 文档变成"可对话的知识库"——上传即入库，提问即作答，答案带出处。

---

## 一、项目简介与价值定位

企业内部的知识往往散落在几十上百份文档里：新人手册、产品说明、API 文档、运维规范……找一句话要翻半天。SmartKB 解决的就是这个问题。

上传文档后，系统自动完成**解析 → 分块 → 向量化 → 入库**；用户用自然语言提问时，系统先做**语义检索**召回最相关的原文片段，再交给大模型**基于原文生成答案**，并逐字流式返回。每个回答都附带**来源引用**，可人工核验，从机制上抑制大模型"一本正经胡说"。

**核心价值：**

- **可信**——答案只依据检索到的原文生成，附引用可溯源；检索为空时直接拒答，不编造。
- **开箱即用**——单进程启动，无需外部数据库；Embedding 模型本地运行，数据不出内网。
- **易扩展**——分层架构 + 依赖注入，模型、向量库、生成逻辑均可替换。

**适用场景：** 企业内部知识库问答、产品/技术文档助手、规章制度查询、客服知识支撑等。

---

## 二、核心功能与特性

| 能力 | 说明 |
|---|---|
| 多格式解析 | PDF / Word(docx) / Markdown / TXT，自动路由到对应解析器 |
| 智能分块 | 按 500 字切分、相邻块重叠 50 字，避免关键句被切断 |
| 语义检索 | 基于向量的余弦相似度召回 top-4 相关片段 |
| 可信生成 | 检索内容注入 Prompt，模型仅依据资料作答，答案标注来源 |
| 流式输出 | 基于 SSE（`text/event-stream`），前端打字机逐字呈现 |
| 多轮对话 | `conversation_id` 维护上下文，支持追问 |
| 内容去重 | SHA-256 内容指纹，重复上传自动识别，不重复入库 |
| 空库兜底 | 知识库无相关内容时不调用模型，直接返回提示话术 |
| 健壮性 | 统一异常层（4xx/5xx 语义正确）、请求校验、上传大小限制 |
| 可观测 | 存活探针 `/health` + 就绪探针 `/health/ready` |

---

## 三、技术架构与实现原理

### 技术栈

| 层次 | 选型 |
|---|---|
| Web 框架 | FastAPI（异步 + 自动 OpenAPI 文档） |
| 大模型 | DeepSeek `deepseek-chat`（OpenAI 兼容协议） |
| Embedding | bge-small-zh-v1.5（512 维，本地离线推理） |
| 向量库 | Chroma（持久化，余弦相似度） |
| 流式协议 | SSE（Server-Sent Events） |
| 前端 | 原生 HTML / CSS / JavaScript（单页，零构建） |
| 文档解析 | PyMuPDF / python-docx / langchain-text-splitters |
| 配置管理 | pydantic-settings（`.env` 驱动，全局单例） |

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
问题 → 向量化 → 检索 top-4 → 命中为空？
                              ├─ 是 → 直接返回拒答话术（不调用模型）
                              └─ 否 → 拼装 system prompt(注入原文) + 多轮历史
                                     → 调用 LLM 流式生成 → SSE 逐字推送
                                     → 流末返回引用来源 + conversation_id
```

**关键实现点：**

- **防幻觉**：`core/prompts.py` 中的 system prompt 明确限定"仅根据以下资料回答"，检索为空时短路，不产生无依据输出。
- **不阻塞事件循环**：解析、分块、嵌入、检索等 CPU/IO 密集操作通过 `asyncio.to_thread` 卸载到线程池。
- **流式错误兜底**：SSE 流一旦开始无法改状态码，异常以 `{"type":"error"}` 事件回传，前端据此提示。
- **内存保护**：会话历史用 `OrderedDict` + LRU 上限（200 个会话），历史轮数上限 6。

---

## 四、快速开始

### 环境要求

- **Python ≥ 3.10**
- 约 500 MB 磁盘空间（含本地 Embedding 模型）
- 一个 DeepSeek API Key（[获取地址](https://platform.deepseek.com/)）

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

### API 速览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/documents/upload` | 上传文档并入库 |
| GET | `/api/v1/documents` | 列出已入库文档 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档及其向量 |
| POST | `/api/v1/chat` | 一次性问答 |
| POST | `/api/v1/chat/stream` | 流式问答（SSE） |
| GET | `/api/v1/health` | 存活探针 |
| GET | `/api/v1/health/ready` | 就绪探针（向量库 + LLM 配置） |

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
pytest                    # 运行测试（50 用例）
ruff check .              # 代码规范检查
```

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
| `app/core/rag.py` | RAG 流水线：检索 → 拼装 → 生成 / 流式生成 |
| `app/core/llm.py` | LLM 客户端：普通 + 流式，上游错误归一化 |
| `app/core/embeddings.py` | Embedding 模型加载与批量向量化 |
| `app/core/vector_store.py` | 向量库读写：add / search / delete / 去重查询 |
| `app/core/prompts.py` | Prompt 模板集中管理 |
| `app/core/errors.py` | 领域异常定义，映射 HTTP 状态码 |
| `app/utils/loaders.py` | 多格式文档解析器（按扩展名路由） |
| `app/utils/logging.py` | 统一日志格式 |
| `frontend/` | 单页前端：上传、聊天、引用渲染、XSS 转义 |
| `tests/` | pytest 用例（含内存向量库 / 桩模型） |

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
引用是"防幻觉 + 可核验"的关键：答案限定只能依据检索到的原文生成，读者可点击引用回看依据。

**Q6：如何更换大模型或向量库？**
模型、向量库均通过构造函数注入。换 LLM 改 `.env`（`DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL`）；换向量库可实现同样的接口后注入，业务层无感。

**Q7：如何调优检索效果？**
`.env` 中 `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` 三个参数控制切分粒度与召回数量，可结合评估集做对照实验。

**Q8：端口 8000 被占用？**
`uvicorn app.main:app --reload --port 8001`，或在 `.env` 中修改 `APP_PORT`。

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
- [x] 测试补全（pytest，50 用例，覆盖率 92%）
- [ ] Docker 一键部署
- [ ] 重排序（Rerank）与混合检索（BM25 + 向量）

---

## License

MIT
