# AskDocs 架构说明

## 分层

```
┌────────────────────────────────────────┐
│  api/         HTTP 路由 + 数据契约       │
├────────────────────────────────────────┤
│  services/    业务编排（ingestion/chat）│
├────────────────────────────────────────┤
│  core/        领域逻辑（LLM/RAG/存储）  │
├────────────────────────────────────────┤
│  utils/       通用工具（日志/加载器）    │
├────────────────────────────────────────┤
│  data/        数据持久化               │
└────────────────────────────────────────┘
```

依赖方向：**上层依赖下层，绝不反过来**。
- `api/` 可以调用 `services/` 和 `core/`
- `services/` 可以调用 `core/` 和 `utils/`
- `core/` 不应调用 `api/` 或 `services/`

## 数据流

### 文档入库

```
UploadFile → ingestion.ingest() → load_document() → chunk() → embed() → vector_store.add()
```

### 问答

```
question → rag.answer() → vector_store.search() → llm.chat() → response
```

## 关键技术决策

| 决策 | 选择 | 理由 |
|---|---|---|
| LLM | DeepSeek | 性价比高、OpenAI 兼容、中文强 |
| Embedding | bge-small-zh | 轻量（~100MB）、中文 SOTA、可本地跑 |
| 向量库 | Chroma（开发） | 零配置、Python 原生、适合 MVP |
| Web 框架 | FastAPI | 异步、自动 OpenAPI 文档、原生类型提示 |
| 前端 | 原生 HTML | 单文件即可运行，降低复杂度 |

## 模块职责速查

| 模块 | 一句话职责 |
|---|---|
| `config.py` | 读 .env、暴露全局 Settings 单例 |
| `main.py` | FastAPI 应用工厂、注册路由、挂前端 |
| `api/chat.py` | `/chat` 端点：问答入口 |
| `api/documents.py` | `/documents/*` 端点：上传/列表/删除 |
| `api/schemas.py` | Pydantic 模型：前后端数据契约 |
| `core/llm.py` | LLM 客户端：普通调用 + 流式调用 |
| `core/embeddings.py` | Embedding 模型：文本转向量 |
| `core/vector_store.py` | 向量库：add / search / delete 三件套 |
| `core/rag.py` | RAG 流水线：检索 + 拼装 + 生成 |
| `core/prompts.py` | Prompt 模板集中管理 |
| `services/ingestion.py` | 文档入库编排：解析→分块→向量化 |
| `services/chat_service.py` | 问答编排：维护会话历史、调 RAG |
| `utils/loaders.py` | 多格式文档解析器 |
| `utils/logging.py` | 统一日志格式 |

## 演进路径

- **MVP**：单文档、单会话、能跑通
- **v1**：多文档、会话持久化（SQLite）、引用高亮、来源跳转
- **v2**：Agent 工具调用（查实时信息、调外部 API）、混合检索（BM25 + 向量）、重排序（Rerank）
- **v3**：多模态（图片问答、表格解析）、用户/权限系统、监控告警
- **v4**：Docker Compose 一键部署、Kubernetes 编排

## 为什么这样分

面试官看代码，30 秒内会判断：
1. **有没有分层**：一个 `main.py` 写 800 行 vs 清晰分层 —— 含金量差一个量级
2. **有没有配置管理**：硬编码 vs 走 `.env` —— 工程素养分水岭
3. **有没有类型提示**：靠注释 vs Pydantic 模型 —— 现代 Python 标杆
4. **有没有测试**：靠 print vs pytest —— 工程化标志
5. **README 写得好不好**：能不能 5 分钟内跑起来 —— 协作能力

本项目按这五点全部覆盖。