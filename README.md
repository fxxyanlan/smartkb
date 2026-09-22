# AskDocs — AI 知识库问答助手

> 基于 RAG 的企业级文档问答系统,支持多文档上传、语义检索、可信回答。

## 这是什么

上传你的文档（PDF / Markdown / Word / TXT），AskDocs 会自动解析、分块、向量化存储。用户用自然语言提问，系统检索相关内容，基于检索内容生成答案，并标注引用来源——避免幻觉。

## 核心特性

- 多格式文档解析（PDF / Markdown / Word / TXT）
- 智能分块（支持中英文）
- 语义检索（基于向量相似度）
- 生成式回答 + 来源标注
- 多轮对话，支持上下文追问
- 流式输出，打字机体验

## 技术栈

| 层 | 技术 |
|---|---|
| LLM | DeepSeek（OpenAI 兼容） |
| Embedding | bge-small-zh-v1.5 |
| 向量数据库 | Chroma（开发）/ Milvus（生产） |
| 后端 | FastAPI |
| 前端 | 原生 HTML / JS（单页应用） |
| 文档解析 | PyMuPDF / python-docx / markdown |
| 测试 | pytest |

## 快速开始

```bash
# 1. 进入项目
cd askdocs

# 2. 安装依赖（建议用虚拟环境）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 4. 启动服务
uvicorn app.main:app --reload

# 5. 打开浏览器
# 访问 http://localhost:8000
```

## API 示例

```bash
# 上传文档
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@your-document.pdf"

# 提问
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "这份文档讲了什么？"}'

# 健康检查
curl http://localhost:8000/api/v1/health
```

## 项目结构

```
askdocs/
├── app/                  # 应用代码（分层架构）
│   ├── api/              # HTTP 路由 & 数据契约
│   ├── core/             # 领域逻辑（LLM / Embedding / RAG）
│   ├── services/         # 高层业务编排
│   ├── utils/            # 通用工具
│   ├── data/             # 数据库相关
│   ├── config.py         # 配置管理
│   └── main.py           # FastAPI 入口
├── frontend/             # 前端（单页聊天）
├── tests/                # 测试
├── data/                 # 运行时数据（已 gitignore）
└── docs/                 # 文档
```

## 架构图

```
用户 ──HTTP──▶ FastAPI
                 │
                 ├──▶ Chat Service ──▶ RAG Pipeline
                 │                        │
                 │                        ├──▶ Embedding
                 │                        ├──▶ Vector Store (检索)
                 │                        └──▶ LLM (生成)
                 │
                 └──▶ Ingestion Service ──▶ Document Loader
                                             │
                                             ├──▶ Chunker
                                             └──▶ Embedding → Vector Store
```

## 开发路线图

- [x] 项目骨架与配置
- [x] 文档解析与分块（PDF / Word / Markdown / TXT）
- [x] Embedding 模型加载（bge-small-zh-v1.5，本地离线）
- [x] 向量库读写（Chroma 持久化 + 余弦相似度检索）
- [x] 端到端 RAG 流水线（上传 → 检索 → 生成 → 引用）
- [x] 流式输出（SSE 协议，`text/event-stream`）
- [x] 多轮对话（conversation_id 会话管理）
- [x] 引用来源高亮（前端引用卡片 + XSS 转义）
- [ ] 单元测试补全
- [ ] Docker 部署
- [ ] 重排序（Rerank）与混合检索

## 为什么做这个项目

- **真实业务场景**：企业知识管理是 LLM 落地最广的场景之一
- **覆盖核心技术栈**：RAG / Embedding / 向量库 / LLM API / FastAPI / 前端
- **可扩展性**：可平滑演进到 Agent（增加工具调用）、多模态（图片问答）
- **可演示性强**：简历上有可在线访问的 demo，胜过 10 份自我介绍

## License

MIT