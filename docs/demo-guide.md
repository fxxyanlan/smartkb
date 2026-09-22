# AskDocs 演示视频脚本 & 简历写法指南

> 项目已完成。这份文档教你最后两件事：录一个 60 秒演示视频、把项目写进简历。

---

## 一、60 秒演示视频脚本

录屏工具推荐：OBS Studio（免费）或 Windows 自带的 Win+G 录制。

**录制前准备（很重要）：**

1. 启动服务：`uvicorn app.main:app --reload`
2. 准备一份 2-3 页的 PDF（推荐用自己的简历，面试官代入感最强）
3. 清空旧文档：`curl -s http://localhost:8000/api/v1/documents` 查看列表，删掉测试遗留的
4. 浏览器强刷（Ctrl+Shift+R）确保最新前端

**脚本分镜（60 秒）：**

| 时间 | 画面 | 旁白（可后期配音或不配） |
|---|---|---|
| 0-8s | 打开 http://localhost:8000，展示空白聊天页 | "这是我自己开发的 RAG 知识库问答系统 AskDocs" |
| 8-20s | 点击上传，选 PDF，等待"已入库" | "上传 PDF 或 Word 文档后，系统自动解析、分块、向量化入库" |
| 20-38s | 输入一个只有文档里才有答案的问题，答案逐字流出 | "提问后系统做语义检索，把最相关的片段喂给大模型，答案基于文档生成、逐字流式返回" |
| 38-50s | 镜头停留在答案下方的引用卡片上 | "每个回答附带来源引用和相似度分数，可溯源、防幻觉" |
| 50-60s | 不刷新页面，追问一个相关跟进问题 | "支持多轮对话，模型记得上下文" |

**录完导出 MP4**，传到 B 站（设为"自见"或公开）或存网盘，简历里放链接。

---

## 二、简历怎么写

### 项目栏模板（直接抄，按实际微调）

```
AskDocs — AI 知识库问答系统（个人项目，独立开发）        2026.09
技术栈：Python / FastAPI / DeepSeek / bge-small-zh / Chroma / SSE / 原生JS

- 独立开发基于 RAG 的文档问答系统：支持 PDF/Word/Markdown 多格式上传，
  经解析→分块→向量化→入库全流程，实现毫秒级语义检索
- 基于 bge-small-zh 本地 Embedding + Chroma 向量库实现余弦相似度检索，
  大模型仅基于检索内容生成，回答附带来源引用，有效抑制幻觉
- 采用 SSE 协议实现流式输出（text/event-stream），前端打字机逐字呈现；
  自行处理 chunk 边界粘包与 UTF-8 多字节中文截断问题
- 实现 conversation_id 会话管理支持多轮对话；前端做 XSS 转义防护
- 工程规范：分层架构（api/core/services）、.env 密钥隔离、Git 版本管理
```

### 面试高频问题预演（必背）

1. **RAG 流程是什么？** 上传→解析→分块→Embedding→入库；提问→向量化→检索 top-k→拼 Prompt→LLM 生成→返回引用。
2. **为什么用余弦相似度而不是欧氏距离？** 文本向量的模长无意义，方向才代表语义；归一化后两者等价但余弦更稳。
3. **流式输出怎么实现的？** 后端 async generator + StreamingResponse（SSE 格式）；前端 getReader() 循环读取，TextDecoder 的 stream:true 处理中文截断。
4. **怎么防幻觉？** 检索内容注入 system prompt 限定"仅根据以下资料回答"；回答附引用可人工核验。
5. **如果检索结果不准怎么办？** （加分题）可加 Rerank 重排序、混合检索（BM25+向量）、查询改写。

---

## 三、可选的下一步（拉开差距用）

| 方向 | 工作量 | 收益 |
|---|---|---|
| Dockerfile + docker-compose | 2h | 简历写"容器化部署"，面试官可一键跑 |
| Rerank 重排序（bge-reranker） | 4h | 检索精度提升，RAG 岗位高频考点 |
| 评测脚本（命中率/引用准确率） | 4h | "可量化优化"是工程能力的证明 |
| 部署上线（Railway/云函数） | 2h | 简历放可访问链接 > 只有 GitHub |

## 四、上传 GitHub 步骤

```bash
cd askdocs
git remote add origin https://github.com/<你的用户名>/askdocs.git
git push -u origin main
```

推送前已确认：`.env`（密钥）、`data/models/`（91MB 模型）、`.venv` 均被 .gitignore 忽略，不会泄露。
