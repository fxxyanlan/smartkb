# syntax=docker/dockerfile:1
#
# SmartKB 一键部署镜像（多阶段构建）
#
# 设计要点：
#   1. 构建阶段用 CPU 版 torch，避免拉取体积巨大的 CUDA 依赖；
#   2. 模型在构建期下载进镜像 → 容器起来即可离线运行（真正的"一键"）；
#   3. 国内网络可用 --build-arg HF_ENDPOINT=https://hf-mirror.com 走镜像加速；
#   4. Rerank 模型体积较大（约 1GB），默认一并下载，可用 INSTALL_RERANK_MODEL=false 关闭；
#   5. 运行阶段以非 root 用户启动，数据目录用卷持久化。

# ---------- 构建阶段：装依赖 + 预下载模型 ----------
FROM python:3.11-slim AS builder

ARG HF_ENDPOINT=https://huggingface.co
ARG EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
ARG RERANK_MODEL_NAME=BAAI/bge-reranker-base
ARG INSTALL_RERANK_MODEL=true

ENV HF_ENDPOINT=${HF_ENDPOINT} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app

# 先装 CPU 版 torch（后续安装 sentence-transformers 时即视为已满足），再装项目本体
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install .

# 预下载模型到 /models，供运行阶段直接使用
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${EMBEDDING_MODEL_NAME}', local_dir='/models/embedding')" \
 && if [ "${INSTALL_RERANK_MODEL}" = "true" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('${RERANK_MODEL_NAME}', local_dir='/models/rerank')"; \
    fi

# ---------- 运行阶段：只带运行所需内容 ----------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 模型直接指向镜像内置目录
    EMBEDDING_MODEL=/models/embedding \
    RERANK_MODEL=/models/rerank \
    RERANK_ENABLED=true \
    RERANK_DEVICE=cpu \
    # 数据目录挂到卷上，容器重建不丢数据
    CHROMA_PERSIST_DIR=/data/vector_store \
    UPLOAD_DIR=/data/documents \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models

WORKDIR /app
COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY pyproject.toml README.md ./

RUN mkdir -p /data/documents /data/vector_store \
 && useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app /data /models
USER appuser

EXPOSE 8000

# 容器自愈：探针打到存活接口，失败则标记 unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-8000}"]
