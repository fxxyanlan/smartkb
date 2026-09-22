"""FastAPI 应用入口。

启动应用：
    uvicorn app.main:app --reload

架构要点：
    - 路由全部挂在 /api/v1 前缀下，便于未来版本切换
    - 前端静态文件挂载在 /（开发期）
    - lifespan 钩子做启动/关闭时的初始化与清理
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import chat, documents, health
from app.config import get_settings
from app.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动初始化 / 关闭清理。"""
    settings = get_settings()
    setup_logging(settings.log_level)
    logging.info("AskDocs 启动 | host=%s port=%s", settings.app_host, settings.app_port)
    yield
    logging.info("AskDocs 关闭")


def create_app() -> FastAPI:
    """工厂函数：方便测试时创建独立实例。"""
    settings = get_settings()

    app = FastAPI(
        title="AskDocs API",
        description="AI 知识库问答助手",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS（开发期放行；生产应限制具体 origin）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(documents.router, prefix="/api/v1", tags=["documents"])

    # 前端静态文件（开发期直接挂载；生产建议用 nginx）
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app


app = create_app()