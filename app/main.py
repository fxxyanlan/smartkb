"""FastAPI 应用入口。

启动应用：
    uvicorn app.main:app --reload

架构要点：
    - 路由全部挂在 /api/v1 前缀下，便于未来版本切换
    - 前端静态文件挂载在 /（开发期）
    - lifespan 钩子做启动/关闭时的初始化与清理
    - AppError → HTTP 状态码的统一映射在这里注册
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, documents, health
from app.config import get_settings
from app.core.errors import AppError
from app.utils.logging import setup_logging

logger = logging.getLogger("askdocs")

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动初始化 / 关闭清理。"""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("AskDocs 启动 | host=%s port=%s", settings.app_host, settings.app_port)
    if not settings.llm_configured:
        logger.warning("未配置 DEEPSEEK_API_KEY，问答功能将不可用")
    yield
    logger.info("AskDocs 关闭")


def create_app() -> FastAPI:
    """工厂函数：方便测试时创建独立实例。"""
    settings = get_settings()

    app = FastAPI(
        title="AskDocs API",
        description="AI 知识库问答助手",
        version=APP_VERSION,
        lifespan=lifespan,
    )

    # CORS：来源走配置；未使用 Cookie，因此不开启 credentials（"*"+credentials 属非法组合）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 业务异常 → 统一 JSON（不泄露堆栈）
    @app.exception_handler(AppError)
    async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("业务异常 | path=%s status=%d msg=%s",
                       request.url.path, exc.status_code, exc.message)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    # 兜底：未预期异常统一返回 500，不把内部细节透给客户端
    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常 | path=%s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})

    # 路由
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(documents.router, prefix="/api/v1", tags=["documents"])

    # 前端静态文件（开发期直接挂载；生产建议用 nginx）
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

    return app


app = create_app()
