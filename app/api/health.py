"""/health 健康检查端点。

/health      存活探针：进程活着就返回 200（负载均衡用）。
/health/ready 就绪探针：额外检查向量库与 LLM 配置。
"""
import asyncio

from fastapi import APIRouter

from app.config import get_settings
from app.core.vector_store import get_vector_store

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """存活检查：负载均衡器和容器探针会调用。"""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness_check() -> dict:
    """就绪检查：依赖是否可用（向量库可读、LLM 是否已配置）。"""
    settings = get_settings()
    checks: dict = {"llm_configured": settings.llm_configured}

    try:
        store = get_vector_store()
        checks["vector_store"] = {
            "status": "ok",
            "chunks": await asyncio.to_thread(store.count),
        }
    except Exception as e:  # noqa: BLE001 - 就绪检查需吞掉异常并以状态呈现
        checks["vector_store"] = {"status": "error", "detail": str(e)}

    ready = checks["vector_store"]["status"] == "ok"
    return {"status": "ok" if ready else "degraded", "checks": checks}
