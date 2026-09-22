"""/health 健康检查端点。"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """健康检查：负载均衡器和探针会调用。"""
    return {"status": "ok"}