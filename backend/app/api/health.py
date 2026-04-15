"""Health check endpoints."""
from fastapi import APIRouter
from app.core.vector_store import get_qdrant, get_es

router = APIRouter()


async def _check_dependencies() -> dict:
    status = {"status": "ok", "services": {}}

    # Qdrant
    try:
        qdrant = get_qdrant()
        if qdrant is None:
            raise RuntimeError("not initialised")
        await qdrant.get_collections()
        status["services"]["qdrant"] = "ok"
    except Exception as e:
        status["services"]["qdrant"] = f"error: {e}"
        status["status"] = "degraded"

    # Elasticsearch
    try:
        es = get_es()
        if es is None:
            raise RuntimeError("not initialised")
        await es.ping()
        status["services"]["elasticsearch"] = "ok"
    except Exception as e:
        status["services"]["elasticsearch"] = f"error: {e}"
        status["status"] = "degraded"

    return status


@router.get("/health/live", summary="Liveness probe")
async def health_live():
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe")
async def health_ready():
    return await _check_dependencies()


@router.get("/health", summary="Service health check")
async def health():
    return await _check_dependencies()
