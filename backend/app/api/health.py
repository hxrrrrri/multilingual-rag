"""Health check endpoints."""
from fastapi import APIRouter
from app.core.vector_store import get_qdrant, get_es

router = APIRouter()


@router.get("/health", summary="Service health check")
async def health():
    status = {"status": "ok", "services": {}}

    # Qdrant
    try:
        qdrant = get_qdrant()
        await qdrant.get_collections()
        status["services"]["qdrant"] = "ok"
    except Exception as e:
        status["services"]["qdrant"] = f"error: {e}"
        status["status"] = "degraded"

    # Elasticsearch
    try:
        es = get_es()
        await es.ping()
        status["services"]["elasticsearch"] = "ok"
    except Exception as e:
        status["services"]["elasticsearch"] = f"error: {e}"
        status["status"] = "degraded"

    return status
