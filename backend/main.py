"""
main.py — FastAPI application entrypoint.

Registers all routers, middleware, Prometheus metrics endpoint,
and startup/shutdown lifecycle hooks.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from loguru import logger

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api import documents, query, feedback, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings.log_level)
    os.makedirs(settings.upload_dir, exist_ok=True)
    logger.info(f"🚀 Multilingual RAG API starting — env={settings.app_env}")
    yield
    logger.info("👋 Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Multilingual Document Intelligence API",
        description=(
            "Production RAG pipeline for English, Hindi, and Malayalam documents. "
            "Hybrid dense+sparse retrieval, LoRA fine-tuned embeddings, DPO-aligned LLM."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://frontend:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(feedback.router)

    # ── Prometheus metrics at /metrics ────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name":    "Multilingual RAG API",
            "version": "1.0.0",
            "docs":    "/docs",
            "health":  "/health",
            "metrics": "/metrics",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
