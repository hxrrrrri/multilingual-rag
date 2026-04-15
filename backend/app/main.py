"""
Multilingual RAG — FastAPI application entry point.
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.api import documents, query, feedback, health
from app.core.config import settings
from app.core.database import init_db
from app.core.vector_store import init_vector_stores
from app.core.metrics import REQUEST_COUNT, REQUEST_LATENCY


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting Multilingual RAG API…")
    env = settings.APP_ENV.lower()
    strict_startup = env == "production"

    try:
        await init_db()
    except Exception as exc:
        if strict_startup:
            raise
        logger.warning(f"Database init skipped in {env}: {exc}")

    if env not in {"test", "ci"}:
        try:
            await init_vector_stores()
        except Exception as exc:
            if strict_startup:
                raise
            logger.warning(f"Vector store init skipped in {env}: {exc}")
    else:
        logger.info("Skipping vector store init in test/ci environment")

    logger.info("Startup completed.")
    yield
    logger.info("Shutting down…")


app = FastAPI(
    title="Multilingual Document Intelligence API",
    description="RAG pipeline for English / Hindi / Malayalam documents",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Prometheus middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint, request.method, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(endpoint).observe(latency)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(query.router,     prefix="/api/v1",           tags=["Query"])
app.include_router(feedback.router,  prefix="/api/v1/feedback",  tags=["Feedback"])
app.include_router(health.router,    prefix="/api/v1",           tags=["Health"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Multilingual RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "metrics": "/metrics",
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
