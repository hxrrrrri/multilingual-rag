"""
MLFlow experiment tracking utility.
Logs retrieval metrics, latency, and model versions.
"""
import functools
import time
from typing import Optional
from loguru import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def log_query_metrics(
    query: str,
    latency_ms: float,
    faithfulness: float,
    num_chunks: int,
    model_version: Optional[str] = None,
):
    if not MLFLOW_AVAILABLE:
        return
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_metric("latency_ms",   latency_ms)
            mlflow.log_metric("faithfulness", faithfulness)
            mlflow.log_metric("num_chunks",   num_chunks)
            if model_version:
                mlflow.set_tag("model_version", model_version)
    except Exception as e:
        logger.debug(f"MLFlow logging failed (non-critical): {e}")


def track_latency(func):
    """Decorator to track function latency."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed:.1f} ms")
        return result
    return wrapper
