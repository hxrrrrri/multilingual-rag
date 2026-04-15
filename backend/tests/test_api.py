"""Tests for FastAPI endpoints."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from main import app
from app.core.database import init_db


def _transport() -> ASGITransport:
    return ASGITransport(app=app)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _prepare_database():
    await init_db()


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=_transport(), base_url="http://test") as ac:
        r = await ac.get("/")
    assert r.status_code == 200
    assert "version" in r.json()


@pytest.mark.asyncio
async def test_health_live():
    async with AsyncClient(transport=_transport(), base_url="http://test") as ac:
        r = await ac.get("/api/v1/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


@pytest.mark.asyncio
async def test_health_ready():
    async with AsyncClient(transport=_transport(), base_url="http://test") as ac:
        r = await ac.get("/api/v1/health/ready")
    assert r.status_code == 200
    assert "status" in r.json()


@pytest.mark.asyncio
async def test_list_documents_schema():
    async with AsyncClient(transport=_transport(), base_url="http://test") as ac:
        r = await ac.get("/api/v1/documents/")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_nonexistent_document():
    async with AsyncClient(transport=_transport(), base_url="http://test") as ac:
        r = await ac.get("/api/v1/documents/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_query_empty_validation():
    async with AsyncClient(transport=_transport(), base_url="http://test") as ac:
        r = await ac.post("/api/v1/query", json={"query": ""})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_feedback_stats_schema():
    async with AsyncClient(transport=_transport(), base_url="http://test") as ac:
        r = await ac.get("/api/v1/feedback/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "positive_rate" in data
