"""Tests for FastAPI endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from main import app


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/")
    assert r.status_code == 200
    assert "version" in r.json()


@pytest.mark.asyncio
async def test_health_live():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


@pytest.mark.asyncio
async def test_health_ready():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/health/ready")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_list_documents_empty():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/documents")
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_get_nonexistent_document():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/documents/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_query_history_empty():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/query/history")
    assert r.status_code == 200
    assert "queries" in r.json()


@pytest.mark.asyncio
async def test_feedback_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/feedback")
    assert r.status_code == 200
    assert "feedback" in r.json()
