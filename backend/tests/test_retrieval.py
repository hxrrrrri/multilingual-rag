"""Tests for RRF fusion."""
import pytest
from app.services.retrieval_service import reciprocal_rank_fusion


def test_rrf_basic():
    dense = [
        {"chunk_id": "a", "text": "doc a", "score": 0.9},
        {"chunk_id": "b", "text": "doc b", "score": 0.8},
    ]
    sparse = [
        {"chunk_id": "b", "text": "doc b", "score": 15.0},
        {"chunk_id": "c", "text": "doc c", "score": 12.0},
    ]
    merged = reciprocal_rank_fusion([dense, sparse], k=60)
    assert len(merged) == 3
    ids = [m["chunk_id"] for m in merged]
    assert "b" in ids
    # "b" appears in both lists — should rank highest
    assert ids[0] == "b"


def test_rrf_deduplication():
    list1 = [{"chunk_id": "x", "text": "t", "score": 1.0}]
    list2 = [{"chunk_id": "x", "text": "t", "score": 1.0}]
    merged = reciprocal_rank_fusion([list1, list2])
    assert len(merged) == 1


def test_rrf_empty():
    merged = reciprocal_rank_fusion([[], []])
    assert merged == []
