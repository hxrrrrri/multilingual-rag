"""Tests for chunking service."""
import pytest
from app.services.chunking_service import chunk_pages, _split_sentences


def test_split_sentences_english():
    text = "The policy applies to all employees. It was updated in 2024. Please review it carefully."
    sents = _split_sentences(text)
    assert len(sents) == 3


def test_split_sentences_hindi():
    text = "यह नीति सभी कर्मचारियों पर लागू होती है। इसे 2024 में अपडेट किया गया।"
    sents = _split_sentences(text)
    assert len(sents) >= 1


def test_chunk_pages_basic():
    pages = [
        {"page": 1, "text": "The quick brown fox jumps over the lazy dog. " * 20, "language": "en"},
        {"page": 2, "text": "This is page two content with more text. " * 20, "language": "en"},
    ]
    chunks = chunk_pages(pages, min_tokens=10, max_tokens=100)
    assert len(chunks) > 1
    for c in chunks:
        assert "text" in c
        assert "page" in c
        assert "chunk_index" in c


def test_chunk_overlap():
    pages = [{"page": 1, "text": " ".join([f"word{i}" for i in range(200)]), "language": "en"}]
    chunks = chunk_pages(pages, max_tokens=50, overlap_ratio=0.2)
    if len(chunks) > 1:
        # Verify some overlap between consecutive chunks
        words0 = set(chunks[0]["text"].split())
        words1 = set(chunks[1]["text"].split())
        assert len(words0 & words1) > 0, "Expected overlap between chunks"
