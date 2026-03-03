"""Tests for OCR service."""
import pytest
from unittest.mock import MagicMock, patch
from app.services.ocr_service import OCRService, OCRPage, OCRResult


def make_mock_page(text="Hello world", conf=0.9, lang="en"):
    return OCRPage(page_number=1, text=text, confidence=conf, language=lang, engine_used="paddleocr")


def test_build_result_avg_confidence():
    svc = OCRService()
    pages = [
        make_mock_page(conf=0.8),
        make_mock_page(conf=0.9),
        make_mock_page(conf=0.7),
    ]
    result = svc._build_result(pages, elapsed=1.0)
    assert abs(result.avg_confidence - 0.8) < 0.01
    assert result.total_pages == 3


def test_build_result_empty_pages():
    svc = OCRService()
    result = svc._build_result([], elapsed=0.0)
    assert result.avg_confidence == 0.0
    assert result.total_pages == 0


def test_detect_lang_short_text():
    svc = OCRService()
    lang = svc._detect_lang("hi")   # too short, should default to 'en'
    assert lang == "en"


def test_detect_lang_english():
    svc = OCRService()
    lang = svc._detect_lang("This is a normal English sentence for testing purposes.")
    assert lang == "en"
