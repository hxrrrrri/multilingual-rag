"""Tests for OCR language detection."""
import pytest
from app.services.ocr_service import detect_language


def test_detect_english():
    assert detect_language("This is a standard English document.") == "en"


def test_detect_hindi():
    # Devanagari script
    assert detect_language("यह एक हिंदी दस्तावेज़ है " * 10) == "hi"


def test_detect_malayalam():
    # Malayalam script
    assert detect_language("ഇത് ഒരു മലയാളം രേഖയാണ് " * 10) == "ml"


def test_detect_empty():
    assert detect_language("") == "en"
