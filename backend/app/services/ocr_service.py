"""
OCR Service — extracts text from PDFs using PaddleOCR + pdfplumber.

Pipeline:
  1. Try pdfplumber (fast, good for digital PDFs)
  2. If confidence low or image-based → fall back to PaddleOCR
  3. Detect language of each page
"""
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import pdfplumber
from PIL import Image
from loguru import logger

# Lazy imports to avoid loading GPU models at import time
_paddle_ocr = None


def _get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _paddle_ocr


def detect_language(text: str) -> str:
    """
    Heuristic language detection based on Unicode ranges.
    Returns ISO 639-1 code: 'en', 'hi', 'ml'.
    """
    if not text:
        return "en"
    total = len(text)
    devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    malayalam  = sum(1 for c in text if "\u0D00" <= c <= "\u0D7F")
    if malayalam / total > 0.1:
        return "ml"
    if devanagari / total > 0.1:
        return "hi"
    return "en"


@dataclass
class OCRPage:
    page_number: int
    text: str
    confidence: float
    language: str
    engine_used: str


@dataclass
class OCRResult:
    pages: List[OCRPage]
    total_pages: int
    avg_confidence: float
    language: str
    processing_time_seconds: float


class OCRService:
    """Class-based OCR facade used by tests and higher-level orchestrators."""

    def _detect_lang(self, text: str) -> str:
        if not text or len(text.strip()) < 8:
            return "en"
        return detect_language(text)

    def _build_result(self, pages: List[OCRPage], elapsed: float) -> OCRResult:
        avg_conf = sum(p.confidence for p in pages) / len(pages) if pages else 0.0
        dominant_lang = self._detect_lang(" ".join(p.text for p in pages)) if pages else "en"
        return OCRResult(
            pages=pages,
            total_pages=len(pages),
            avg_confidence=avg_conf,
            language=dominant_lang,
            processing_time_seconds=elapsed,
        )

    def extract(self, pdf_path: str, confidence_threshold: float = 0.75) -> OCRResult:
        start = time.perf_counter()
        raw_pages, language = extract_text(pdf_path, confidence_threshold=confidence_threshold)
        pages = [
            OCRPage(
                page_number=p["page"],
                text=p["text"],
                confidence=float(p.get("confidence", 0.0)),
                language=p.get("language", language),
                engine_used=p.get("method", "unknown"),
            )
            for p in raw_pages
        ]
        result = self._build_result(pages, elapsed=time.perf_counter() - start)
        result.language = language
        return result


def extract_with_pdfplumber(pdf_path: str) -> List[Dict]:
    """Extract text page-by-page using pdfplumber (digital PDFs)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            confidence = len(text.strip()) / max(page.width * page.height / 100, 1)
            pages.append({
                "page": i + 1,
                "text": text,
                "confidence": min(confidence, 1.0),
                "method": "pdfplumber",
            })
    return pages


def extract_with_paddleocr(pdf_path: str) -> List[Dict]:
    """Extract text using PaddleOCR (scanned / image-based PDFs)."""
    import fitz  # PyMuPDF
    pages = []
    doc = fitz.open(pdf_path)
    ocr = _get_paddle_ocr()

    for i, page in enumerate(doc):
        mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR quality
        clip = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [clip.width, clip.height], clip.samples)

        img_path = Path(tempfile.gettempdir()) / f"page_{i}.png"
        img.save(img_path)

        result = ocr.ocr(img_path, cls=True)
        text = ""
        total_conf = 0
        count = 0
        if result and result[0]:
            for line in result[0]:
                text += line[1][0] + "\n"
                total_conf += line[1][1]
                count += 1

        os.remove(str(img_path))
        pages.append({
            "page": i + 1,
            "text": text,
            "confidence": (total_conf / count) if count else 0.0,
            "method": "paddleocr",
        })

    doc.close()
    return pages


def extract_text(pdf_path: str, confidence_threshold: float = 0.75) -> Tuple[List[Dict], str]:
    """
    Main extraction function.
    Returns (pages, overall_language).

    Steps:
      1. Try pdfplumber
      2. For pages with confidence < threshold, re-run PaddleOCR on those pages
      3. Detect dominant language
    """
    logger.info(f"Extracting text from: {pdf_path}")
    pages = extract_with_pdfplumber(pdf_path)
    low_conf = [p for p in pages if p["confidence"] < confidence_threshold]

    if len(low_conf) > len(pages) * 0.3:
        logger.info(f"{len(low_conf)} low-confidence pages → using PaddleOCR")
        pages = extract_with_paddleocr(pdf_path)

    all_text = " ".join(p["text"] for p in pages)
    language = detect_language(all_text)
    for p in pages:
        p["language"] = detect_language(p["text"]) or language

    logger.info(f"Extracted {len(pages)} pages, language={language}")
    return pages, language
