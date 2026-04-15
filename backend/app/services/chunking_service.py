"""
Semantic chunking service.

Strategy:
  - Split pages into paragraphs / sentences
  - Merge small chunks (< min_tokens) with neighbours
  - Split large chunks (> max_tokens) on sentence boundaries
  - Add 10% overlap between consecutive chunks
"""
import re
from typing import List, Dict


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter that handles Hindi/Malayalam too."""
    # Split on '.', '।' (Devanagari danda), '|' used as danda in Malayalam
    pattern = r"(?<=[.!?।|])\s+"
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_pages(
    pages: List[Dict],
    min_tokens: int = 50,
    max_tokens: int = 400,
    overlap_ratio: float = 0.1,
) -> List[Dict]:
    """
    Convert extracted pages into overlapping text chunks.

    Returns list of dicts:
      { text, page, language, chunk_index, token_count }
    """
    all_sentences: List[Dict] = []
    for page in pages:
        sentences = _split_sentences(page["text"])
        for sent in sentences:
            words = sent.split()
            if len(words) < 3:
                continue
            all_sentences.append({
                "text": sent,
                "page": page["page"],
                "language": page.get("language", "en"),
                "tokens": len(words),
            })

    # Guard: if no sentences were extracted, return empty list
    if not all_sentences:
        return []

    # Group sentences into chunks
    chunks: List[Dict] = []
    current_text = ""
    current_tokens = 0
    current_page = all_sentences[0]["page"]
    current_lang = all_sentences[0]["language"]

    overlap_buf: List[str] = []

    for sent in all_sentences:
        if current_tokens + sent["tokens"] > max_tokens and current_tokens >= min_tokens:
            chunks.append({
                "text": current_text.strip(),
                "page": current_page,
                "language": current_lang,
                "chunk_index": len(chunks),
                "token_count": current_tokens,
            })
            # Overlap: carry last overlap_ratio worth of sentences
            overlap_words = int(current_tokens * overlap_ratio)
            words = current_text.split()
            overlap_text = " ".join(words[-overlap_words:]) if overlap_words else ""
            current_text = overlap_text + " " + sent["text"]
            current_tokens = len(current_text.split())
            current_page = sent["page"]
            current_lang = sent["language"]
        else:
            current_text += " " + sent["text"]
            current_tokens += sent["tokens"]

    if current_text.strip() and current_tokens >= min_tokens // 2:
        chunks.append({
            "text": current_text.strip(),
            "page": current_page,
            "language": current_lang,
            "chunk_index": len(chunks),
            "token_count": current_tokens,
        })

    return chunks
