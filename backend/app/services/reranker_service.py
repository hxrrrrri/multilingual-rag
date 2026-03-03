"""
Cross-encoder reranking service.
Uses ms-marco-MiniLM to rerank top-k candidates from hybrid retrieval.
"""
import asyncio
from typing import List, Dict

import torch
from loguru import logger
from sentence_transformers.cross_encoder import CrossEncoder

from app.core.config import settings


class RerankerService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def _load_model(self):
        if self._initialised:
            return
        logger.info(f"Loading reranker: {settings.RERANKER_MODEL}")
        self.model = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
        self._initialised = True
        logger.info("Reranker ready.")

    async def rerank(self, query: str, candidates: List[Dict], top_n: int = None) -> List[Dict]:
        """
        Score each (query, passage) pair and return top_n sorted by score.
        """
        top_n = top_n or settings.RERANK_TOP_N
        self._load_model()

        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]

        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict(pairs, show_progress_bar=False).tolist()
        )

        for c, s in zip(candidates, scores):
            c["rerank_score"] = s

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(f"Reranked {len(candidates)} → top {top_n}")
        return reranked[:top_n]


reranker_service = RerankerService()
