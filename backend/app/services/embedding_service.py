"""
Embedding service — wraps multilingual-e5-large with optional LoRA adapter.

Usage:
    embedder = EmbeddingService()
    vectors = await embedder.embed(["query: What is the policy?",
                                    "passage: The policy states..."])
"""
import asyncio
from typing import List
from functools import lru_cache

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import settings


class EmbeddingService:
    """
    Singleton embedding service.
    multilingual-e5 requires prefix:
      - "query: " for questions
      - "passage: " for documents to index
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def _load_model(self):
        if self._initialised:
            return
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Load LoRA adapter if specified
        if settings.LORA_ADAPTER_PATH:
            logger.info(f"Loading LoRA adapter from: {settings.LORA_ADAPTER_PATH}")
            try:
                from peft import PeftModel
                self.model[0].auto_model = PeftModel.from_pretrained(
                    self.model[0].auto_model,
                    settings.LORA_ADAPTER_PATH,
                )
                logger.info("LoRA adapter loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load LoRA adapter: {e}. Using base model.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self._initialised = True
        logger.info(f"Embedding model ready on {self.device}.")

    async def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed a list of texts. Returns list of float vectors."""
        self._load_model()
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        )
        return vectors

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query with the required 'query: ' prefix."""
        results = await self.embed([f"query: {query}"])
        return results[0]

    async def embed_passages(self, passages: List[str]) -> List[List[float]]:
        """Embed document passages with the required 'passage: ' prefix."""
        prefixed = [f"passage: {p}" for p in passages]
        return await self.embed(prefixed)


# Module-level singleton
embedding_service = EmbeddingService()
