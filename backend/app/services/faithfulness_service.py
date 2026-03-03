"""
NLI-based faithfulness checker.

Uses a cross-encoder NLI model to verify that the generated answer
is entailed by the retrieved context. Suppresses or flags hallucinations.

Score interpretation:
  >= 0.7  → ENTAILED   (safe to return)
  0.4-0.7 → NEUTRAL    (return with warning badge)
  < 0.4   → CONTRADICTION (flag, attempt regeneration)
"""
import asyncio
from typing import Tuple

from loguru import logger
from sentence_transformers.cross_encoder import CrossEncoder

from app.core.config import settings

ENTAILMENT_LABEL = 0   # label index for entailment in NLI models
NEUTRAL_LABEL    = 1
CONTRADICTION_LABEL = 2


class FaithfulnessService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def _load_model(self):
        if self._initialised:
            return
        logger.info(f"Loading NLI model: {settings.NLI_MODEL}")
        self.model = CrossEncoder(settings.NLI_MODEL, num_labels=3)
        self._initialised = True

    async def check(self, answer: str, context: str) -> Tuple[float, str]:
        """
        Check if `answer` is entailed by `context`.
        Returns (entailment_score, label).
        """
        self._load_model()
        loop = asyncio.get_event_loop()

        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict([(context[:1500], answer[:500])],
                                        apply_softmax=True)[0].tolist()
        )

        entail_score = scores[ENTAILMENT_LABEL]
        neutral_score = scores[NEUTRAL_LABEL]
        contra_score = scores[CONTRADICTION_LABEL]

        if entail_score >= 0.7:
            label = "entailed"
        elif contra_score >= 0.5:
            label = "contradiction"
        else:
            label = "neutral"

        logger.debug(f"Faithfulness: {label} (E={entail_score:.2f}, N={neutral_score:.2f}, C={contra_score:.2f})")
        return entail_score, label


faithfulness_service = FaithfulnessService()
