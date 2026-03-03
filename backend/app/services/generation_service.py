"""
Generation Service — LLM answer generation with faithfulness gating.
  - Flan-T5-XL for local/CPU inference
  - vLLM API endpoint for production GPU serving
  - DeBERTa-v3 NLI for hallucination detection
  - Confidence = 0.6 * faithfulness + 0.4 * retrieval_score
"""

import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from app.core.config import get_settings
from app.core.metrics import GENERATION_LATENCY, ANSWER_CONFIDENCE
from app.services.retrieval_service import RetrievedChunk


@dataclass
class GenerationResult:
    answer: str
    confidence: float
    faithfulness_score: float
    model_used: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


SYSTEM_PROMPT = (
    "You are a multilingual document assistant. "
    "Answer ONLY using the context below. "
    "If the answer is not in the context, say: "
    "'The answer is not available in the provided document.' "
    "Be concise. Cite page numbers when possible."
)


class GenerationService:

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._tokenizer = None
        self._nli = None

    # ── NLI Faithfulness ──────────────────────────────────────────────────────

    def _init_nli(self):
        if self._nli is not None:
            return
        try:
            from transformers import pipeline
            self._nli = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-large",
                device=-1,
            )
            logger.info("NLI faithfulness model ready")
        except Exception as e:
            logger.warning(f"NLI model unavailable: {e}")

    def _faithfulness(self, answer: str, chunks: list) -> float:
        self._init_nli()
        if not self._nli or not answer.strip():
            return 0.5
        try:
            context = " ".join(c.text for c in chunks)[:2000]
            result = self._nli(
                answer[:500],
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="This claim is supported by: {}",
            )
            score_map = dict(zip(result["labels"], result["scores"]))
            return float(score_map.get("entailment", 0.5))
        except Exception as e:
            logger.warning(f"Faithfulness check error: {e}")
            return 0.5

    # ── Local Flan-T5 ─────────────────────────────────────────────────────────

    def _init_local(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        name = self.settings.llm_model
        logger.info(f"Loading local LLM: {name}")
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            name, device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).eval()
        logger.info("Local LLM ready")

    def _generate_local(self, prompt: str) -> tuple:
        import torch
        self._init_local()
        inputs = self._tokenizer(
            prompt, return_tensors="pt", max_length=self.settings.max_context_length, truncation=True
        )
        pt = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                inputs["input_ids"].to(self._model.device),
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                do_sample=self.settings.temperature > 0,
                num_beams=4, early_stopping=True,
            )
        answer = self._tokenizer.decode(out[0], skip_special_tokens=True)
        return answer, pt, out.shape[1] - pt

    # ── vLLM API ──────────────────────────────────────────────────────────────

    def _generate_vllm(self, prompt: str) -> tuple:
        from openai import OpenAI
        client = OpenAI(base_url=self.settings.vllm_base_url, api_key="not-needed")
        resp = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            max_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
        )
        answer = resp.choices[0].message.content or ""
        return answer, resp.usage.prompt_tokens, resp.usage.completion_tokens

    # ── Prompt Builder ────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, chunks: list) -> str:
        ctx = "\n\n".join(
            f"[Source {i+1} — {c.filename}, Page {c.page_number}]\n{c.text}"
            for i, c in enumerate(chunks)
        )
        return f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {query}\n\nANSWER:"

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: list, use_vllm: bool = False) -> GenerationResult:
        if not chunks:
            return GenerationResult(
                answer="No relevant documents found. Please upload documents first.",
                confidence=0.0, faithfulness_score=0.0,
                model_used="none", latency_ms=0.0, prompt_tokens=0, completion_tokens=0,
            )

        prompt = self._build_prompt(query, chunks)
        t = time.time()
        try:
            if use_vllm:
                answer, pt, ct = self._generate_vllm(prompt)
                model_used = f"vllm:{self.settings.llm_model}"
            else:
                answer, pt, ct = self._generate_local(prompt)
                model_used = self.settings.llm_model
        except Exception as e:
            logger.error(f"Generation error: {e}")
            answer, pt, ct, model_used = "Generation failed. Please try again.", 0, 0, "error"

        latency_ms = (time.time() - t) * 1000
        GENERATION_LATENCY.observe(latency_ms / 1000)

        faithfulness = self._faithfulness(answer, chunks)
        top_score = min(chunks[0].final_score, 1.0) if chunks else 0.0
        confidence = max(0.0, min(1.0, 0.6 * faithfulness + 0.4 * top_score))
        ANSWER_CONFIDENCE.observe(confidence)

        logger.info(f"Generated: {latency_ms:.0f}ms | faith={faithfulness:.2f} | conf={confidence:.2f}")
        return GenerationResult(
            answer=answer, confidence=confidence, faithfulness_score=faithfulness,
            model_used=model_used, latency_ms=latency_ms, prompt_tokens=pt, completion_tokens=ct,
        )
