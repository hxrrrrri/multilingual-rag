"""
RAG Pipeline — orchestrates the full query flow:
  query → hybrid_search → rerank → faithfulness_check → generate → return
"""
from typing import Dict, List, Optional
from loguru import logger

from app.services.retrieval_service import hybrid_search
from app.services.reranker_service import reranker_service
from app.services.llm_service import generate_answer
from app.services.faithfulness_service import faithfulness_service
from app.core.config import settings

MAX_REGENERATION_ATTEMPTS = 2


async def run_rag_pipeline(
    query: str,
    doc_id: Optional[str] = None,
) -> Dict:
    """
    Full RAG pipeline.

    Returns:
        {
          answer, context_chunks, faithfulness_score,
          faithfulness_label, sources, regenerated
        }
    """
    logger.info(f"RAG query: '{query[:80]}' | doc_id={doc_id}")

    # 1. Hybrid retrieval
    candidates = await hybrid_search(query, doc_id=doc_id)
    if not candidates:
        return {
            "answer": "No relevant documents found for your query.",
            "context_chunks": [],
            "faithfulness_score": 0.0,
            "faithfulness_label": "no_context",
            "sources": [],
            "regenerated": False,
        }

    # 2. Rerank
    top_chunks = await reranker_service.rerank(query, candidates)

    # 3. Generate + faithfulness loop
    context_text = "\n\n".join(c["text"] for c in top_chunks)
    regenerated = False

    for attempt in range(MAX_REGENERATION_ATTEMPTS + 1):
        answer = await generate_answer(query, top_chunks)
        faith_score, faith_label = await faithfulness_service.check(answer, context_text)

        if faith_label != "contradiction":
            break
        if attempt < MAX_REGENERATION_ATTEMPTS:
            logger.warning(f"Faithfulness check failed (attempt {attempt+1}), regenerating…")
            regenerated = True
        else:
            logger.warning("Max regeneration attempts reached, returning with warning.")

    # 4. Build sources list
    sources = [
        {
            "doc_id": c.get("doc_id"),
            "page":   c.get("page"),
            "text_preview": c["text"][:200] + "…",
            "score": c.get("rerank_score", c.get("rrf_score", 0)),
        }
        for c in top_chunks
    ]

    return {
        "answer":             answer,
        "context_chunks":     top_chunks,
        "faithfulness_score": faith_score,
        "faithfulness_label": faith_label,
        "sources":            sources,
        "regenerated":        regenerated,
    }
