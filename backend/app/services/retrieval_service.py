"""
Hybrid Retrieval Service — Qdrant (dense) + Elasticsearch (BM25) + RRF fusion.

Reciprocal Rank Fusion:
    RRF_score(d) = Σ_r  1 / (k + rank_r(d))
    where k=60 (constant that smooths rankings), r iterates over retrieval methods.
"""
from typing import List, Dict, Tuple

from loguru import logger

from app.core.config import settings
from app.core.vector_store import get_qdrant, get_es


async def dense_search(query_vector: List[float], doc_id: str = None, top_k: int = None) -> List[Dict]:
    """Search Qdrant with dense vector. Optionally filter by doc_id."""
    top_k = top_k or settings.RETRIEVAL_TOP_K
    qdrant = get_qdrant()

    filter_condition = None
    if doc_id:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filter_condition = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    results = await qdrant.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        query_filter=filter_condition,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "chunk_id": r.id,
            "text":     r.payload.get("text", ""),
            "score":    r.score,
            "page":     r.payload.get("page", 0),
            "doc_id":   r.payload.get("doc_id", ""),
            "language": r.payload.get("language", "en"),
            "source":   "dense",
        }
        for r in results
    ]


async def sparse_search(query: str, doc_id: str = None, top_k: int = None) -> List[Dict]:
    """BM25 search via Elasticsearch."""
    top_k = top_k or settings.RETRIEVAL_TOP_K
    es = get_es()

    body: Dict = {
        "query": {
            "bool": {
                "must": [{"match": {"text": {"query": query, "fuzziness": "AUTO"}}}]
            }
        },
        "size": top_k,
    }
    if doc_id:
        body["query"]["bool"]["filter"] = [{"term": {"doc_id": doc_id}}]

    response = await es.search(index=settings.ES_INDEX, body=body)
    hits = response["hits"]["hits"]

    return [
        {
            "chunk_id": h["_source"].get("chunk_id", h["_id"]),
            "text":     h["_source"].get("text", ""),
            "score":    h["_score"],
            "page":     h["_source"].get("page", 0),
            "doc_id":   h["_source"].get("doc_id", ""),
            "language": h["_source"].get("language", "en"),
            "source":   "sparse",
        }
        for h in hits
    ]


def reciprocal_rank_fusion(
    result_lists: List[List[Dict]],
    k: int = None,
) -> List[Dict]:
    """
    Merge multiple ranked lists using RRF.
    Returns deduplicated list sorted by RRF score (descending).
    """
    k = k or settings.RRF_K
    scores: Dict[str, float] = {}
    docs:   Dict[str, Dict]  = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in docs:
                docs[cid] = doc

    merged = sorted(docs.values(), key=lambda d: scores[d["chunk_id"]], reverse=True)
    for doc in merged:
        doc["rrf_score"] = scores[doc["chunk_id"]]
    return merged


async def hybrid_search(query: str, doc_id: str = None) -> List[Dict]:
    """
    Full hybrid search pipeline:
      dense_search + sparse_search → RRF fusion → top-k results
    """
    from app.services.embedding_service import embedding_service

    query_vector = await embedding_service.embed_query(query)

    dense_results  = await dense_search(query_vector, doc_id=doc_id)
    sparse_results = await sparse_search(query, doc_id=doc_id)

    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    logger.debug(f"Hybrid search returned {len(fused)} results for query: '{query[:60]}'")

    return fused[:settings.RETRIEVAL_TOP_K]
