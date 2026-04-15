"""
Query API.
POST /api/v1/query         — query documents
GET  /api/v1/query/stream  — streaming response (SSE)
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from app.services.rag_pipeline import run_rag_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list
    faithfulness_score: float
    faithfulness_label: str
    regenerated: bool


@router.post("/query", response_model=QueryResponse, summary="Query documents with RAG")
async def query_documents(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    if req.stream:
        raise HTTPException(400, "Use /query/stream for streaming.")

    result = await run_rag_pipeline(req.query, doc_id=req.doc_id)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        faithfulness_score=result["faithfulness_score"],
        faithfulness_label=result["faithfulness_label"],
        regenerated=result["regenerated"],
    )


@router.post("/query/stream", summary="Stream query response (SSE)")
async def stream_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    from app.services.llm_service import stream_answer
    from app.services.retrieval_service import hybrid_search
    from app.services.reranker_service import reranker_service

    candidates = await hybrid_search(req.query, doc_id=req.doc_id)
    top_chunks  = await reranker_service.rerank(req.query, candidates)

    async def event_generator():
        # First emit sources
        sources = [
            {"doc_id": c.get("doc_id"), "page": c.get("page"),
             "text_preview": c["text"][:200]}
            for c in top_chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Then stream tokens
        async for token in stream_answer(req.query, top_chunks):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
