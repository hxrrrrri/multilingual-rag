"""
Feedback API — collect thumbs up/down for DPO training loop.
POST /api/v1/feedback
GET  /api/v1/feedback/stats
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.models.feedback import Feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    context: str
    doc_ids: list = []
    thumbs_up: bool
    comment: Optional[str] = None
    faithfulness_score: Optional[float] = None


@router.post("/", summary="Submit feedback for a RAG response")
async def submit_feedback(req: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    fb = Feedback(
        query=req.query,
        answer=req.answer,
        context=req.context,
        doc_ids=req.doc_ids,
        thumbs_up=req.thumbs_up,
        comment=req.comment,
        faithfulness=req.faithfulness_score,
    )
    db.add(fb)
    await db.commit()
    return {"id": fb.id, "message": "Feedback recorded. Thank you!"}


@router.get("/stats", summary="Get feedback statistics")
async def feedback_stats(db: AsyncSession = Depends(get_db)):
    total = await db.scalar(select(func.count(Feedback.id)))
    positive = await db.scalar(select(func.count(Feedback.id)).where(Feedback.thumbs_up == True))
    negative = total - positive
    avg_faith = await db.scalar(select(func.avg(Feedback.faithfulness)))
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_rate": round(positive / total, 3) if total else 0,
        "avg_faithfulness": round(float(avg_faith or 0), 3),
    }


@router.get("/export", summary="Export feedback as preference pairs for DPO")
async def export_feedback(db: AsyncSession = Depends(get_db)):
    """
    Returns positive/negative pairs grouped by query for DPO training.
    """
    result = await db.execute(select(Feedback).order_by(Feedback.created_at))
    rows = result.scalars().all()

    from collections import defaultdict
    by_query: dict = defaultdict(lambda: {"chosen": [], "rejected": []})
    for row in rows:
        key = row.query[:100]
        if row.thumbs_up:
            by_query[key]["chosen"].append(row.answer)
        else:
            by_query[key]["rejected"].append(row.answer)

    pairs = []
    for query, data in by_query.items():
        for chosen in data["chosen"]:
            for rejected in data["rejected"]:
                pairs.append({"query": query, "chosen": chosen, "rejected": rejected})

    return {"count": len(pairs), "pairs": pairs}
