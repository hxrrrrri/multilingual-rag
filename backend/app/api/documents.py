"""
Document ingestion API.
POST /api/v1/documents/ingest  — upload a PDF
GET  /api/v1/documents/        — list documents
GET  /api/v1/documents/{id}    — get document details
DELETE /api/v1/documents/{id}  — delete document
"""
import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document

router = APIRouter()


@router.post("/ingest", summary="Upload and process a PDF document")
async def ingest(
    file: UploadFile = File(..., description="PDF file to ingest"),
    db: AsyncSession = Depends(get_db),
):
    # Validate
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_FILE_SIZE_MB} MB.")

    logger.info(f"Ingesting: {file.filename} ({size_mb:.2f} MB)")
    from app.services.ingestion_service import ingest_document

    doc = await ingest_document(content, file.filename, db)

    return {
        "id":          doc.id,
        "filename":    doc.filename,
        "status":      doc.status.value,
        "page_count":  doc.page_count,
        "chunk_count": doc.chunk_count,
        "language":    doc.language,
        "message":     "Document ingested successfully.",
    }


@router.get("/", summary="List all documents")
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
    )
    docs = result.scalars().all()
    return [
        {
            "id":          d.id,
            "filename":    d.filename,
            "language":    d.language,
            "status":      d.status.value,
            "page_count":  d.page_count,
            "chunk_count": d.chunk_count,
            "created_at":  d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.get("/{doc_id}", summary="Get document details")
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    return {
        "id":          doc.id,
        "filename":    doc.filename,
        "language":    doc.language,
        "status":      doc.status.value,
        "page_count":  doc.page_count,
        "chunk_count": doc.chunk_count,
        "meta":        doc.meta,
        "created_at":  doc.created_at.isoformat(),
    }


@router.delete("/{doc_id}", summary="Delete a document and all its vectors")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    from app.core.vector_store import get_qdrant, get_es
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")

    # Delete from Qdrant
    qdrant = get_qdrant()
    await qdrant.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )

    # Delete from Elasticsearch
    es = get_es()
    await es.delete_by_query(
        index=settings.ES_INDEX,
        body={"query": {"term": {"doc_id": doc_id}}}
    )

    # Delete from DB (cascades to chunks)
    await db.delete(doc)
    await db.commit()

    # Delete file
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    return {"message": f"Document {doc_id} deleted."}
