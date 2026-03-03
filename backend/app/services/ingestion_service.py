"""
Document ingestion pipeline orchestrator.

Stages:
  1. Save uploaded file
  2. OCR extraction (pdfplumber → PaddleOCR fallback)
  3. Semantic chunking
  4. Embedding (multilingual-e5-large + optional LoRA)
  5. Index into Qdrant (dense) + Elasticsearch (BM25)
  6. Persist metadata to PostgreSQL
"""
import os
import uuid
from pathlib import Path
from typing import List, Dict

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.models import PointStruct

from app.core.config import settings
from app.core.vector_store import get_qdrant, get_es
from app.models.document import Document, Chunk, ProcessingStatus
from app.services.ocr_service import extract_text
from app.services.chunking_service import chunk_pages
from app.services.embedding_service import embedding_service


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    db: AsyncSession,
) -> Document:
    """
    Full ingestion pipeline. Returns the created Document record.
    """
    # 1. Save file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{filename}")
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved file: {file_path}")

    # 2. Create DB record
    doc = Document(id=doc_id, filename=filename, file_path=file_path,
                   status=ProcessingStatus.processing)
    db.add(doc)
    await db.commit()

    try:
        # 3. OCR
        pages, language = extract_text(file_path)
        doc.page_count = len(pages)
        doc.language = language

        # 4. Chunk
        chunks_data = chunk_pages(pages)
        logger.info(f"Created {len(chunks_data)} chunks from {len(pages)} pages")

        # 5. Embed
        texts = [c["text"] for c in chunks_data]
        vectors = await embedding_service.embed_passages(texts)

        # 6. Index into Qdrant
        qdrant = get_qdrant()
        points = []
        chunk_records = []

        for i, (chunk, vector) in enumerate(zip(chunks_data, vectors)):
            chunk_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "doc_id":   doc_id,
                    "text":     chunk["text"],
                    "page":     chunk["page"],
                    "language": chunk["language"],
                    "chunk_index": i,
                }
            ))
            chunk_records.append(Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk["text"],
                page=chunk["page"],
                language=chunk["language"],
                chunk_index=i,
            ))

        await qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )

        # 7. Index into Elasticsearch
        es = get_es()
        es_docs = [
            {
                "_index": settings.ES_INDEX,
                "_id": chunk_records[i].id,
                "_source": {
                    "chunk_id": chunk_records[i].id,
                    "doc_id":   doc_id,
                    "text":     chunk["text"],
                    "page":     chunk["page"],
                    "language": chunk["language"],
                }
            }
            for i, chunk in enumerate(chunks_data)
        ]
        from elasticsearch.helpers import async_bulk
        await async_bulk(es, es_docs)

        # 8. Persist chunks to DB
        db.add_all(chunk_records)
        doc.chunk_count = len(chunk_records)
        doc.status = ProcessingStatus.completed
        await db.commit()

        logger.info(f"Ingestion complete for doc {doc_id}: {len(chunk_records)} chunks indexed")
        return doc

    except Exception as e:
        logger.exception(f"Ingestion failed for {doc_id}: {e}")
        doc.status = ProcessingStatus.failed
        await db.commit()
        raise
