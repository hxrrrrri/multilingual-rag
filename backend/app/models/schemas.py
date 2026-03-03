from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    READY      = "ready"
    FAILED     = "failed"


class Language(str, Enum):
    AUTO      = "auto"
    ENGLISH   = "en"
    HINDI     = "hi"
    MALAYALAM = "ml"


class FeedbackSentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


# ── Document ──────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str
    created_at: datetime


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    language_detected: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    ocr_confidence: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentStatusResponse]
    total: int
    page: int
    page_size: int


# ── Query ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    document_ids: Optional[list[str]] = None
    language: Language = Language.AUTO
    top_k: int = Field(5, ge=1, le=20)
    include_sources: bool = True


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    similarity_score: float
    retrieval_method: str


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    confidence: float
    language_detected: str
    sources: list[SourceChunk]
    faithfulness_score: Optional[float] = None
    latency_ms: float
    model_used: str
    created_at: datetime


# ── Feedback ──────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str
    sentiment: FeedbackSentiment
    comment: Optional[str] = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    feedback_id: str
    query_id: str
    sentiment: FeedbackSentiment
    message: str
    created_at: datetime


# ── Health ────────────────────────────────────────────────────

class ServiceHealth(BaseModel):
    name: str
    status: str
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    services: list[ServiceHealth]
    timestamp: datetime
