"""
SQLAlchemy models for documents and chunks.
"""
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, JSON, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.database import Base


class ProcessingStatus(str, enum.Enum):
    pending    = "pending"
    processing = "processing"
    completed  = "completed"
    failed     = "failed"


class Document(Base):
    __tablename__ = "documents"

    id:          Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename:    Mapped[str] = mapped_column(String(512))
    language:    Mapped[str] = mapped_column(String(20), default="en")
    status:      Mapped[ProcessingStatus] = mapped_column(SAEnum(ProcessingStatus), default=ProcessingStatus.pending)
    page_count:  Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    file_path:   Mapped[str] = mapped_column(String(1024))
    meta:        Mapped[dict] = mapped_column(JSON, default=dict)
    created_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete")


class Chunk(Base):
    __tablename__ = "chunks"

    id:          Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id:      Mapped[str] = mapped_column(String(36), ForeignKey("documents.id", ondelete="CASCADE"))
    text:        Mapped[str] = mapped_column(Text)
    page:        Mapped[int] = mapped_column(Integer, default=0)
    language:    Mapped[str] = mapped_column(String(20), default="en")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    meta:        Mapped[dict] = mapped_column(JSON, default=dict)
    created_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
