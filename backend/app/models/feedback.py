"""
SQLAlchemy model for user feedback (thumbs up/down) used in DPO loop.
"""
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Boolean, DateTime, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id:           Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    query:        Mapped[str] = mapped_column(Text)
    answer:       Mapped[str] = mapped_column(Text)
    context:      Mapped[str] = mapped_column(Text)
    doc_ids:      Mapped[list] = mapped_column(JSON, default=list)
    thumbs_up:    Mapped[bool] = mapped_column(Boolean)
    comment:      Mapped[str]  = mapped_column(Text, nullable=True)
    faithfulness: Mapped[float] = mapped_column(Float, nullable=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
