"""Only synthetic demo transactions in the current release."""
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import DateTime, Float, ForeignKey, Index, JSON, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base

class Transaction(Base):
    __tablename__ = 'transactions'
    __table_args__ = (Index('ix_tx_sender_created', 'sender', 'created_at'), Index('ix_tx_recipient', 'recipient'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    sender: Mapped[str] = mapped_column(String(64), nullable=False)
    recipient: Mapped[str] = mapped_column(String(64), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default='KZT')
    scenario: Mapped[str] = mapped_column(String(48), default='manual')
    device_changed: Mapped[bool] = mapped_column(Boolean, default=False)
    external_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_score: Mapped[int] = mapped_column(default=0)
    risk_level: Mapped[str] = mapped_column(String(12), default='low')
    decision: Mapped[str] = mapped_column(String(24), default='allow')
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    indicators: Mapped[dict] = mapped_column(JSON, default=dict)
    review_state: Mapped[str] = mapped_column(String(24), default='pending')


class InvestigationEvent(Base):
    """Append-only local demo audit trail; the shared API key is not user identity."""

    __tablename__ = 'investigation_events'
    __table_args__ = (Index('ix_investigation_tx_created', 'transaction_id', 'created_at'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    transaction_id: Mapped[str] = mapped_column(String(36), ForeignKey('transactions.id'), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    action: Mapped[str] = mapped_column(String(24), nullable=False)
    from_state: Mapped[str | None] = mapped_column(String(24))
    to_state: Mapped[str | None] = mapped_column(String(24))
    comment: Mapped[str | None] = mapped_column(String(2000))
    actor_source: Mapped[str] = mapped_column(String(32), default='local_demo_api_key')


class InvestigationMlAssessment(Base):
    """A server-produced ML result linked to a case, never a human verdict."""

    __tablename__ = 'investigation_ml_assessments'
    __table_args__ = (Index('ix_ml_assessment_tx_created', 'transaction_id', 'created_at'),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    transaction_id: Mapped[str] = mapped_column(String(36), ForeignKey('transactions.id'), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    model_key: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    score_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    warning: Mapped[str] = mapped_column(String(1000), nullable=False)
    # Keep only the exact schema keys; raw inputs can be sensitive and remain
    # outside the local demo investigation store.
    feature_names: Mapped[list] = mapped_column(JSON, default=list)
