"""Only synthetic demo transactions in the current release."""
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import DateTime, Float, Index, JSON, String, Boolean
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
