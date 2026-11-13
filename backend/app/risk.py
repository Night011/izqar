"""Explainable, deterministic demo risk rules; NOT a trained ML classifier."""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from sqlalchemy import func, select, distinct
from sqlalchemy.orm import Session
from .models import Transaction
from .schemas import TransferIn

@dataclass(frozen=True)
class RiskResult:
    score: int
    level: str
    decision: str
    reasons: list[str]
    indicators: dict


def assess(transfer: TransferIn, db: Session) -> RiskResult:
    # Only operations already present in the DB are used; never look into future events.
    now = datetime.now(timezone.utc)
    past_window = now - timedelta(minutes=15)
    new_recipient = db.scalar(select(Transaction.id).where(
        Transaction.sender == transfer.sender, Transaction.recipient == transfer.recipient).limit(1)) is None
    recent_count = db.scalar(select(func.count(Transaction.id)).where(
        Transaction.sender == transfer.sender, Transaction.created_at >= past_window)) or 0
    source_count = db.scalar(select(func.count(distinct(Transaction.sender))).where(
        Transaction.recipient == transfer.recipient)) or 0
    destination_count = db.scalar(select(func.count(distinct(Transaction.recipient))).where(
        Transaction.sender == transfer.sender)) or 0
    reasons: list[str] = []
    score = 5

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    if transfer.amount >= 1_000_000:
        add(33, 'Крупная сумма перевода (≥ 1 000 000 ₸)')
    elif transfer.amount >= 300_000:
        add(18, 'Повышенная сумма перевода (≥ 300 000 ₸)')
    elif transfer.amount >= 100_000:
        add(7, 'Сумма выше 100 000 ₸')
    if new_recipient:
        add(12, 'Новый получатель для отправителя')
    if transfer.device_changed:
        add(24, 'Изменение устройства (сигнал симулятора)')
    if transfer.external_flag:
        add(16, 'Внешний сигнал риска (синтетический)')
    if recent_count >= 3:
        add(25, 'Серия переводов за последние 15 минут')
    elif recent_count >= 2:
        add(13, 'Повторные переводы за короткий период')
    if source_count >= 3:
        add(22, 'Получатель уже собирал средства с ≥ 3 счетов')
    if destination_count >= 3:
        add(15, 'Отправитель переводил средства на ≥ 3 разных счета')
    score = min(100, score)
    if score >= 70:
        level, decision = 'high', 'review'
    elif score >= 35:
        level, decision = 'medium', 'step_up'
    else:
        level, decision = 'low', 'allow'
    if not reasons:
        reasons.append('По демонстрационным правилам значимых сигналов не обнаружено')
    return RiskResult(score, level, decision, reasons, {
        'new_recipient': new_recipient,
        'recent_sender_transfers_15m': recent_count,
        'recipient_unique_senders_before': source_count,
        'sender_unique_recipients_before': destination_count,
        'score_type': 'rule_based_demo_not_probability',
    })
