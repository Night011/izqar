from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

ACCOUNT_CHARS = 64

class TransferIn(BaseModel):
    sender: str = Field(min_length=2, max_length=ACCOUNT_CHARS, pattern=r'^[A-Za-z0-9_-]+$')
    recipient: str = Field(min_length=2, max_length=ACCOUNT_CHARS, pattern=r'^[A-Za-z0-9_-]+$')
    amount: float = Field(gt=0, le=100_000_000, allow_inf_nan=False)
    currency: Literal['KZT'] = 'KZT'
    device_changed: bool = False
    external_flag: bool = False

    @field_validator('recipient')
    @classmethod
    def not_empty(cls, value: str) -> str:
        return value

class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime
    sender: str
    recipient: str
    amount: float
    currency: str
    scenario: str
    device_changed: bool
    external_flag: bool
    risk_score: int
    risk_level: str
    decision: str
    reasons: list[str]
    indicators: dict
    review_state: str

class ReviewIn(BaseModel):
    review_state: Literal['pending', 'in_review', 'step_up', 'escalated', 'allowed',
                          'confirmed_fraud', 'false_positive']
    comment: str | None = Field(default=None, max_length=2000)

    @field_validator('comment')
    @classmethod
    def validate_comment(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError('Comment must contain text')
        return value


class CommentIn(BaseModel):
    comment: str = Field(min_length=1, max_length=2000)

    @field_validator('comment')
    @classmethod
    def validate_comment(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('Comment must contain text')
        return value


class InvestigationEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    transaction_id: str
    created_at: datetime
    action: str
    from_state: str | None
    to_state: str | None
    comment: str | None
    actor_source: str


class InvestigationListOut(BaseModel):
    total: int
    items: list[TransactionOut]


class InvestigationDetailOut(BaseModel):
    transaction: TransactionOut
    events: list[InvestigationEventOut]
    events_total: int
    related_transactions: list[TransactionOut]

class SimulationIn(BaseModel):
    scenario: Literal['normal', 'new_recipient', 'velocity', 'fan_in', 'chain', 'cycle']

class IeeeFeaturesIn(BaseModel):
    features: dict[str, float | str | None] = Field(min_length=1, max_length=100)

class PaySimFeaturesIn(BaseModel):
    features: dict[str, float | str | None] = Field(min_length=4, max_length=4)


class BafFeaturesIn(BaseModel):
    features: dict[str, float | str | None] = Field(min_length=1, max_length=50)


class IbmAmlFeaturesIn(BaseModel):
    features: dict[str, float | str | None] = Field(min_length=1, max_length=20)


class UnifiedFeaturesIn(BaseModel):
    features: dict[str, float | str | None] = Field(min_length=1, max_length=100)
