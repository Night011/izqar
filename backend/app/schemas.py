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
    review_state: Literal['pending', 'confirmed_fraud', 'false_positive']

class SimulationIn(BaseModel):
    scenario: Literal['normal', 'new_recipient', 'velocity', 'fan_in', 'chain']

class IeeeFeaturesIn(BaseModel):
    features: dict[str, float | str | None] = Field(min_length=1, max_length=100)
