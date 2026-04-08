from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str
    amount: float
    merchant: str | None = None
    category: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    episode_id: str
    step: int
    task_id: str | None = None
    current_transaction: Transaction | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class Action(BaseModel):
    action_str: str


class Reward(BaseModel):
    value: float
    reason: str | None = None


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
