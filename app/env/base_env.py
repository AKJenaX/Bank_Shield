from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import Observation, StepResult


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self, task_name: str) -> StepResult:
        """Reset environment state and return initial transition payload."""

    @abstractmethod
    def step(self, action_str: str) -> StepResult:
        """Apply one action and return transition payload."""

    @abstractmethod
    def state(self) -> Observation:
        """Return current observation without stepping."""
