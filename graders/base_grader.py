"""Base grader interface for transaction anomaly tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, TypedDict


class GradeResult(TypedDict):
    """Normalized scoring payload returned by all graders."""

    score: float


class BaseGrader(ABC):
    """Abstract and reusable grader contract.

    Subclasses must implement `evaluate_step` and return a dict with
    `{"score": float}` where score is already normalized to [0.0, 1.0].
    """

    def reset_episode(self) -> None:
        """Reset any per-episode state.

        Stateless graders can keep the default no-op implementation.
        """

    def reset(self) -> None:
        """Compatibility alias used by some env wrappers."""
        self.reset_episode()

    def evaluate(
        self,
        action: Dict,
        transaction: Dict,
        true_label: str | None = None,
    ) -> GradeResult:
        """Compatibility wrapper around `evaluate_step`.

        If `true_label` is omitted, it is read from transaction["true_label"].
        """
        label = true_label
        if label is None:
            label = str(transaction.get("true_label", "normal"))
        return self.evaluate_step(action=action, transaction=transaction, true_label=label)

    @abstractmethod
    def evaluate_step(
        self,
        action: Dict,
        transaction: Dict,
        true_label: str,
    ) -> GradeResult:
        """Evaluate one environment step and return a normalized score."""
        raise NotImplementedError

