"""Hard difficulty grader with per-episode adaptive penalties."""

from __future__ import annotations

from typing import Dict

from .base_grader import BaseGrader, GradeResult
from .reward_utils import normalize_reward


class HardGrader(BaseGrader):
    """Tracks repeated mistakes and rewards measurable improvement.

    Determinism note:
    - No random branches are used.
    - Given the same sequence of inputs in an episode, outputs are identical.
    """

    def __init__(self) -> None:
        self.reset_episode()

    def reset_episode(self) -> None:
        self._missed_fraud_count = 0
        self._overflag_count = 0

    def evaluate_step(
        self,
        action: Dict,
        transaction: Dict,
        true_label: str,
    ) -> GradeResult:
        predicted = str(
            action.get("decision")
            or action.get("action")
            or action.get("prediction")
            or ""
        ).strip().lower()
        if predicted in {"fraud", "anomaly"}:
            predicted = "flag"
        if predicted in {"normal", "allow", "approve"}:
            predicted = "allow"
        label = str(true_label).strip().lower()

        if predicted == "flag" and label == "fraud":
            # Strong base reward for correct fraud detection.
            score = 0.86
            # Improvement bonus if agent previously struggled on fraud misses.
            if self._missed_fraud_count > 0:
                score += 0.03 * min(self._missed_fraud_count, 3)
                self._missed_fraud_count = max(0, self._missed_fraud_count - 1)
        elif predicted != "flag" and label == "fraud":
            # Repeated missed-fraud mistakes get increasingly costly.
            self._missed_fraud_count += 1
            score = 0.22 - 0.05 * min(self._missed_fraud_count - 1, 4)
        elif predicted == "flag" and label != "fraud":
            # Repeated over-flagging also gets increasingly costly.
            self._overflag_count += 1
            score = 0.42 - 0.05 * min(self._overflag_count - 1, 4)
        else:
            # Correct normal gets decent reward; slight bonus if recovering from
            # prior over-flagging tendency.
            score = 0.66
            if self._overflag_count > 0:
                score += 0.02 * min(self._overflag_count, 3)
                self._overflag_count = max(0, self._overflag_count - 1)

        rationale = str(action.get("rationale", "")).lower()
        # In hard mode, explanation quality matters but stays low-weight.
        rationale_bonus = 0.0
        if "amount" in rationale:
            rationale_bonus += 0.01
        if "location" in rationale:
            rationale_bonus += 0.01
        if "unusual pattern" in rationale:
            rationale_bonus += 0.01

        return {"score": normalize_reward(score + rationale_bonus)}

