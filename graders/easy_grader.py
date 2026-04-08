"""Easy difficulty grader with transparent reward shaping."""

from __future__ import annotations

from typing import Dict

from .base_grader import BaseGrader, GradeResult
from .reward_utils import normalize_reward


class EasyGrader(BaseGrader):
    """Rewards obvious fraud detection behavior with light rationale bonuses."""

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
            score = 0.90  # correct fraud detection, high reward
        elif predicted != "flag" and label == "fraud":
            score = 0.10  # missed fraud, heavy penalty
        elif predicted == "flag" and label != "fraud":
            score = 0.30  # false positive, medium penalty
        else:
            score = 0.70  # correct normal, moderate reward

        rationale = str(action.get("rationale", "")).lower()
        bonus_terms = ("amount", "location", "unusual pattern")
        bonus = 0.0
        for term in bonus_terms:
            if term in rationale:
                bonus += 0.03

        return {"score": normalize_reward(score + bonus)}

