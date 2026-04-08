"""Medium difficulty grader (simple bridge implementation)."""

from __future__ import annotations

from typing import Dict

from .base_grader import BaseGrader, GradeResult
from .reward_utils import normalize_reward


class MediumGrader(BaseGrader):
    """Balanced rewards between easy and hard behavior."""

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
            score = 0.84
        elif predicted != "flag" and label == "fraud":
            score = 0.18
        elif predicted == "flag" and label != "fraud":
            score = 0.36
        else:
            score = 0.68

        return {"score": normalize_reward(score)}

