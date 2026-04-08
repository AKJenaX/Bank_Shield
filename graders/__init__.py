"""Grading & reward logic (Person 3).

This package is intentionally dependency-light so it can be imported by the
environment server without pulling in LLM or API packages.
"""

from .base_grader import BaseGrader, GradeResult
from .easy_grader import EasyGrader
from .medium_grader import MediumGrader
from .hard_grader import HardGrader


def get_grader(difficulty: str) -> BaseGrader:
    difficulty = (difficulty or "").strip().lower()
    if difficulty in {"easy", "e"}:
        return EasyGrader()
    if difficulty in {"medium", "m", "med"}:
        return MediumGrader()
    if difficulty in {"hard", "h"}:
        return HardGrader()
    raise ValueError(f"Unknown difficulty: {difficulty!r}")

