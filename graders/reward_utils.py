"""Reward utility helpers for grader outputs."""


def normalize_reward(score: float) -> float:
    """Clamp any numeric score to [0.0, 1.0].

    Handles negative values, very large values, and non-float inputs that can
    be cast to float.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value

