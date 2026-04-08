from graders.easy_grader import EasyGrader
from graders.hard_grader import HardGrader
from graders.reward_utils import normalize_reward


def test_normalize_reward_bounds():
    assert normalize_reward(-5) == 0.0
    assert normalize_reward(5) == 1.0
    assert normalize_reward(0.42) == 0.42


def test_easy_correct_fraud_detection_high_reward():
    grader = EasyGrader()
    result = grader.evaluate_step(
        action={"decision": "flag", "rationale": "High amount and unusual pattern"},
        transaction={"id": "tx1"},
        true_label="fraud",
    )
    assert 0.9 <= result["score"] <= 1.0


def test_easy_missed_fraud_penalty():
    grader = EasyGrader()
    result = grader.evaluate_step(
        action={"decision": "allow", "rationale": "Looks normal"},
        transaction={"id": "tx2"},
        true_label="fraud",
    )
    assert 0.0 <= result["score"] < 0.3


def test_easy_false_positive_penalty():
    grader = EasyGrader()
    result = grader.evaluate_step(
        action={"decision": "flag", "rationale": "Noisy signal"},
        transaction={"id": "tx3"},
        true_label="normal",
    )
    assert 0.2 <= result["score"] <= 0.5


def test_easy_rationale_bonus_terms():
    grader = EasyGrader()
    no_bonus = grader.evaluate_step(
        action={"decision": "flag", "rationale": "Suspicious"},
        transaction={"id": "tx4"},
        true_label="fraud",
    )["score"]
    with_bonus = grader.evaluate_step(
        action={
            "decision": "flag",
            "rationale": "High amount, odd location, unusual pattern",
        },
        transaction={"id": "tx4"},
        true_label="fraud",
    )["score"]
    assert with_bonus >= no_bonus
    assert 0.0 <= with_bonus <= 1.0


def test_hard_grader_repeated_missed_fraud_penalty_increases():
    grader = HardGrader()
    first = grader.evaluate_step(
        action={"decision": "allow", "rationale": "Unsure"},
        transaction={"id": "h1"},
        true_label="fraud",
    )["score"]
    second = grader.evaluate_step(
        action={"decision": "allow", "rationale": "Still unsure"},
        transaction={"id": "h2"},
        true_label="fraud",
    )["score"]
    assert second <= first
    assert 0.0 <= first <= 1.0
    assert 0.0 <= second <= 1.0


def test_hard_grader_improvement_bonus_after_mistakes():
    grader = HardGrader()
    grader.evaluate_step(
        action={"decision": "allow", "rationale": "Missed clue"},
        transaction={"id": "h3"},
        true_label="fraud",
    )
    improved = grader.evaluate_step(
        action={"decision": "flag", "rationale": "amount and location unusual pattern"},
        transaction={"id": "h4"},
        true_label="fraud",
    )["score"]
    assert improved > 0.86
    assert 0.0 <= improved <= 1.0


def test_compatibility_aliases_for_integration():
    grader = EasyGrader()
    via_evaluate = grader.evaluate(
        action={"action": "fraud", "rationale": "amount location unusual pattern"},
        transaction={"id": "alias-1", "true_label": "fraud"},
    )
    assert 0.0 <= via_evaluate["score"] <= 1.0

