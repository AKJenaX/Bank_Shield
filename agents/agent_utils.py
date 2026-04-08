from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


DECISIONS = {
    "flag_as_fraud",
    "allow",
    "request_review",
    "request_additional_verification",
}


def _format_transaction_details(observation: Dict[str, Any]) -> str:
    # Best-effort extraction; environment may name this differently.
    tx = (
        observation.get("transaction")
        or observation.get("transaction_details")
        or observation.get("data")
        or observation.get("txn")
    )
    if isinstance(tx, dict):
        pretty = json.dumps(tx, indent=2, sort_keys=True)
        return f"Transaction details (JSON):\n{pretty}"
    if tx is not None:
        return f"Transaction details:\n{tx}"

    # Fallback: include all observation keys except prompt to avoid duplication.
    rest = {k: v for k, v in observation.items() if k != "prompt"}
    if rest:
        return "Observation context:\n" + json.dumps(rest, indent=2, sort_keys=True)
    return "Observation context: (none)"


def format_prompt(observation: Dict[str, Any]) -> str:
    """
    Build a single string prompt from an environment observation.

    Requirements:
    - Use observation["prompt"]
    - Include transaction details clearly
    - Instruct model to choose ONE decision:
      - flag_as_fraud
      - allow
      - request_review
      - request_additional_verification
    - Ask for short rationale
    """

    base_prompt = str(observation.get("prompt", "")).strip()
    tx_details = _format_transaction_details(observation)

    return (
        f"{base_prompt}\n\n"
        f"{tx_details}\n\n"
        "You are a financial fraud detection agent.\n"
        "Choose exactly ONE decision from:\n"
        "- flag_as_fraud\n"
        "- allow\n"
        "- request_review\n"
        "- request_additional_verification\n\n"
        "Return your answer as JSON with keys:\n"
        '{ "decision": "<one_of_the_choices>", "rationale": "<short>" }\n'
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    # Try direct JSON first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Then try to find a JSON object inside text.
    # Non-greedy between first { and last } can still fail on braces in rationale,
    # so we use a simple scan for balanced braces.
    start = text.find("{")
    if start == -1:
        return None
    for end in range(len(text) - 1, start, -1):
        if text[end] != "}":
            continue
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _normalize_decision(decision: str) -> Optional[str]:
    d = (decision or "").strip().lower()
    d = d.replace("-", "_").replace(" ", "_")
    if d in DECISIONS:
        return d

    # Common variants / messy outputs.
    aliases = {
        "fraud": "flag_as_fraud",
        "flag": "flag_as_fraud",
        "flag_fraud": "flag_as_fraud",
        "flag_as_fraud.": "flag_as_fraud",
        "review": "request_review",
        "request_a_review": "request_review",
        "additional_verification": "request_additional_verification",
        "verify": "request_additional_verification",
        "verification": "request_additional_verification",
        "approve": "allow",
        "accept": "allow",
        "legit": "allow",
    }
    return aliases.get(d)


def _regex_pick_decision(text: str) -> Optional[str]:
    t = text.lower()
    # Prefer exact decision tokens if present.
    for d in DECISIONS:
        if re.search(rf"\b{re.escape(d)}\b", t):
            return d
    # Then alias-like words.
    if re.search(r"\b(flag|fraud|scam|suspicious)\b", t):
        return "flag_as_fraud"
    if re.search(r"\b(allow|approve|accept|legit|not\s+fraud)\b", t):
        return "allow"
    if re.search(r"\b(review|manual\s+review)\b", t):
        return "request_review"
    if re.search(r"\b(verify|verification|additional)\b", t):
        return "request_additional_verification"
    return None


def _split_decision_rationale(text: str) -> Tuple[Optional[str], str]:
    # Try patterns like "decision: X" and "rationale: Y"
    decision_match = re.search(
        r"(?:decision|action)\s*[:=\-]\s*([a-zA-Z_\- ]+)",
        text,
        flags=re.IGNORECASE,
    )
    rationale_match = re.search(
        r"(?:rationale|reason|explanation)\s*[:=\-]\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    decision = decision_match.group(1).strip() if decision_match else None
    rationale = rationale_match.group(1).strip() if rationale_match else text.strip()
    return decision, rationale


def parse_action_from_text(text: str) -> Dict[str, str]:
    """
    Extract:
      { "decision": string, "rationale": string }

    Handles messy outputs robustly:
    - JSON output
    - JSON embedded in text
    - "decision: X" / "rationale: Y"
    - Keyword-based fallback
    """

    raw = (text or "").strip()
    if not raw:
        return {"decision": "allow", "rationale": "fallback"}

    obj = _extract_json_object(raw)
    if obj is not None:
        decision = _normalize_decision(str(obj.get("decision", ""))) or _regex_pick_decision(raw)
        rationale = str(obj.get("rationale", "")).strip()
        if not rationale:
            # try alternative fields
            rationale = str(obj.get("reason", "") or obj.get("explanation", "")).strip()
        if not decision:
            decision = "allow"
        if not rationale:
            rationale = "fallback"
        return {"decision": decision, "rationale": rationale}

    decision_str, rationale = _split_decision_rationale(raw)
    decision = _normalize_decision(decision_str or "") or _regex_pick_decision(raw) or "allow"
    final_rationale = (rationale or "").strip() or "fallback"
    return {"decision": decision, "rationale": final_rationale}
