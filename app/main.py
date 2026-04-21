from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.env.transaction_env import TransactionEnvironment
from app.models import Reward, StepResult
from app.routes import build_router
from app.tasks import EasyTask, HardTask, MediumTask


class ResetRequest(BaseModel):
    task_name: str = "anomaly_easy"


class ResetObservationPayload(BaseModel):
    transaction: dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""
    is_done: bool = False


class ResetResponse(BaseModel):
    observation: ResetObservationPayload


class StepResponse(BaseModel):
    observation: dict[str, Any] = Field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str


class StepRequest(BaseModel):
    action_str: str | None = None
    decision: str | None = None
    rationale: str | None = None
    action: str | None = None
    prediction: str | None = None


TASK_REGISTRY = {
    EasyTask.name: EasyTask,
    MediumTask.name: MediumTask,
    HardTask.name: HardTask,
}
TASK_ALIASES = {
    "easy": EasyTask.name,
    "medium": MediumTask.name,
    "hard": HardTask.name,
    EasyTask.name: EasyTask.name,
    MediumTask.name: MediumTask.name,
    HardTask.name: HardTask.name,
}


def _build_environment() -> TransactionEnvironment:
    default_data_dir = Path(__file__).resolve().parent.parent / "data"
    configured_data_dir = os.getenv("TRANSACTION_DATA_DIR", str(default_data_dir))
    return TransactionEnvironment(data_dir=configured_data_dir)


app = FastAPI(title="RL Transaction Environment API", version="1.0.0")
env = _build_environment()
initialized = False


@app.get("/")
def root() -> dict[str, str]:
    return {
        "title": "RL Transaction Environment API",
        "version": "1.0.0",
        "description": "API for transaction anomaly detection environment",
        "endpoints": {
            "health": "/health",
            "reset": "/reset",
            "step": "/step",
            "state": "/state"
        }
    }


def health() -> dict[str, str]:
    return {"status": "ok"}


def _sanitize_none_values(payload: Any) -> Any:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        return {key: _sanitize_none_values(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_none_values(item) for item in payload]
    return payload


def _serialize_step_result(result: StepResult) -> StepResponse:
    observation = _sanitize_none_values(result.observation.model_dump())
    info = _sanitize_none_values(result.info)
    return StepResponse(
        observation=observation,
        reward=float(result.reward.value),
        done=bool(result.done),
        info=info,
    )


def _log_step(result: StepResult, action_str: str) -> None:
    try:
        step_no = int(result.observation.step)
    except Exception:
        step_no = -1

    txn_id = "unknown"
    try:
        info_txn_id = result.info.get("last_transaction_id")
        if info_txn_id:
            txn_id = str(info_txn_id)
        elif result.observation.current_transaction:
            txn_id = str(result.observation.current_transaction.transaction_id)
    except Exception:
        txn_id = "unknown"

    try:
        reward_value = float(result.reward.value)
    except Exception:
        reward_value = 0.0

    action_label = action_str
    try:
        parsed = json.loads(action_str)
        if isinstance(parsed, dict):
            action_label = str(parsed.get("decision") or action_str)
    except Exception:
        action_label = action_str

    print(
        f"[ENV] Step {step_no} | txn={txn_id} | action={action_label} | reward={reward_value}"
    )


def parse_action(raw_input: Any) -> str:
    try:
        decision = "allow"
        rationale = "fallback"
        if raw_input is None:
            return json.dumps({"decision": decision, "rationale": rationale})

        text = raw_input if isinstance(raw_input, str) else str(raw_input)
        lowered = text.lower()
        if "fraud" in lowered or "flag" in lowered:
            decision = "flag_as_fraud"
        elif "allow" in lowered or "approve" in lowered:
            decision = "allow"

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                raw_decision = str(
                    parsed.get("decision")
                    or parsed.get("action")
                    or parsed.get("prediction")
                    or ""
                ).lower()
                raw_rationale = str(parsed.get("rationale") or parsed.get("reason") or "").strip()
                if raw_decision in {"fraud", "flag", "flag_as_fraud", "reject", "deny"}:
                    decision = "flag_as_fraud"
                elif raw_decision in {"allow", "approve", "accept"}:
                    decision = "allow"
                if raw_rationale:
                    rationale = raw_rationale
        except Exception:
            if ":" in text:
                rationale = text[:300]

        return json.dumps({"decision": decision, "rationale": rationale})
    except Exception:
        return json.dumps({"decision": "allow", "rationale": "fallback"})


def _extract_action_payload(action: StepRequest) -> Any:
    if action.action_str and action.action_str.strip():
        return action.action_str
    payload = {
        "decision": action.decision or action.action or action.prediction or "allow",
        "rationale": action.rationale or "fallback",
    }
    return payload


def _normalize_task_name(task_name: str) -> str:
    return TASK_ALIASES.get((task_name or "").strip().lower(), "")


def reset_environment(request: ResetRequest | None = None) -> ResetResponse | ErrorResponse:
    try:
        if request is None:
            request = ResetRequest()
        global initialized
        task_name = _normalize_task_name(request.task_name)
        if not task_name or task_name not in TASK_REGISTRY:
            initialized = False
            return ErrorResponse(
                error="Invalid task. Use anomaly_easy, anomaly_medium, anomaly_hard (or easy/medium/hard aliases)."
            )

        result = env.reset(task_name=task_name)
        if result.done:
            initialized = False
            return ErrorResponse(error="No transactions available for selected task.")

        initialized = True
        current_transaction = result.observation.current_transaction
        transaction_payload = (
            _sanitize_none_values(current_transaction.model_dump()) if current_transaction else {}
        )
        prompt = f"Evaluate the next transaction for task '{result.info.get('task_name', task_name)}'."
        return ResetResponse(
            observation=ResetObservationPayload(
                transaction=transaction_payload,
                prompt=prompt,
                is_done=bool(result.done),
            )
        )
    except Exception:
        initialized = False
        return ErrorResponse(error="Failed to reset environment.")


def step_environment(action: StepRequest | None = None) -> StepResponse | ErrorResponse:
    try:
        if action is None:
            action = StepRequest()
        if not initialized:
            return ErrorResponse(error="Call /reset first")

        current_state = env.state()
        if current_state.current_transaction is None:
            return StepResponse(
                observation={},
                reward=0.0,
                done=True,
                info={"msg": "No more transactions"},
            )

        normalized_action = parse_action(_extract_action_payload(action))
        result = env.step(action_str=normalized_action)
        _log_step(result, normalized_action)
        return _serialize_step_result(result)
    except Exception:
        return ErrorResponse(error="Step execution failed safely.")


def get_state() -> StepResponse:
    try:
        observation = env.state()
        done = observation.current_transaction is None
        result = StepResult(
            observation=observation,
            reward=Reward(value=0.0, reason="state_snapshot"),
            done=done,
            info={"message": "current_state"},
        )
        return _serialize_step_result(result)
    except Exception:
        return StepResponse(
            observation={},
            reward=0.0,
            done=True,
            info={"msg": "State unavailable"},
        )


app.include_router(
    build_router(
        health_handler=health,
        reset_handler=reset_environment,
        step_handler=step_environment,
        state_handler=get_state,
    )
)
