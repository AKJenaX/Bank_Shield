from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import logging
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.env.transaction_env import TransactionEnvironment
from app.models import Reward, StepResult
from app.routes import build_router
from app.tasks import EasyTask, HardTask, MediumTask
from app.db import log_telemetry, get_telemetry_stats, get_all_logs

# Set up JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger("bank_shield")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# API Key Auth
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Depends(api_key_header)):
    # If not set in env, default to demo-key for ease of use but normally fail
    expected_key = os.getenv("API_KEY", "bank-shield-demo-key")
    if api_key_header != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key_header



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


class HealthResponse(BaseModel):
    status: str
    title: str = "RL Transaction Environment API"
    version: str = "1.0.0"


class RootResponse(BaseModel):
    status: str
    title: str
    version: str
    message: str


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

# Add SlowAPI exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add Exception Handler for Validation Errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation Error", "details": exc.errors()},
    )

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = _build_environment()
initialized = False


@app.get("/", response_model=RootResponse)
@limiter.limit("60/minute")
def root(request: Request) -> RootResponse:
    """Root endpoint - returns API status and information."""
    try:
        return RootResponse(
            status="running",
            title="RL Transaction Environment API",
            version="1.0.0",
            message="API is operational. Use /docs for interactive API documentation.",
        )
    except Exception as e:
        return RootResponse(
            status="error",
            title="RL Transaction Environment API",
            version="1.0.0",
            message=f"Error: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
def health(request: Request) -> HealthResponse:
    """Health check endpoint - lightweight status report."""
    try:
        return HealthResponse(
            status="healthy",
            title="RL Transaction Environment API",
            version="1.0.0",
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            title="RL Transaction Environment API",
            version="1.0.0",
        )


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

    logger.info(f"Step executed", extra={
        "step": step_no,
        "txn_id": txn_id,
        "action": action_label,
        "reward": reward_value
    })

    # Log to SQLite DB
    fraud_detected = bool(result.observation.flagged) if hasattr(result.observation, 'flagged') else False
    session_id = result.info.get("session_id", "default_session")
    task_name = result.info.get("task_name", "unknown_task")

    log_telemetry(
        session_id=session_id,
        task_name=task_name,
        step_no=step_no,
        transaction_id=txn_id,
        action=action_label,
        reward=reward_value,
        fraud_detected=fraud_detected,
        observation=result.observation.model_dump() if hasattr(result.observation, "model_dump") else {}
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


def reset_environment(request: Request, reset_req: ResetRequest | None = None, api_key: str = Depends(get_api_key)) -> ResetResponse | ErrorResponse:
    try:
        if reset_req is None:
            reset_req = ResetRequest()
        global initialized
        task_name = _normalize_task_name(reset_req.task_name)
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


@limiter.limit("120/minute")
def step_environment(request: Request, action: StepRequest | None = None, api_key: str = Depends(get_api_key)) -> StepResponse | ErrorResponse:
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


@limiter.limit("60/minute")
def get_state(request: Request, api_key: str = Depends(get_api_key)) -> StepResponse:
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

def get_telemetry(request: Request, api_key: str = Depends(get_api_key)):
    """Mock/Real telemetry data for the frontend dashboard."""
    import random
    from datetime import datetime

    db_stats = get_telemetry_stats()
    
    # Process recent logs into UI format
    events = []
    for log in db_stats["recent_logs"]:
        ts_str, action, fraud, txn = log
        try:
            ts = datetime.fromisoformat(ts_str)
            mins_ago = int((datetime.utcnow() - ts).total_seconds() / 60)
            time_ago = f"{mins_ago}m ago" if mins_ago > 0 else "Just now"
        except Exception:
            time_ago = "Unknown"
            
        events.append({
            "title": f"Action: {action.upper()}",
            "description": f"Txn: {txn}",
            "time_ago": time_ago,
            "type": "warning" if fraud else "success"
        })
        
    if not events:
        events = [
            {
                "title": "Node Cluster Verification",
                "description": "All 24 secondary nodes synchronized successfully.",
                "time_ago": "2m ago",
                "type": "success"
            }
        ]

    return {
        "global_telemetry": {
            "neural_health": round(random.uniform(98.0, 99.9), 1),
            "threat_resistance": round(random.uniform(80.0, 95.0), 1),
            "bandwidth_load": round(random.uniform(30.0, 60.0), 1),
        },
        "simulation_console": {
            "request_volume": f"{db_stats['total_requests'] + round(random.uniform(100, 150), 1)}k",
            "threats_neutralized": db_stats["threats_neutralized"],
            "engine_load": round(random.uniform(20.0, 40.0), 1),
            "engine_confidence": random.randint(90, 98),
        },
        "header_status": {
            "status": "Nominal",
            "sync": "100%",
            "latency": f"{random.randint(8, 24)}ms",
        },
        "recent_events": events
    }

def get_logs(request: Request, fraud_only: bool = False, api_key: str = Depends(get_api_key)):
    """Fetch logs from DB."""
    return get_all_logs(fraud_only)

# Create specific routes for Limiter since it requires the request object
app.add_api_route("/reset", reset_environment, methods=["POST"])
app.add_api_route("/step", step_environment, methods=["POST"])
app.add_api_route("/state", get_state, methods=["GET"])
app.add_api_route("/api/logs", get_logs, methods=["GET"])
app.add_api_route("/api/telemetry", get_telemetry, methods=["GET"])
