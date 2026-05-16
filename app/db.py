from datetime import datetime
import json
from typing import Any

# In-memory storage for telemetry logs
_telemetry_logs = []
_id_counter = 1

def init_db():
    """Initialize the in-memory database."""
    global _telemetry_logs, _id_counter
    _telemetry_logs = []
    _id_counter = 1

def log_telemetry(
    session_id: str,
    task_name: str,
    step_no: int,
    transaction_id: str,
    action: str,
    reward: float,
    fraud_detected: bool,
    observation: dict[str, Any]
):
    """Log an interaction step to the in-memory database."""
    global _id_counter
    try:
        log_entry = {
            "id": _id_counter,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "task_name": task_name,
            "step_no": step_no,
            "transaction_id": transaction_id,
            "action": action,
            "reward": reward,
            "fraud_detected": int(fraud_detected),
            "observation": json.dumps(observation, default=str)
        }
        _telemetry_logs.append(log_entry)
        _id_counter += 1
    except Exception as e:
        print(f"[DB] Error logging telemetry: {e}")

def get_telemetry_stats() -> dict[str, Any]:
    """Retrieve aggregate stats from in-memory telemetry DB."""
    try:
        total_requests = len(_telemetry_logs)
        threats_neutralized = sum(1 for log in _telemetry_logs if log["action"] == 'flag_as_fraud')
        
        recent_logs = []
        for log in reversed(_telemetry_logs[-5:]):
            recent_logs.append((log["timestamp"], log["action"], log["fraud_detected"], log["transaction_id"]))
            
        return {
            "total_requests": total_requests,
            "threats_neutralized": threats_neutralized,
            "recent_logs": recent_logs
        }
    except Exception as e:
        print(f"[DB] Error getting stats: {e}")
        return {
            "total_requests": 0,
            "threats_neutralized": 0,
            "recent_logs": []
        }

def get_all_logs(fraud_only: bool = False) -> list[dict[str, Any]]:
    """Retrieve all logs, optionally filtered by fraud_detected."""
    try:
        if fraud_only:
            filtered = [
                log for log in _telemetry_logs 
                if log["fraud_detected"] == 1 or log["action"] in ('flag_as_fraud', 'flag')
            ]
        else:
            filtered = list(_telemetry_logs)
            
        # Order by id DESC LIMIT 500
        filtered.reverse()
        return filtered[:500]
    except Exception as e:
        print(f"[DB] Error getting all logs: {e}")
        return []

# Initialize DB on import
init_db()
