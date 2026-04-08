from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agents.agent_utils import format_prompt, parse_action_from_text
from agents.llm_client import LLMClient, get_llm_client


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    score: float
    rewards: List[float]


class ClientEnv:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self._base_url = (base_url or "").strip().rstrip("/")
        if not self._base_url:
            raise ValueError("Missing required env var: SPACE_URL")
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def reset(self, task_name: str) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self._base_url}/reset",
            json={"task_name": task_name},
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self._base_url}/step",
            json=action,
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        return resp.json()


def _load_env_file_if_present() -> None:
    def _parse_env_lines(text: str) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                parsed[key] = value
        return parsed

    root = Path(__file__).resolve().parent
    candidate_files = [root / ".env", root / "env_variables_template.txt"]
    for file_path in candidate_files:
        if not file_path.exists():
            continue
        try:
            values = _parse_env_lines(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key, value in values.items():
            # Keep explicit shell exports as the highest priority.
            if not os.getenv(key):
                os.environ[key] = value

def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _extract_step_fields(step_resp: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Optional[str]]:
    # Backend response shape may vary; handle common keys.
    obs = step_resp.get("observation") or step_resp.get("obs") or step_resp.get("state") or {}
    reward = step_resp.get("reward")
    if reward is None:
        reward = step_resp.get("rewards")
    try:
        reward_f = float(reward) if reward is not None else 0.0
    except Exception:
        reward_f = 0.0

    done = step_resp.get("done")
    if done is None:
        done = step_resp.get("terminated")
    done_b = _as_bool(done)

    err = step_resp.get("error")
    if err is None:
        err = step_resp.get("message") if step_resp.get("status") == "error" else None
    err_s = None if err in (None, "", "null") else str(err)
    return obs if isinstance(obs, dict) else {}, reward_f, done_b, err_s


def run_episode(
    *,
    task_name: str,
    env: ClientEnv,
    llm: LLMClient,
    model_name: str,
    max_steps: int = 200,
    env_name: Optional[str] = None,
) -> EpisodeResult:
    env_label = env_name or os.getenv("SPACE_URL") or "unknown"

    print(f"[START] task={task_name} env={env_label} model={model_name}")

    try:
        reset_resp = env.reset(task_name)
        observation = reset_resp.get("observation") or reset_resp.get("obs") or reset_resp.get("state") or reset_resp
        if not isinstance(observation, dict):
            observation = {"prompt": str(observation)}
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        return EpisodeResult(success=False, steps=0, score=0.0, rewards=[])

    rewards: List[float] = []
    success = True
    done = False

    for step_idx in range(1, max_steps + 1):
        if done:
            break

        prompt = format_prompt(observation)
        messages = [
            {"role": "system", "content": "You are a helpful, precise fraud detection agent."},
            {"role": "user", "content": prompt},
        ]

        action_text = ""
        error: Optional[str] = None
        try:
            completion = llm.chat.completions.create(model=model_name, messages=messages)
            action_text = completion.text
            action = parse_action_from_text(action_text)
        except Exception as e:
            # On LLM errors, default to safe behavior.
            action = {"decision": "allow", "rationale": "fallback"}
            error = str(e)
            success = False

        try:
            step_resp = env.step(action)
            observation, reward, done, env_err = _extract_step_fields(step_resp)
            if env_err:
                error = env_err
                success = False
        except Exception as e:
            reward = 0.0
            done = True
            observation = {}
            error = str(e)
            success = False

        rewards.append(float(reward))

        action_log = json.dumps(action, ensure_ascii=False, sort_keys=True)
        done_log = "true" if done else "false"
        error_log = "null" if not error else json.dumps(error, ensure_ascii=False)
        print(
            f"[STEP] step={step_idx} action={action_log} reward={reward} done={done_log} error={error_log}"
        )

    steps = len(rewards)
    score = sum(rewards) / steps if steps > 0 else 0.0

    print(
        f"[END] success={'true' if success else 'false'} steps={steps} score={score} rewards={json.dumps(rewards)}"
    )
    return EpisodeResult(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    _load_env_file_if_present()

    # Defaults to local Llama 3 Setup unless env variables are specified
    model_name = (os.getenv("MODEL_NAME") or "llama3").strip()
    space_url = (os.getenv("SPACE_URL") or "").strip()

    if not model_name:
        raise ValueError("Missing required env var: MODEL_NAME")
    if not space_url:
        raise ValueError("Missing required env var: SPACE_URL")

    task_name = (os.getenv("TASK_NAME") or "").strip()

    llm = get_llm_client()
    env = ClientEnv(base_url=space_url)
    tasks = [task_name] if task_name else ["anomaly_easy", "anomaly_medium", "anomaly_hard"]

    all_results: Dict[str, EpisodeResult] = {}
    for task in tasks:
        all_results[task] = run_episode(
            task_name=task,
            env=env,
            llm=llm,
            model_name=model_name,
            env_name=space_url,
        )

    baseline = 0.0
    if all_results:
        baseline = sum(v.score for v in all_results.values()) / len(all_results)
    print(f"[BASELINE] tasks={json.dumps(tasks)} mean_score={baseline}")


if __name__ == "__main__":
    main()
