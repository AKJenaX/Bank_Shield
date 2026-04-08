from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.env.base_env import BaseEnvironment
from app.models import Observation, Reward, StepResult, Transaction
from graders.base_grader import BaseGrader
from graders.easy_grader import EasyGrader
from graders.hard_grader import HardGrader
from graders.medium_grader import MediumGrader
from graders.reward_utils import normalize_reward


class TransactionEnvironment(BaseEnvironment):
    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._transactions: list[dict[str, Any]] = []
        self._dataset_name = "anomaly_easy"
        self._episode_id = ""
        self._step = 0
        self._history: list[dict[str, Any]] = []
        self._graders: dict[str, BaseGrader] = {
            "anomaly_easy": EasyGrader(),
            "anomaly_medium": MediumGrader(),
            "anomaly_hard": HardGrader(),
        }

    @staticmethod
    def _normalize_task(task_name: str) -> str:
        mapping = {
            "easy": "anomaly_easy",
            "medium": "anomaly_medium",
            "hard": "anomaly_hard",
            "anomaly_easy": "anomaly_easy",
            "anomaly_medium": "anomaly_medium",
            "anomaly_hard": "anomaly_hard",
        }
        normalized = (task_name or "").strip().lower()
        return mapping.get(normalized, "")

    def reset(self, task_name: str = "easy") -> StepResult:
        normalized_task = self._normalize_task(task_name) or "anomaly_easy"

        self._dataset_name = normalized_task
        self._transactions = self._load_transactions(self._dataset_name)
        self._episode_id = str(uuid4())
        self._step = 0
        self._history = []

        return StepResult(
            observation=self._build_observation(),
            reward=Reward(value=0.0, reason="environment_reset"),
            done=self._is_done(),
            info={
                "message": "episode_reset",
                "task_name": self._dataset_name,
                "transactions_loaded": len(self._transactions),
            },
        )

    def step(self, action_str: str) -> StepResult:
        try:
            if not self._episode_id:
                return StepResult(
                    observation=self._build_observation(),
                    reward=Reward(value=0.0, reason="episode_not_initialized"),
                    done=True,
                    info={"message": "Call /reset first", "task_name": self._dataset_name},
                )

            if self._is_done():
                return StepResult(
                    observation=self._build_observation(),
                    reward=Reward(value=0.0, reason="episode_already_done"),
                    done=True,
                    info={"message": "no_more_steps"},
                )

            parsed_action = self._parse_action(action_str)
            txn = self._transactions[self._step] if self._step < len(self._transactions) else {}
            reward_value = self.compute_reward(parsed_action, txn)
            reward_reason = "graded" if reward_value != 0.5 else "fallback_reward"

            self._history.append(
                {
                    "step": self._step,
                    "transaction_id": str(txn.get("id") or txn.get("transaction_id") or "unknown"),
                    "action_str": action_str,
                    "parsed_action": parsed_action,
                    "reward": reward_value,
                }
            )
            self._step += 1

            return StepResult(
                observation=self._build_observation(),
                reward=Reward(value=reward_value, reason=reward_reason),
                done=self._is_done(),
                info={
                    "remaining_steps": max(len(self._transactions) - self._step, 0),
                    "last_transaction_id": str(txn.get("id") or txn.get("transaction_id") or "unknown"),
                    "task_name": self._dataset_name,
                },
            )
        except Exception:
            return StepResult(
                observation=self._build_observation(),
                reward=Reward(value=0.5, reason="fallback_reward"),
                done=self._is_done(),
                info={"message": "step_failed_safely", "task_name": self._dataset_name},
            )

    def state(self) -> Observation:
        if not self._episode_id:
            self._episode_id = str(uuid4())
        return self._build_observation()

    def _build_observation(self) -> Observation:
        current_txn = None
        if self._step < len(self._transactions):
            current_txn = self._to_transaction_model(self._transactions[self._step])

        return Observation(
            episode_id=self._episode_id,
            step=self._step,
            task_id=self._dataset_name,
            current_transaction=current_txn,
            history=self._history,
        )

    def _is_done(self) -> bool:
        return self._step >= len(self._transactions)

    @staticmethod
    def _parse_action(action_str: str) -> dict[str, Any]:
        text = (action_str or "").strip()
        if not text:
            return {"decision": "allow", "rationale": "fallback"}

        lowered = text.lower()
        decision = "allow"
        if "flag_as_fraud" in lowered or "fraud" in lowered or "flag" in lowered:
            decision = "flag"
        elif "allow" in lowered or "approve" in lowered:
            decision = "allow"

        rationale = "fallback"
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                raw_decision = str(parsed.get("decision") or parsed.get("action") or "").strip().lower()
                raw_rationale = str(parsed.get("rationale") or parsed.get("reason") or "").strip()
                if raw_decision in {"fraud", "flag", "flag_as_fraud", "deny", "reject"}:
                    decision = "flag"
                elif raw_decision in {"allow", "approve", "accept"}:
                    decision = "allow"
                rationale = raw_rationale or rationale
        except Exception:
            if ":" in text:
                rationale = text

        return {"decision": decision, "rationale": rationale}

    def compute_reward(self, action: dict[str, Any], transaction: dict[str, Any]) -> float:
        try:
            grader = self._graders[self._dataset_name]
            label = str(transaction.get("true_label", "normal"))
            result = grader.evaluate_step(action=action, transaction=transaction, true_label=label)
            return float(normalize_reward(result.get("score", 0.5)))
        except Exception:
            return 0.5

    def _load_transactions(self, task_name: str) -> list[dict[str, Any]]:
        transactions: list[dict[str, Any]] = []
        if not self._data_dir.exists():
            return transactions

        filename_mapping = {
            "anomaly_easy": ["transactions_easy.json", "easy.json"],
            "anomaly_medium": ["transactions_medium.json", "medium.json"],
            "anomaly_hard": ["transactions_hard.json", "hard.json"],
        }
        file_path = None
        for candidate in filename_mapping.get(task_name, []):
            candidate_path = self._data_dir / candidate
            if candidate_path.exists():
                file_path = candidate_path
                break
        if file_path is None:
            return transactions

        try:
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            return transactions

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return transactions

        for item in payload:
            if not isinstance(item, dict):
                continue
            txn_id = item.get("id") or item.get("transaction_id")
            amount = item.get("amount")
            true_label = item.get("true_label")
            if txn_id is None or amount is None or true_label is None:
                continue
            try:
                float(amount)
            except (TypeError, ValueError):
                continue
            transactions.append(item)

        return transactions

    @staticmethod
    def _to_transaction_model(raw_txn: dict[str, Any]) -> Transaction:
        metadata = dict(raw_txn.get("metadata", {})) if isinstance(raw_txn.get("metadata"), dict) else {}
        for key, value in raw_txn.items():
            if key not in {"transaction_id", "id", "amount", "merchant", "category", "timestamp", "metadata"}:
                metadata[key] = value
        txn_payload = {
            "transaction_id": str(raw_txn.get("transaction_id") or raw_txn.get("id") or "unknown"),
            "amount": float(raw_txn.get("amount", 0.0)),
            "merchant": raw_txn.get("merchant"),
            "category": raw_txn.get("category"),
            "timestamp": raw_txn.get("timestamp"),
            "metadata": metadata,
        }
        try:
            return Transaction.model_validate(txn_payload)
        except Exception:
            txn_payload["timestamp"] = None
            return Transaction.model_validate(txn_payload)
