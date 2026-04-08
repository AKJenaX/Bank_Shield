---
title: BankShield Agent
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# my-openenv-anomaly

Real-world simulation of transaction anomaly triage, a task performed by fraud
operations analysts in banks/fintech teams.

## OpenEnv Compliance

- Typed Pydantic models: `Observation`, `Action`, `Reward` in `app/models.py`
- Environment interface: `reset()`, `step(action)`, `state()` in `app/env/base_env.py`
- API endpoints: `/reset`, `/step`, `/state`, `/health` via `app/main.py`
- Metadata: `openenv.yaml`

## Tasks and Graders

- `anomaly_easy` -> `EasyGrader`
- `anomaly_medium` -> `MediumGrader`
- `anomaly_hard` -> `HardGrader`

All graders return deterministic normalized scores in `[0.0, 1.0]`.

## Reward Behavior

- Dense per-step reward over the full trajectory
- Partial progress rewards (not only terminal binary outcomes)
- Penalties for risky/incorrect decisions (false positives, missed fraud)
- Safe fallback reward `0.5` if grading fails unexpectedly

## Baseline Inference

`inference.py`:
- Uses OpenAI SDK client (`agents/llm_client.py`)
- Reads credentials from `OPENAI_API_KEY`
- Uses `SPACE_URL` for backend API
- Runs all 3 tasks by default (easy/medium/hard)
- Emits `[START]`, `[STEP]`, `[END]`, and `[BASELINE]` logs

## Environment Variables

- `OPENAI_API_KEY` (required)
- `MODEL_NAME` (required)
- `SPACE_URL` (required)
- `API_BASE_URL` (optional for OpenAI-compatible proxy/base URL override)
- `TASK_NAME` (optional; if omitted, runs all 3 tasks)

## Run

```bash
python -m pip install -r requirements.txt
python inference.py
```

## Tests

```bash
python -m unittest -q
```

