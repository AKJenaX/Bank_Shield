# 🛡️ Bank Shield

**Bank Shield** is a real-world simulation environment for **transaction anomaly triage** — the kind of decision-making performed daily by fraud operations analysts at banks and fintech companies.

Built as an [OpenEnv](https://openenv.ai)-compliant agentic environment, Bank Shield lets AI agents learn to flag, investigate, and resolve suspicious transactions across varying difficulty levels, with dense reward signals and deterministic grading.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [OpenEnv Compliance](#openenv-compliance)
- [Tasks & Difficulty Levels](#tasks--difficulty-levels)
- [Reward Design](#reward-design)
- [Baseline Inference](#baseline-inference)
- [Environment Variables](#environment-variables)
- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [Docker](#docker)

---

## Overview

Bank Shield simulates a fraud operations desk. An agent receives transaction observations and must decide whether to flag, escalate, approve, or decline each case — mimicking real analyst workflows under uncertainty.

The environment is designed for:
- **RL / LLM agent benchmarking** — structured reward signals across easy, medium, and hard scenarios
- **Fraud detection research** — realistic transaction anomaly patterns
- **Agentic evaluation** — multi-step decision making with partial credit and penalties

---

## Project Structure

```
Bank_Shield/
├── agents/           # LLM client and agent logic
├── app/
│   ├── env/          # OpenEnv environment (reset, step, state)
│   ├── models.py     # Pydantic models: Observation, Action, Reward
│   └── main.py       # FastAPI endpoints
├── data/             # Transaction datasets
├── graders/          # EasyGrader, MediumGrader, HardGrader
├── server/           # Server configuration
├── tests/            # Unit tests
├── inference.py      # Baseline agent runner
├── openenv.yaml      # OpenEnv metadata
├── Dockerfile        # Container definition
└── requirements.txt
```

---

## OpenEnv Compliance

Bank Shield follows the [OpenEnv](https://openenv.ai) specification:

| Component | Location |
|---|---|
| Typed Pydantic models (`Observation`, `Action`, `Reward`) | `app/models.py` |
| Environment interface (`reset()`, `step()`, `state()`) | `app/env/base_env.py` |
| REST API (`/reset`, `/step`, `/state`, `/health`) | `app/main.py` |
| Metadata | `openenv.yaml` |

---

## Tasks & Difficulty Levels

Three task tiers are available, each mapped to a dedicated grader:

| Task Name | Grader | Description |
|---|---|---|
| `anomaly_easy` | `EasyGrader` | Clear-cut fraud signals, minimal noise |
| `anomaly_medium` | `MediumGrader` | Mixed signals, requires multi-step reasoning |
| `anomaly_hard` | `HardGrader` | Ambiguous cases, adversarial patterns |

All graders return **deterministic, normalized scores** in the range `[0.0, 1.0]`.

By default, `inference.py` runs all three tasks sequentially.

---

## Reward Design

The reward system is designed to reflect the real cost of fraud decisions:

- **Dense per-step rewards** — feedback at every step, not just at episode end
- **Partial progress** — agents receive credit for correct intermediate reasoning
- **Penalties** — false positives (blocking legit transactions) and missed fraud (false negatives) both carry costs
- **Safe fallback** — if grading fails unexpectedly, a neutral reward of `0.5` is returned

---

## Baseline Inference

`inference.py` provides a ready-to-run baseline agent using an OpenAI-compatible LLM:

- Uses the OpenAI SDK via `agents/llm_client.py`
- Connects to the environment backend via `SPACE_URL`
- Runs easy → medium → hard tasks by default
- Emits structured logs:

```
[START]    Task initialized
[STEP]     Agent action and environment response
[END]      Episode complete with final score
[BASELINE] Aggregate results across all tasks
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | API key for the LLM provider |
| `MODEL_NAME` | ✅ | Model identifier (e.g. `gpt-4o`) |
| `SPACE_URL` | ✅ | URL of the running Bank Shield backend |
| `API_BASE_URL` | Optional | Override for OpenAI-compatible proxy endpoints |
| `TASK_NAME` | Optional | Run a specific task (`anomaly_easy`, `anomaly_medium`, `anomaly_hard`); omit to run all |

---

## Getting Started

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set environment variables**

```bash
export OPENAI_API_KEY=your_key_here
export MODEL_NAME=gpt-4o
export SPACE_URL=http://localhost:7860
```

**3. Start the environment server**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

**4. Run the baseline agent**

```bash
python inference.py
```

To run a single task:

```bash
TASK_NAME=anomaly_hard python inference.py
```

---

## Running Tests

```bash
python -m unittest -q
```

---

## Docker

Build and run the full environment in a container:

```bash
docker build -t bank-shield .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o \
  bank-shield
```

The server will be available at `http://localhost:7860`.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

*Built for the OpenEnv ecosystem — bridging agentic AI research and real-world financial operations.*