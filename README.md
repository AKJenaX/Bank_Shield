# Bank Shield

Bank Shield is an elite cybersecurity SaaS dashboard and fraud operations platform. It provides real-time simulation and anomaly triaging designed for financial tech teams to detect, analyze, and neutralize fraudulent transactions.

## Architecture

This project is structured as a modern full-stack application:

- **Frontend (`/frontend`)**: A React application built with Vite and Tailwind CSS. It features a premium, glassmorphism-inspired UI with rich micro-interactions and dynamic event logs.
- **Backend (`/app`)**: A Python FastAPI backend providing robust, stateless operations with in-memory telemetry logging and grading mechanics.

## Prerequisites

- **Node.js**: v18 or later
- **Python**: v3.10 or later
- **uv** (optional, recommended): Fast Python package installer

## Local Development Setup

### 1. Backend Setup

The backend runs on port `8000`.

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Create an environment file if not present
cp .env.example .env

# Start the FastAPI server with hot-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

*Note: Ensure the `API_KEY` in your `.env` matches the `API_KEY` configured in the frontend (defaults to `bank-shield-demo-key`).*

### 2. Frontend Setup

The frontend runs on port `5173`.

```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

Open your browser to `http://localhost:5173` to access the Bank Shield dashboard.

## Features

- **Global Telemetry**: Real-time network health, threat resistance, and bandwidth load monitoring.
- **Simulation Console**: Execute simulated actions ("Allow" or "Flag as Fraud") on anomaly tasks and receive immediate grading.
- **Recent Event Log**: Instantly updating timeline of system events, transaction decisions, and anomalies.
- **Advanced UI Views**: Dedicated panes for Neural Network insights, Threat Vault history, Tactical Map visualization, and System Configurations.

## Testing

To run the backend unit tests and ensure environment grading mechanics are functioning correctly:

```bash
python -m unittest discover tests
```
