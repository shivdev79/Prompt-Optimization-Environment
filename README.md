---
title: Promtoptimizer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Prompt-Optimization-Environment

Auto Prompt Optimizer is a hackathon-ready OpenEnv environment where an agent iteratively improves prompts for a fixed task family. Each episode starts with a weak prompt, the agent rewrites it, the environment scores the resulting model output, and reward improves over multiple steps.

## Compliance highlights

- OpenEnv-native `reset` / `step` / `state` endpoints via `openenv-core`
- Typed `Action`, `Observation`, and `State` models built on Pydantic
- Root-level `inference.py`
- Inference uses `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- Inference uses the OpenAI client for LLM prompt proposal
- Structured stdout logs in the required `[START]`, `[STEP]`, `[END]` format
- Dockerized FastAPI server for local runs and HF Spaces

## Folder structure

```text
rl3/
|-- README.md
|-- Dockerfile
|-- pyproject.toml
|-- requirements.txt
|-- openenv.yaml
|-- __init__.py
|-- api.py
|-- client.py
|-- environment.py
|-- evaluator.py
|-- inference.py
|-- models.py
|-- task_bank.py
|-- data/
|   |-- sample_tasks.json
|-- server/
|   |-- __init__.py
|   |-- app.py
|   |-- rl3_environment.py
|-- tests/
|   |-- conftest.py
|   |-- test_api.py
|   |-- test_environment.py
|   |-- test_evaluator.py
|   |-- test_inference_contract.py
|-- validate_submission.py
```

## Required environment variables

- `API_BASE_URL`: OpenAI-compatible base URL
- `MODEL_NAME`: model name for prompt proposal
- `HF_TOKEN`: API key used by the OpenAI client
- `ENV_URL`: optional environment server URL, default `http://127.0.0.1:8000`

## Quick start

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Second terminal:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_api_key
set ENV_URL=http://127.0.0.1:8000
python inference.py --task-id factual_capital_japan
```

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /tasks`
- `GET /health`
- `WS /ws`

## Task families

- `factual_qa`
- `json_extraction`
- `reasoning`

The sample task bank includes 6 tasks across these three families.

## Validation

```bash
pytest
python validate_submission.py
```
