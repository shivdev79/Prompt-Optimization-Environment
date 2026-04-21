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

This is a real-world productivity simulation rather than a game: prompt engineering for production-style NLP tasks such as factual answering, extracting structured business data, and following multi-step instructions with strict formatting requirements.

## Compliance highlights

- OpenEnv-native `reset` / `step` / `state` endpoints via `openenv-core`
- Typed `Action`, `Observation`, and `State` models built on Pydantic
- Root-level `inference.py`
- Inference uses `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- Inference uses the OpenAI client for LLM prompt proposal
- Structured stdout logs in the required `[START]`, `[STEP]`, `[END]` format
- Dockerized FastAPI server for local runs and HF Spaces

## Environment description and motivation

This environment simulates the practical workflow of improving prompts for real task families humans actually care about:

- concise factual QA for assistant-style lookup tasks
- structured JSON extraction for support and operations workflows
- reasoning and instruction-following for process automation

The agent does not solve the downstream task directly. Instead, it acts like a prompt engineer, rewriting prompts over multiple steps and using reward feedback to discover stronger instructions. That makes the environment useful for RL experiments on iterative prompt refinement, self-improvement loops, and prompt policy search.

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
- `OPENAI_API_KEY`: primary API key used by the OpenAI client
- `HF_TOKEN`: optional compatibility alias for the API key
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
set OPENAI_API_KEY=your_api_key
set ENV_URL=http://127.0.0.1:8000
python inference.py --task-id factual_capital_japan
```

## Action, observation, and state spaces

### Action space

`PromptAction`

- `new_prompt`: complete replacement prompt proposed by the agent

This supports rewriting instructions, adding constraints, changing formatting requirements, introducing examples, and simplifying wording.

### Observation space

`PromptObservation`

- `task_id`, `task_family`, `task_title`, `task_description`
- `task_input`
- `expected_format`
- `current_prompt`
- `model_response`
- `score_breakdown`
- `mistakes`
- `previous_best_score`
- `hints`
- `step_count`
- inherited OpenEnv fields: `reward`, `done`, `metadata`

### State space

`PromptState`

- `episode_id`
- `step_count`
- `current_task`
- `current_prompt`
- `previous_prompt_versions`
- `model_outputs`
- `reward_history`
- `best_prompt_so_far`
- `best_score_so_far`
- `max_steps`

## Reward function

The reward is dense and normalized into the `0.0` to `1.0` range.

Positive components:

- exact match
- format adherence
- semantic correctness
- task completion
- clarity
- consistency across repeated deterministic trials
- improvement bonus over the previous best score

Penalties:

- hallucinations
- invalid JSON or missing fields
- incorrect final answers
- prompt bloat
- unstable prompts across trials

This gives useful shaping over the full trajectory rather than only binary success at the end.

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

## Task descriptions and expected difficulty

Easy:

- `factual_capital_japan`: answer a short geography question with strict brevity
- `factual_red_planet`: answer a simple science fact with no extra text

Medium:

- `json_support_ticket`: extract `product`, `issue`, and `urgency` from a support message
- `json_invoice_parse`: extract structured invoice fields into valid JSON

Hard:

- `reasoning_ticket_packs`: solve arithmetic with numbered reasoning and final answer formatting
- `reasoning_minutes_math`: execute multiple arithmetic transformations with exact output structure

Each task includes a deterministic grader and returns a reward in the `0.0` to `1.0` range.

## Baseline inference

`inference.py` uses the OpenAI client against the deployed or local OpenEnv server and emits structured logs in `[START]`, `[STEP]`, and `[END]` format.

Representative baseline results from local validation:

- `factual_capital_japan`: initial reward `0.32`, optimized reward about `0.95-1.00`
- `json_support_ticket`: initial reward `0.00`, optimized reward improves once valid JSON constraints are introduced
- `reasoning_ticket_packs`: initial reward about `0.26`, optimized reward improves when numbered reasoning plus `Final answer:` formatting is enforced

Because the evaluator is deterministic, the same task and same prompt sequence produce reproducible scores.

## Validation

```bash
pytest
python validate_submission.py
```

## Deployment

### Docker

```bash
docker build -t auto-prompt-optimizer .
docker run --rm -p 8000:8000 auto-prompt-optimizer
```

### Hugging Face Spaces

- Create a Docker Space
- Add `API_BASE_URL`, `MODEL_NAME`, and `OPENAI_API_KEY` or `HF_TOKEN` in Space settings
- Push the repo root to the Space remote
- Verify `/`, `/health`, `/docs`, `/schema`, and `/tasks`
