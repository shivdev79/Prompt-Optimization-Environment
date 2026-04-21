"""Baseline inference script compliant with the hackathon submission contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import requests
from openai import OpenAI

try:
    from rl3.client import PromptOptimizerEnv
    from rl3.models import PromptAction, PromptObservation
except ImportError:  # pragma: no cover
    from client import PromptOptimizerEnv
    from models import PromptAction, PromptObservation


API_BASE_URL = os.environ.get("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-4o-mini"
HF_TOKEN = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or ""
ENV_URL = os.environ.get("ENV_URL") or "http://127.0.0.1:8000"
BENCHMARK = "auto-prompt-optimizer"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def list_tasks(base_url: str) -> List[Dict[str, Any]]:
    response = requests.get(f"{base_url.rstrip('/')}/tasks", timeout=30)
    response.raise_for_status()
    return response.json()["tasks"]


def create_openai_client() -> OpenAI:
    if not HF_TOKEN:
        raise RuntimeError("OPENAI_API_KEY or HF_TOKEN must be set before running inference.py")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def compact_action(prompt: str) -> str:
    return json.dumps({"new_prompt": prompt}, ensure_ascii=True, separators=(",", ":"))


def propose_prompt(
    client: OpenAI,
    observation: PromptObservation,
    step_index: int,
    history: List[Dict[str, Any]],
) -> str:
    history_blob = json.dumps(history[-4:], ensure_ascii=True)
    system_prompt = (
        "You are optimizing prompts inside an RL environment. "
        "Return only a JSON object with the field new_prompt. "
        "The prompt must improve correctness, formatting, and stability while staying concise."
    )
    user_prompt = (
        f"Task family: {observation.task_family}\n"
        f"Task title: {observation.task_title}\n"
        f"Task description: {observation.task_description}\n"
        f"Task input: {observation.task_input}\n"
        f"Expected format: {observation.expected_format}\n"
        f"Current prompt: {observation.current_prompt}\n"
        f"Model response: {observation.model_response}\n"
        f"Score breakdown: {json.dumps(observation.score_breakdown, ensure_ascii=True)}\n"
        f"Mistakes: {json.dumps(observation.mistakes, ensure_ascii=True)}\n"
        f"Hints: {json.dumps(observation.hints, ensure_ascii=True)}\n"
        f"Previous best score: {observation.previous_best_score}\n"
        f"History: {history_blob}\n"
        f"Step index: {step_index}\n"
        "Return only JSON like {\"new_prompt\": \"...\"}."
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = (completion.choices[0].message.content or "").strip()
    payload = extract_json_object(content)
    new_prompt = str(payload.get("new_prompt", "")).strip()
    if not new_prompt:
        raise RuntimeError("Model did not return a usable new_prompt.")
    return new_prompt


def extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    index = 0
    while True:
        index = text.find("{", index)
        if index == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text, index)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            index += 1
    raise RuntimeError(f"Could not parse model JSON output: {text[:200]}")


def run_task(base_url: str, task_id: str, max_steps: int) -> None:
    client = create_openai_client()
    env = PromptOptimizerEnv(base_url=base_url).sync()
    rewards: List[float] = []
    history: List[Dict[str, Any]] = []
    score = 0.0
    steps_taken = 0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    with env:
        reset_result = env.reset(task_id=task_id, max_steps=max_steps)
        observation = reset_result.observation
        for step in range(1, max_steps + 1):
            try:
                prompt = propose_prompt(client, observation, step, history)
                action = PromptAction(new_prompt=prompt)
                result = env.step(action)
                reward = float(result.reward or 0.0)
                rewards.append(reward)
                history.append(
                    {
                        "step": step,
                        "reward": reward,
                        "mistakes": observation.mistakes,
                        "hints": observation.hints,
                    }
                )
                observation = result.observation
                steps_taken = step
                score = max(score, reward)
                log_step(step=step, action=compact_action(prompt), reward=reward, done=result.done, error=None)
                if result.done:
                    break
            except Exception as exc:
                log_step(step=step, action="{}", reward=0.0, done=True, error=str(exc).replace(" ", "_"))
                break

    success = score >= 0.8
    log_end(success=success, steps=steps_taken, score=min(max(score, 0.0), 1.0), rewards=rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline optimizer against the OpenEnv server.")
    parser.add_argument("--env-url", default=ENV_URL)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()

    if args.task_id:
        tasks = [args.task_id]
    else:
        tasks = [task["task_id"] for task in list_tasks(args.env_url)[:3]]

    for task_id in tasks:
        run_task(base_url=args.env_url, task_id=task_id, max_steps=args.max_steps)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
        raise SystemExit(str(exc))
