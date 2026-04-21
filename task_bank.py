"""Task bank loading utilities for the Auto Prompt Optimizer."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .models import TaskSpec
except ImportError:  # pragma: no cover
    from models import TaskSpec


DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_tasks.json"


@lru_cache(maxsize=1)
def load_task_bank() -> List[TaskSpec]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [TaskSpec.model_validate(item) for item in payload["tasks"]]


def get_task(task_id: str) -> TaskSpec:
    for task in load_task_bank():
        if task.task_id == task_id:
            return task
    raise KeyError(f"Unknown task_id: {task_id}")


def list_task_summaries() -> List[Dict[str, str]]:
    return [
        {
            "task_id": task.task_id,
            "family": task.family,
            "title": task.title,
            "description": task.description,
        }
        for task in load_task_bank()
    ]


def choose_task(task_id: Optional[str], episode_index: int) -> TaskSpec:
    if task_id:
        return get_task(task_id)
    tasks = load_task_bank()
    return tasks[episode_index % len(tasks)]
