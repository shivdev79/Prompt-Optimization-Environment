"""FastAPI application exposing the environment through OpenEnv endpoints."""

from __future__ import annotations

from typing import Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

try:
    from .environment import PromptOptimizationEnvironment
    from .models import PromptAction, PromptObservation
    from .task_bank import list_task_summaries
except ImportError:  # pragma: no cover
    from environment import PromptOptimizationEnvironment
    from models import PromptAction, PromptObservation
    from task_bank import list_task_summaries


app = create_app(
    PromptOptimizationEnvironment,
    PromptAction,
    PromptObservation,
    env_name="auto-prompt-optimizer",
    max_concurrent_envs=4,
)


@app.get("/tasks", tags=["Tasks"])
def list_tasks() -> Dict[str, list]:
    return {"tasks": list_task_summaries()}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    del request
    return JSONResponse(status_code=500, content={"detail": str(exc)})
