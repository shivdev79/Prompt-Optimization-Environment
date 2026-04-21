"""OpenEnv WebSocket client for the prompt optimization environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import PromptAction, PromptObservation, PromptState
except ImportError:  # pragma: no cover
    from models import PromptAction, PromptObservation, PromptState


class PromptOptimizerEnv(EnvClient[PromptAction, PromptObservation, PromptState]):
    def _step_payload(self, action: PromptAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PromptObservation]:
        observation = PromptObservation.model_validate(payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PromptState:
        return PromptState.model_validate(payload)
