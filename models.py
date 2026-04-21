"""Typed models for the Auto Prompt Optimizer OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator


TaskFamily = Literal["factual_qa", "json_extraction", "reasoning"]


class TaskSpec(BaseModel):
    task_id: str
    family: TaskFamily
    title: str
    description: str
    input_text: str
    expected_output: Dict[str, Any]
    expected_format: str
    scoring_rubric: Dict[str, str]
    initial_prompt: str
    hints: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptAction(Action):
    """The only controllable action: submit a revised prompt."""

    new_prompt: str = Field(
        ...,
        min_length=3,
        max_length=1200,
        description="A complete replacement prompt for the current task.",
    )

    @field_validator("new_prompt")
    @classmethod
    def validate_prompt_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Prompt cannot be empty.")
        forbidden_fragments = (
            "modify evaluator",
            "change the tests",
            "ignore the score",
            "edit the task file",
            "bypass validation",
        )
        lowered = text.lower()
        for fragment in forbidden_fragments:
            if fragment in lowered:
                raise ValueError("Prompt attempts to tamper with protected system logic.")
        return text


class ScoringBreakdown(BaseModel):
    exact_match: float = 0.0
    format_adherence: float = 0.0
    semantic_score: float = 0.0
    task_completion: float = 0.0
    clarity: float = 0.0
    consistency: float = 0.0
    hallucination_penalty: float = 0.0
    prompt_bloat_penalty: float = 0.0
    improvement_bonus: float = 0.0
    total: float = 0.0


class EvaluationResult(BaseModel):
    model_response: str
    score_breakdown: ScoringBreakdown
    mistakes: List[str] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)
    trial_outputs: List[str] = Field(default_factory=list)


class PromptObservation(Observation):
    task_id: str = ""
    task_family: TaskFamily = "factual_qa"
    task_title: str = ""
    task_description: str = ""
    task_input: str = ""
    expected_format: str = ""
    current_prompt: str = ""
    model_response: str = ""
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    mistakes: List[str] = Field(default_factory=list)
    previous_best_score: float = 0.0
    hints: List[str] = Field(default_factory=list)
    step_count: int = 0


class PromptState(State):
    current_task: Optional[TaskSpec] = None
    current_prompt: str = ""
    previous_prompt_versions: List[str] = Field(default_factory=list)
    model_outputs: List[str] = Field(default_factory=list)
    reward_history: List[float] = Field(default_factory=list)
    best_prompt_so_far: str = ""
    best_score_so_far: float = 0.0
    max_steps: int = 6


class EpisodeSummary(BaseModel):
    episode_id: str
    best_prompt: str
    best_score: float
    reward_history: List[float]
