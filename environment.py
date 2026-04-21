"""Core OpenEnv environment for prompt optimization episodes."""

from __future__ import annotations

from itertools import count
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from .evaluator import evaluate_prompt
    from .models import PromptAction, PromptObservation, PromptState
    from .task_bank import choose_task
except ImportError:  # pragma: no cover
    from evaluator import evaluate_prompt
    from models import PromptAction, PromptObservation, PromptState
    from task_bank import choose_task


class PromptOptimizationEnvironment(Environment):
    """Multi-step environment where actions are prompt rewrites."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    _latest_state: PromptState = PromptState(episode_id=None, step_count=0)

    def __init__(self) -> None:
        self._episode_counter = count()
        self._state = self.__class__._latest_state.model_copy(deep=True)

    @property
    def state(self) -> PromptState:
        return self._state if self._state.current_task is not None else self.__class__._latest_state

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        max_steps: int = 6,
    ) -> PromptObservation:
        del seed
        task = choose_task(task_id, next(self._episode_counter))
        resolved_episode_id = episode_id or str(uuid4())
        self._state = PromptState(
            episode_id=resolved_episode_id,
            step_count=0,
            current_task=task,
            current_prompt=task.initial_prompt,
            best_prompt_so_far=task.initial_prompt,
            max_steps=max_steps,
        )

        evaluation = evaluate_prompt(task=task, prompt=task.initial_prompt, previous_best_score=0.0)
        reward = round(evaluation.score_breakdown.total, 4)
        self._state.model_outputs.append(evaluation.model_response)
        self._state.reward_history.append(reward)
        self._state.best_score_so_far = reward
        self.__class__._latest_state = self._state.model_copy(deep=True)
        return self._build_observation(evaluation=evaluation, reward=reward, done=False)

    def step(self, action: PromptAction) -> PromptObservation:  # type: ignore[override]
        if self._state.current_task is None:
            if self.__class__._latest_state.current_task is None:
                self.reset()
            else:
                self._state = self.__class__._latest_state.model_copy(deep=True)

        previous_best = self._state.best_score_so_far
        self._state.previous_prompt_versions.append(self._state.current_prompt)
        self._state.current_prompt = action.new_prompt
        self._state.step_count += 1

        evaluation = evaluate_prompt(
            task=self._state.current_task,
            prompt=action.new_prompt,
            previous_best_score=previous_best,
        )
        reward = round(evaluation.score_breakdown.total, 4)
        done = self._state.step_count >= self._state.max_steps or reward >= 0.97

        self._state.model_outputs.append(evaluation.model_response)
        self._state.reward_history.append(reward)
        if reward >= self._state.best_score_so_far:
            self._state.best_score_so_far = reward
            self._state.best_prompt_so_far = action.new_prompt
        self.__class__._latest_state = self._state.model_copy(deep=True)

        return self._build_observation(evaluation=evaluation, reward=reward, done=done)

    def _build_observation(
        self,
        evaluation,
        reward: float,
        done: bool,
    ) -> PromptObservation:
        assert self._state.current_task is not None
        return PromptObservation(
            task_id=self._state.current_task.task_id,
            task_family=self._state.current_task.family,
            task_title=self._state.current_task.title,
            task_description=self._state.current_task.description,
            task_input=self._state.current_task.input_text,
            expected_format=self._state.current_task.expected_format,
            current_prompt=self._state.current_prompt,
            model_response=evaluation.model_response,
            score_breakdown=evaluation.score_breakdown.model_dump(),
            mistakes=evaluation.mistakes,
            previous_best_score=round(self._state.best_score_so_far, 4),
            hints=evaluation.hints,
            step_count=self._state.step_count,
            reward=reward,
            done=done,
            metadata={
                "best_prompt_so_far": self._state.best_prompt_so_far,
                "reward_history": self._state.reward_history,
            },
        )
