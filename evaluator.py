"""Deterministic evaluator and task engine for prompt optimization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    from .models import EvaluationResult, ScoringBreakdown, TaskSpec
except ImportError:  # pragma: no cover
    from models import EvaluationResult, ScoringBreakdown, TaskSpec


BREVITY_TOKENS = (
    "concise",
    "brief",
    "only the answer",
    "only the correct answer",
    "one line",
    "1 sentence",
    "1-3 words",
    "1 to 3 words",
)
GROUNDING_TOKENS = ("use only", "do not hallucinate", "if missing", "use the provided input")
JSON_TOKENS = ("json", "valid json", "double quotes")
STEP_TOKENS = ("step by step", "numbered", "final answer", "show your reasoning")
EXAMPLE_TOKENS = ("example", "for example", "few-shot")


@dataclass(frozen=True)
class PromptFeatures:
    brevity: bool
    grounding: bool
    json_mode: bool
    stepwise: bool
    examples: bool
    strong_constraints: int
    contradictory: bool
    prompt_length: int


def extract_prompt_features(prompt: str) -> PromptFeatures:
    lowered = prompt.lower()
    brevity = any(token in lowered for token in BREVITY_TOKENS)
    grounding = any(token in lowered for token in GROUNDING_TOKENS)
    json_mode = any(token in lowered for token in JSON_TOKENS)
    stepwise = any(token in lowered for token in STEP_TOKENS)
    examples = any(token in lowered for token in EXAMPLE_TOKENS)
    strong_constraints = sum(lowered.count(token) for token in ("must", "exactly", "only", "do not", "return"))
    contradictory = (
        ("only the answer" in lowered and "step by step" in lowered)
        or ("json" in lowered and "paragraph" in lowered)
        or ("concise" in lowered and "detailed" in lowered)
    )
    return PromptFeatures(
        brevity=brevity,
        grounding=grounding,
        json_mode=json_mode,
        stepwise=stepwise,
        examples=examples,
        strong_constraints=strong_constraints,
        contradictory=contradictory,
        prompt_length=len(prompt),
    )


def evaluate_prompt(task: TaskSpec, prompt: str, previous_best_score: float = 0.0, trials: int = 2) -> EvaluationResult:
    trial_outputs = [generate_model_output(task, prompt, trial_index) for trial_index in range(trials)]
    primary_output = trial_outputs[0]
    score_breakdown, mistakes, hints = score_output(
        task=task,
        prompt=prompt,
        primary_output=primary_output,
        trial_outputs=trial_outputs,
        previous_best_score=previous_best_score,
    )
    return EvaluationResult(
        model_response=primary_output,
        score_breakdown=score_breakdown,
        mistakes=mistakes,
        hints=hints,
        trial_outputs=trial_outputs,
    )


def generate_model_output(task: TaskSpec, prompt: str, trial_index: int) -> str:
    features = extract_prompt_features(prompt)
    if task.family == "factual_qa":
        return _generate_factual_output(task, features, trial_index)
    if task.family == "json_extraction":
        return _generate_json_output(task, features, trial_index)
    return _generate_reasoning_output(task, features, trial_index)


def score_output(
    task: TaskSpec,
    prompt: str,
    primary_output: str,
    trial_outputs: List[str],
    previous_best_score: float,
) -> Tuple[ScoringBreakdown, List[str], List[str]]:
    if task.family == "factual_qa":
        return _score_factual(task, prompt, primary_output, trial_outputs, previous_best_score)
    if task.family == "json_extraction":
        return _score_json(task, prompt, primary_output, trial_outputs, previous_best_score)
    return _score_reasoning(task, prompt, primary_output, trial_outputs, previous_best_score)


def _generate_factual_output(task: TaskSpec, features: PromptFeatures, trial_index: int) -> str:
    answer = task.expected_output["answer"]
    verbose_answer = f"The answer is {answer}."
    hallucinated = f"{verbose_answer} It is also the largest city in Europe."
    if features.contradictory:
        return answer if trial_index == 0 else verbose_answer
    if features.brevity and features.grounding and features.strong_constraints >= 2:
        return answer
    if features.brevity:
        return verbose_answer
    if features.grounding:
        return f"{answer}, based on the provided facts."
    return hallucinated


def _generate_json_output(task: TaskSpec, features: PromptFeatures, trial_index: int) -> str:
    expected = task.expected_output
    if features.contradictory:
        if trial_index == 0:
            return json.dumps(expected)
        return f"Here is the JSON: {json.dumps(expected)}"
    if features.json_mode and features.strong_constraints >= 2:
        return json.dumps(expected, ensure_ascii=True, separators=(",", ":"))
    if features.json_mode:
        partial = {key: value for index, (key, value) in enumerate(expected.items()) if index < max(1, len(expected) - 1)}
        return json.dumps(partial, ensure_ascii=True)
    return f"Customer issue summary: {expected}"


def _generate_reasoning_output(task: TaskSpec, features: PromptFeatures, trial_index: int) -> str:
    answer = task.expected_output["final_answer"]
    steps = task.expected_output["steps"]
    compact = f"Final answer: {answer}"
    detailed = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps)) + f"\nFinal answer: {answer}"
    wrong = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps[:-1])) + "\nFinal answer: 11"
    if features.contradictory:
        return detailed if trial_index == 0 else compact
    if features.stepwise and features.strong_constraints >= 2:
        return detailed
    if features.stepwise:
        return detailed
    if features.brevity:
        return compact
    return wrong


def _score_factual(
    task: TaskSpec,
    prompt: str,
    output: str,
    trial_outputs: List[str],
    previous_best_score: float,
) -> Tuple[ScoringBreakdown, List[str], List[str]]:
    answer = task.expected_output["answer"].lower()
    aliases = [candidate.lower() for candidate in task.expected_output.get("aliases", [])]
    normalized = _normalize(output)
    exact_match = 1.0 if normalized in {answer, *aliases} else 0.0
    contains_answer = 1.0 if answer in normalized or any(alias in normalized for alias in aliases) else 0.0
    brevity_limit = task.metadata.get("max_words", 6)
    word_count = len(output.split())
    format_adherence = 1.0 if word_count <= brevity_limit else 0.35 if contains_answer else 0.0
    hallucination_penalty = 0.2 if "largest city in europe" in normalized else 0.0
    clarity = 1.0 if "based on the provided facts" not in normalized else 0.8
    consistency = 1.0 if len(set(trial_outputs)) == 1 else 0.55
    prompt_bloat_penalty = _prompt_bloat_penalty(prompt)
    total = (
        exact_match * 0.35
        + format_adherence * 0.2
        + contains_answer * 0.2
        + clarity * 0.1
        + consistency * 0.1
        - hallucination_penalty
        - prompt_bloat_penalty
    )
    total = max(0.0, min(1.0, total))
    improvement_bonus = max(0.0, total - previous_best_score) * 0.2
    total = min(1.0, total + improvement_bonus)
    mistakes: List[str] = []
    hints: List[str] = list(task.hints)
    if not exact_match:
        mistakes.append("The answer is not an exact match.")
        hints.append("Ask for the shortest possible factual answer.")
    if format_adherence < 1.0:
        mistakes.append("The answer is too verbose for a concise QA task.")
        hints.append("Add a strict length limit such as 'answer in 1-3 words'.")
    if hallucination_penalty:
        mistakes.append("The answer includes unsupported extra facts.")
        hints.append("Explicitly forbid unsupported details or speculation.")
    breakdown = ScoringBreakdown(
        exact_match=exact_match,
        format_adherence=format_adherence,
        semantic_score=contains_answer,
        task_completion=contains_answer,
        clarity=clarity,
        consistency=consistency,
        hallucination_penalty=hallucination_penalty,
        prompt_bloat_penalty=prompt_bloat_penalty,
        improvement_bonus=improvement_bonus,
        total=round(total, 4),
    )
    return breakdown, _dedupe(mistakes), _dedupe(hints)


def _score_json(
    task: TaskSpec,
    prompt: str,
    output: str,
    trial_outputs: List[str],
    previous_best_score: float,
) -> Tuple[ScoringBreakdown, List[str], List[str]]:
    expected = task.expected_output
    parsed, valid_json = _try_parse_json(output)
    matched_fields = 0
    if valid_json:
        matched_fields = sum(1 for key, value in expected.items() if parsed.get(key) == value)
    field_ratio = matched_fields / len(expected)
    exact_match = 1.0 if valid_json and parsed == expected else 0.0
    format_adherence = 1.0 if valid_json else 0.0
    semantic_score = field_ratio
    task_completion = field_ratio
    clarity = 1.0 if output.strip().startswith("{") and output.strip().endswith("}") else 0.3
    consistency = 1.0 if len(set(trial_outputs)) == 1 else 0.6
    hallucination_penalty = 0.0 if valid_json else 0.15
    prompt_bloat_penalty = _prompt_bloat_penalty(prompt)
    total = (
        exact_match * 0.3
        + format_adherence * 0.2
        + semantic_score * 0.2
        + task_completion * 0.15
        + clarity * 0.1
        + consistency * 0.1
        - hallucination_penalty
        - prompt_bloat_penalty
    )
    total = max(0.0, min(1.0, total))
    improvement_bonus = max(0.0, total - previous_best_score) * 0.2
    total = min(1.0, total + improvement_bonus)
    mistakes: List[str] = []
    hints: List[str] = list(task.hints)
    if not valid_json:
        mistakes.append("Output is not valid JSON.")
        hints.append("Require 'valid JSON only' and forbid commentary.")
    missing = [key for key in expected if not valid_json or key not in parsed]
    if missing:
        mistakes.append(f"Missing required fields: {', '.join(missing)}.")
        hints.append(f"List the required keys explicitly: {', '.join(expected.keys())}.")
    breakdown = ScoringBreakdown(
        exact_match=exact_match,
        format_adherence=format_adherence,
        semantic_score=semantic_score,
        task_completion=task_completion,
        clarity=clarity,
        consistency=consistency,
        hallucination_penalty=hallucination_penalty,
        prompt_bloat_penalty=prompt_bloat_penalty,
        improvement_bonus=improvement_bonus,
        total=round(total, 4),
    )
    return breakdown, _dedupe(mistakes), _dedupe(hints)


def _score_reasoning(
    task: TaskSpec,
    prompt: str,
    output: str,
    trial_outputs: List[str],
    previous_best_score: float,
) -> Tuple[ScoringBreakdown, List[str], List[str]]:
    answer = str(task.expected_output["final_answer"]).lower()
    steps = task.expected_output["steps"]
    normalized = output.lower()
    exact_match = 1.0 if f"final answer: {answer}" in normalized else 0.0
    numbered_steps = len(re.findall(r"^\d+\.", output, flags=re.MULTILINE))
    format_adherence = 1.0 if numbered_steps >= len(steps) and "final answer:" in normalized else 0.4 if "final answer:" in normalized else 0.0
    semantic_score = 1.0 if exact_match else 0.35 if any(str(step).lower()[:12] in normalized for step in steps) else 0.0
    task_completion = 1.0 if exact_match else 0.25
    clarity = 1.0 if output.strip().splitlines()[-1].lower().startswith("final answer:") else 0.5
    consistency = 1.0 if len(set(trial_outputs)) == 1 else 0.55
    hallucination_penalty = 0.15 if "11" in normalized and answer != "11" else 0.0
    prompt_bloat_penalty = _prompt_bloat_penalty(prompt)
    total = (
        exact_match * 0.3
        + format_adherence * 0.2
        + semantic_score * 0.15
        + task_completion * 0.15
        + clarity * 0.1
        + consistency * 0.1
        - hallucination_penalty
        - prompt_bloat_penalty
    )
    total = max(0.0, min(1.0, total))
    improvement_bonus = max(0.0, total - previous_best_score) * 0.2
    total = min(1.0, total + improvement_bonus)
    mistakes: List[str] = []
    hints: List[str] = list(task.hints)
    if not exact_match:
        mistakes.append("The final answer is incorrect or missing.")
        hints.append("Require a numbered reasoning trace followed by a final answer line.")
    if format_adherence < 1.0:
        mistakes.append("The reasoning output does not follow the requested step format.")
        hints.append("Specify an exact output template with numbered steps.")
    breakdown = ScoringBreakdown(
        exact_match=exact_match,
        format_adherence=format_adherence,
        semantic_score=semantic_score,
        task_completion=task_completion,
        clarity=clarity,
        consistency=consistency,
        hallucination_penalty=hallucination_penalty,
        prompt_bloat_penalty=prompt_bloat_penalty,
        improvement_bonus=improvement_bonus,
        total=round(total, 4),
    )
    return breakdown, _dedupe(mistakes), _dedupe(hints)


def _try_parse_json(text: str) -> Tuple[Dict[str, Any], bool]:
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return {}, False


def _prompt_bloat_penalty(prompt: str) -> float:
    overflow = max(0, len(prompt) - 320)
    return min(0.15, overflow / 1000)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped
