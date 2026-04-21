from rl3.evaluator import evaluate_prompt
from rl3.task_bank import get_task


def test_factual_prompt_scores_better_with_constraints():
    task = get_task("factual_capital_japan")
    weak = evaluate_prompt(task, "Answer the question.", previous_best_score=0.0)
    strong = evaluate_prompt(
        task,
        "Use only the provided question. Answer in 1-3 words with only the correct answer. Do not add extra facts.",
        previous_best_score=weak.score_breakdown.total,
    )
    assert strong.score_breakdown.total > weak.score_breakdown.total
    assert strong.model_response == "Tokyo"


def test_json_task_rewards_valid_json():
    task = get_task("json_support_ticket")
    result = evaluate_prompt(
        task,
        "Return valid JSON only using double quotes. Include exactly these keys: product, issue, urgency.",
    )
    assert result.score_breakdown.format_adherence == 1.0
    assert result.score_breakdown.task_completion == 1.0

