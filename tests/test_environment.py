from rl3.environment import PromptOptimizationEnvironment
from rl3.models import PromptAction


def test_environment_tracks_best_prompt_and_rewards():
    env = PromptOptimizationEnvironment()
    reset_obs = env.reset(task_id="factual_capital_japan", max_steps=3)
    assert reset_obs.task_id == "factual_capital_japan"

    result = env.step(
        PromptAction(
            new_prompt="Use only the provided question. Answer in 1-3 words with only the correct answer. Do not add extra facts."
        )
    )
    assert result.reward >= reset_obs.reward
    assert env.state.best_prompt_so_far == result.current_prompt
    assert len(env.state.reward_history) == 2
