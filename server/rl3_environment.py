"""Backward-compatible import path for the environment class."""

try:
    from rl3.environment import PromptOptimizationEnvironment
except ImportError:  # pragma: no cover
    from environment import PromptOptimizationEnvironment

__all__ = ["PromptOptimizationEnvironment"]
