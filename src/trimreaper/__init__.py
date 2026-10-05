"""TrimReaper: evolutionary MLP width-pruning for Qwen models.

Implements the first proof-of-concept from docs/PLAN.md: test whether a
genetic algorithm can discover sequences of pairwise Givens rotations that
let more complete MLP hidden channels be removed while keeping the model's
next-token distribution within a configurable KL divergence of the original.
"""

__version__ = "0.1.0"
