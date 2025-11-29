"""
Agent0: Self-Evolving Agents via Tool-Integrated Reasoning
Adapted for Qwen3-0.6B from the original 8B implementation.
"""

from .grpo import create_grpo_trainer, Agent0GRPOConfig
from .rewards import compute_r_unc, compute_r_tool, compute_r_curriculum

__all__ = [
    "create_grpo_trainer",
    "Agent0GRPOConfig",
    "compute_r_unc",
    "compute_r_tool", 
    "compute_r_curriculum",
]
