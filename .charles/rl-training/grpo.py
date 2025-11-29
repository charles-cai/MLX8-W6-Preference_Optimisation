"""
GRPO (Group Relative Policy Optimization) Trainer for Agent0.
Implements Eq 1 from the paper: clipped policy loss with group-relative advantages.

Uses TRL's GRPOTrainer as the foundation, with custom reward functions.
"""

import torch
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
from datasets import Dataset
from loguru import logger


@dataclass
class Agent0GRPOConfig(TRLGRPOConfig):
    """Extended GRPO config for Agent0 with paper-specific defaults."""
    
    # Agent0 paper defaults (Table 8)
    num_generations: int = 4  # k=4 rollouts for self-consistency
    max_new_tokens: int = 4096  # Long-form reasoning
    temperature: float = 1.0  # Sampling temperature
    
    # GRPO clipping (Eq 1)
    epsilon: float = 0.2  # PPO-style clipping
    
    # Training defaults for 0.6B model
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # Effective batch = 8
    learning_rate: float = 1e-6
    num_train_epochs: int = 1
    
    # Memory optimization for hackathon setup
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100


def create_grpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    reward_funcs: List[Callable],
    config: Optional[Agent0GRPOConfig] = None,
    eval_dataset: Optional[Dataset] = None,
) -> GRPOTrainer:
    """
    Create a GRPO trainer with Agent0-specific configuration.
    
    Args:
        model: The policy model (Qwen3-0.6B)
        tokenizer: Tokenizer for the model
        train_dataset: Dataset with 'prompt' column
        reward_funcs: List of reward functions [r1, r2, ...] 
                     Each takes (prompts, completions, **kwargs) -> List[float]
        config: GRPO configuration
        eval_dataset: Optional evaluation dataset
    
    Returns:
        Configured GRPOTrainer instance
    """
    if config is None:
        config = Agent0GRPOConfig(output_dir="./agent0_grpo_output")
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )
    
    logger.info(f"Created GRPO trainer with {len(reward_funcs)} reward functions")
    logger.info(f"Config: k={config.num_generations}, batch={config.per_device_train_batch_size}, "
                f"grad_accum={config.gradient_accumulation_steps}")
    
    return trainer


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int = 4,
) -> torch.Tensor:
    """
    Compute group-relative advantages for GRPO (Eq 1).
    
    For each group of k responses to the same prompt, normalize advantages:
    A_i = (r_i - mean(r)) / (std(r) + eps)
    
    Args:
        rewards: Tensor of shape [batch_size * group_size]
        group_size: Number of generations per prompt (k)
    
    Returns:
        Normalized advantages of same shape
    """
    # Reshape to [num_prompts, group_size]
    num_prompts = rewards.shape[0] // group_size
    rewards_grouped = rewards.view(num_prompts, group_size)
    
    # Compute group statistics
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True)
    
    # Normalize (with stability epsilon)
    advantages = (rewards_grouped - mean) / (std + 1e-8)
    
    # Flatten back
    return advantages.view(-1)


if __name__ == "__main__":
    # Quick sanity check
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
    advantages = compute_group_advantages(rewards, group_size=4)
    print(f"Rewards: {rewards}")
    print(f"Advantages: {advantages}")
    print("✅ GRPO module ready!")