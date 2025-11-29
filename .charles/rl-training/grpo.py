"""
GRPO (Group Relative Policy Optimization) Trainer for Agent1.
Implements Eq 1 from the paper: clipped policy loss with group-relative advantages.

Uses TRL's GRPOTrainer as the foundation, with custom reward functions.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import torch
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
from datasets import Dataset
from loguru import logger


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.getenv(key)
    return int(val) if val else default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.getenv(key)
    return float(val) if val else default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default).lower())
    return val.lower() in ("true", "1", "yes")


@dataclass
class Agent1GRPOConfig(TRLGRPOConfig):
    """Extended GRPO config for Agent1 with paper-specific defaults."""
    
    # Load defaults from environment
    num_generations: int = field(default_factory=lambda: _get_env_int("NUM_GENERATIONS", 4))
    max_completion_length: int = field(default_factory=lambda: _get_env_int("MAX_COMPLETION_LENGTH", 2048))
    
    # Generation parameters - explicitly set to override model defaults
    temperature: float = 1.0  # Paper default, higher for diversity
    top_p: float = 0.99  # Near 1.0 for broad sampling
    top_k: int = None  # Disable top_k to use top_p
    
    # GRPO clipping
    epsilon: float = 0.2
    
    per_device_train_batch_size: int = field(default_factory=lambda: _get_env_int("PER_DEVICE_BATCH_SIZE", 2) * _get_env_int("NUM_GENERATIONS", 4))
    gradient_accumulation_steps: int = field(default_factory=lambda: _get_env_int("GRADIENT_ACCUMULATION_STEPS", 2))
    learning_rate: float = field(default_factory=lambda: _get_env_float("LEARNING_RATE", 1e-6))
    num_train_epochs: int = 1
    
    bf16: bool = True
    gradient_checkpointing: bool = field(default_factory=lambda: _get_env_bool("GRADIENT_CHECKPOINTING", True))
    
    logging_steps: int = field(default_factory=lambda: _get_env_int("LOGGING_STEPS", 1))
    save_steps: int = field(default_factory=lambda: _get_env_int("SAVE_STEPS", 5))


# Alias for backward compatibility
Agent0GRPOConfig = Agent1GRPOConfig


def create_grpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    reward_funcs: List[Callable],
    config: Optional[Agent1GRPOConfig] = None,
    eval_dataset: Optional[Dataset] = None,
) -> GRPOTrainer:
    """Create a GRPO trainer with Agent1-specific configuration."""
    if config is None:
        config = Agent1GRPOConfig(output_dir="./agent1_grpo_output")
    
    # Validate batch size vs num_generations
    if config.per_device_train_batch_size % config.num_generations != 0:
        old_batch = config.per_device_train_batch_size
        config.per_device_train_batch_size = config.num_generations
        logger.warning(
            f"⚠️ per_device_train_batch_size ({old_batch}) must be divisible by "
            f"num_generations ({config.num_generations}). Adjusted to {config.per_device_train_batch_size}."
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Override model's generation_config to use our settings
    if hasattr(model, 'generation_config'):
        model.generation_config.temperature = config.temperature
        model.generation_config.top_p = config.top_p
        if config.top_k is not None:
            model.generation_config.top_k = config.top_k
        model.generation_config.do_sample = True
        logger.info(f"🔧 Override model generation_config: temperature={config.temperature}, top_p={config.top_p}")
    
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
                f"grad_accum={config.gradient_accumulation_steps}, max_completion={config.max_completion_length}")
    logger.info(f"Generation: temperature={config.temperature}, top_p={config.top_p}")
    
    return trainer


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int = 4,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO (Eq 1)."""
    num_prompts = rewards.shape[0] // group_size
    rewards_grouped = rewards.view(num_prompts, group_size)
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True)
    advantages = (rewards_grouped - mean) / (std + 1e-8)
    return advantages.view(-1)


if __name__ == "__main__":
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
    advantages = compute_group_advantages(rewards, group_size=4)
    print(f"Rewards: {rewards}")
    print(f"Advantages: {advantages}")
    print("✅ GRPO module ready!")