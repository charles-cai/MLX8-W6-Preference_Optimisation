"""
Curriculum Agent Training for Agent0.

Trains the curriculum agent to generate challenging tasks that maximize
the executor agent's uncertainty (self-consistency ≈ 0.5).

Paper Reference:
    - Algorithm 1, Lines 3-10: Curriculum Evolution
    - Section 3.2: Curriculum Agent Training
    - Table 7: Prompt templates
    - Table 8: Hyperparameters
    
Key Equations:
    - Eq 2: R_unc(x; π_φ) = 1 - 2|p̂(x; π_φ) - 0.5|  (Uncertainty reward)
    - Eq 3: R_tool(x; π_φ) = γ · min(N_tool(y), C)  (Tool use reward)
    - Eq 5: R_C = R_format · max(0, λ_unc·R_unc + λ_tool·R_tool - R_rep)

Usage:
    uv run train_curriculum.py --model_name Qwen/Qwen3-0.6B --output_dir ./outputs/curriculum
    uv run train_curriculum.py --prompt_preset data_scientist
    uv run train_curriculum.py --help
"""

import os
import re
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from tqdm import tqdm

# Configure loguru for colorful output
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed, W&B logging disabled")

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft not installed, LoRA disabled")

# Local imports
from grpo import Agent0GRPOConfig, create_grpo_trainer
from rewards import compute_r_unc, compute_r_tool, check_format
from self_consistency import (
    compute_p_hat,
    extract_boxed_answer,
    save_generations_to_jsonl,
    save_generations_to_csv,
)
from prompts import load_prompts, get_available_presets


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(
    model_name: str,
    use_lora: bool = False,
    lora_r: int = 32,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
) -> tuple:
    """
    Load model and tokenizer with optional LoRA adapters.
    
    For Agent0, both curriculum and executor agents are initialized from
    the same base model (π_base). This function handles loading with
    optional parameter-efficient fine-tuning via LoRA.
    
    Paper Reference:
        Section 3.1: "two functionally distinct agents initialized from 
        the same base LLM, π_base"
    
    Args:
        model_name: HuggingFace model name or local path
        use_lora: Whether to apply LoRA adapters for memory efficiency
        lora_r: LoRA rank (higher = more capacity, more memory)
        device_map: Device placement strategy ("auto" for multi-GPU)
        torch_dtype: Model precision (None = auto-detect bf16/fp16)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"📦 Loading model: {model_name}")
    
    # Auto-detect best dtype for hardware
    if torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("Using bfloat16 precision")
        else:
            torch_dtype = torch.float16
            logger.info("Using float16 precision")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    # Ensure pad token exists for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token to eos_token")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Apply LoRA for memory-efficient fine-tuning
    if use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft not installed. Run: pip install peft")
        
        logger.info(f"🔧 Applying LoRA with r={lora_r}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_r * 2,  # Common heuristic: alpha = 2*r
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.success(f"✅ Model loaded: {num_params:,} total params, {trainable_params:,} trainable")
    
    return model, tokenizer


# ============================================================================
# Dataset Creation
# ============================================================================

def create_curriculum_prompts_dataset(
    num_prompts: int = 100,
    system_prompt: str = "",
    user_prompt: str = "",
    tokenizer: Optional[AutoTokenizer] = None,
) -> Dataset:
    """
    Create dataset of curriculum generation prompts.
    
    Each prompt instructs the curriculum agent to generate a new task.
    The prompts are identical (same instruction repeated) because the
    model learns to generate diverse tasks through RL optimization.
    
    Paper Reference:
        Table 7: Curriculum Agent prompt template
        Section 3.2: "For each task x_i generated by π_θ"
    
    Args:
        num_prompts: Number of generation prompts (batch size for curriculum)
        system_prompt: System instruction for task generation
        user_prompt: User request for task generation
        tokenizer: Tokenizer for applying chat template
    
    Returns:
        HuggingFace Dataset with 'prompt' and 'id' columns
    """
    prompts = []
    
    logger.info(f"📝 Creating {num_prompts} curriculum prompts...")
    
    for i in tqdm(range(num_prompts), desc="Creating prompts", unit="prompt"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        if tokenizer is not None:
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_str = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        prompts.append({
            "prompt": prompt_str,
            "id": f"curriculum_{i}",
        })
    
    logger.success(f"✅ Created dataset with {len(prompts)} prompts")
    return Dataset.from_list(prompts)


# ============================================================================
# Executor Sampling (for computing p̂)
# ============================================================================

@torch.no_grad()
def sample_executor_responses(
    executor_model,
    executor_tokenizer,
    task: str,
    k: int = 4,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    executor_system_prompt: str = "",
) -> List[str]:
    """
    Sample k responses from the executor agent for self-consistency computation.
    
    This implements the sampling step needed for computing p̂ (Eq 6).
    The executor attempts to solve the task k times, and we use majority
    voting to determine uncertainty.
    
    Paper Reference:
        Section 3.2: "we compute its reward by sampling k responses 
        {y_j}_{j=1}^k from the current Executor π_φ"
        Eq 6: p̂(x) = (1/k) * Σ I(σ_i = ỹ)
    
    Args:
        executor_model: Frozen executor model for response sampling
        executor_tokenizer: Tokenizer for the executor
        task: The problem/task to solve
        k: Number of responses to sample (paper default: 10)
        max_new_tokens: Maximum generation length
        temperature: Sampling temperature (paper: 1.0)
        executor_system_prompt: System prompt for executor
    
    Returns:
        List of k response strings
    """
    messages = [
        {"role": "system", "content": executor_system_prompt or "Solve the following problem step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": task},
    ]
    
    prompt_text = executor_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = executor_tokenizer(prompt_text, return_tensors="pt").to(executor_model.device)
    
    responses = []
    
    for _ in range(k):
        outputs = executor_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.99,  # Paper: top_p = 0.99
            pad_token_id=executor_tokenizer.pad_token_id,
        )
        
        response = executor_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        responses.append(response)
    
    return responses


# ============================================================================
# Custom Reward Function for GRPO
# ============================================================================

class CurriculumRewardComputer:
    """
    Computes curriculum reward R_C using frozen executor.
    
    The curriculum agent is rewarded for generating tasks that:
    1. Maximize executor uncertainty (R_unc, Eq 2)
    2. Encourage tool use (R_tool, Eq 3)
    3. Maintain valid format (R_format gate)
    
    Paper Reference:
        Section 3.2: Curriculum Agent Training
        Eq 5: R_C(x_i) = R_format(x_i) · max(0, (λ_unc·R_unc + λ_tool·R_tool) - R_rep(x_i))
        
        Table 8 Hyperparameters:
        - λ_unc = 1.0
        - λ_tool = 0.6
        - γ (tool scale) = 0.6
        - C (tool cap) = 4
    
    Attributes:
        executor_model: Frozen executor for computing uncertainty
        executor_tokenizer: Tokenizer for executor
        k: Number of samples for self-consistency
        lambda_unc: Weight for uncertainty reward
        lambda_tool: Weight for tool use reward
        gamma_tool: Scale factor for tool reward
        cap_tool: Maximum rewarded tool calls
    """
    
    def __init__(
        self,
        executor_model,
        executor_tokenizer,
        k: int = 4,
        lambda_unc: float = 1.0,
        lambda_tool: float = 0.6,
        gamma_tool: float = 0.6,
        cap_tool: int = 4,
        output_dir: str = "./outputs",
        executor_system_prompt: str = "",
    ):
        """
        Initialize the curriculum reward computer.
        
        Args:
            executor_model: Frozen executor model
            executor_tokenizer: Tokenizer for executor
            k: Number of responses for self-consistency (paper: k=10)
            lambda_unc: R_unc weight (paper: 1.0)
            lambda_tool: R_tool weight (paper: 0.6)
            gamma_tool: Tool reward scale (paper: 0.6)
            cap_tool: Max rewarded tool calls (paper: 4)
            output_dir: Directory for saving generation logs
            executor_system_prompt: System prompt for executor sampling
        """
        self.executor_model = executor_model
        self.executor_tokenizer = executor_tokenizer
        self.k = k
        self.lambda_unc = lambda_unc
        self.lambda_tool = lambda_tool
        self.gamma_tool = gamma_tool
        self.cap_tool = cap_tool
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor_system_prompt = executor_system_prompt
        self.generation_log = []
        self.call_count = 0
        self.batch_rewards = []
        self.batch_p_hats = []
    
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for generated curriculum tasks.
        
        This is called by GRPOTrainer during training. For each generated
        task, we:
        1. Check format validity
        2. Sample k executor responses
        3. Compute self-consistency p̂
        4. Calculate composite reward R_C
        
        Args:
            prompts: List of curriculum prompts
            completions: List of generated tasks
            **kwargs: Additional arguments (unused)
        
        Returns:
            List of reward scores
        """
        rewards = []
        self.call_count += 1
        
        pbar = tqdm(
            zip(prompts, completions),
            total=len(prompts),
            desc=f"Computing rewards (batch {self.call_count})",
            unit="task",
            leave=False,
        )
        
        for idx, (prompt, task) in enumerate(pbar):
            reward, log_entry = self._compute_single_reward(prompt, task, idx)
            rewards.append(reward)
            self.generation_log.append(log_entry)
            pbar.set_postfix({"reward": f"{reward:.3f}"})
        
        self.batch_rewards.extend(rewards)
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        logger.info(f"📊 Batch {self.call_count}: avg_reward={avg_reward:.3f}, samples={len(rewards)}")
        
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "curriculum/batch_reward": avg_reward,
                "curriculum/batch_size": len(rewards),
                "curriculum/total_samples": len(self.batch_rewards),
            })
        
        return rewards
    
    def _compute_single_reward(
        self,
        prompt: str,
        task: str,
        idx: int,
    ) -> tuple:
        """
        Compute reward for a single generated task.
        
        Implements Eq 5 from the paper (simplified without R_rep).
        """
        log_entry = {
            "batch": self.call_count,
            "idx": idx,
            "prompt": str(prompt)[:200],
            "task": task,
            "reward": 0.0,
            "p_hat": None,
            "majority_answer": None,
            "r_unc": None,
            "r_tool": None,
            "status": "unknown",
            "executor_responses": [],
        }
        
        # R_format gate: Check valid output format
        if not check_format(task):
            log_entry["status"] = "format_fail"
            return 0.0, log_entry
        
        # Extract question for executor
        question_match = re.search(r'<question>(.*?)</question>', task, re.DOTALL)
        if not question_match:
            log_entry["status"] = "no_question"
            return 0.0, log_entry
        
        question = question_match.group(1).strip()
        
        # Sample k executor responses
        try:
            executor_responses = sample_executor_responses(
                self.executor_model,
                self.executor_tokenizer,
                question,
                k=self.k,
                executor_system_prompt=self.executor_system_prompt,
            )
        except Exception as e:
            logger.error(f"Executor sampling failed: {e}")
            log_entry["status"] = "executor_error"
            return 0.0, log_entry
        
        log_entry["executor_responses"] = [r[:500] for r in executor_responses]
        
        # Compute p̂ (self-consistency, Eq 6)
        p_hat, majority_answer, votes = compute_p_hat(executor_responses)
        log_entry["p_hat"] = p_hat
        log_entry["majority_answer"] = majority_answer
        log_entry["vote_distribution"] = votes
        self.batch_p_hats.append(p_hat)
        
        # R_unc: Uncertainty reward (Eq 2)
        r_unc = compute_r_unc(p_hat)
        log_entry["r_unc"] = r_unc
        
        # R_tool: Tool use reward (Eq 3)
        r_tool = compute_r_tool(
            executor_responses[0] if executor_responses else "",
            gamma=self.gamma_tool,
            cap=self.cap_tool,
        )
        log_entry["r_tool"] = r_tool
        
        # Composite reward R_C (Eq 5, without R_rep for MVP)
        reward = self.lambda_unc * r_unc + self.lambda_tool * r_tool
        reward = max(0.0, reward)
        
        log_entry["reward"] = reward
        log_entry["status"] = "success"
        
        return reward, log_entry
    
    def save_logs(self, suffix: str = ""):
        """Save generation logs to JSONL and CSV files."""
        if not self.generation_log:
            logger.warning("No logs to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        jsonl_path = self.output_dir / f"curriculum_generations_{timestamp}{suffix}.jsonl"
        save_generations_to_jsonl(self.generation_log, str(jsonl_path), mode='w')
        
        csv_data = [
            {
                "batch": log.get("batch"),
                "idx": log.get("idx"),
                "status": log.get("status"),
                "reward": log.get("reward"),
                "p_hat": log.get("p_hat"),
                "r_unc": log.get("r_unc"),
                "r_tool": log.get("r_tool"),
                "majority_answer": log.get("majority_answer"),
                "task_preview": log.get("task", "")[:200],
            }
            for log in self.generation_log
        ]
        
        csv_path = self.output_dir / f"curriculum_summary_{timestamp}{suffix}.csv"
        save_generations_to_csv(csv_data, str(csv_path), mode='w')
        
        logger.success(f"💾 Saved {len(self.generation_log)} logs to {self.output_dir}")
        
        saved_count = len(self.generation_log)
        self.generation_log = []
        return saved_count


# ============================================================================
# Main Training Function
# ============================================================================

def train_curriculum_agent(
    model_name: str = "Qwen/Qwen3-0.6B",
    executor_model_path: Optional[str] = None,
    output_dir: str = "./outputs/curriculum",
    prompt_preset: str = "data_scientist",
    num_prompts: int = 100,
    num_generations: int = 4,
    max_steps: int = 10,
    learning_rate: float = 1e-6,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    use_lora: bool = False,
    lora_r: int = 32,
    use_wandb: bool = True,
    wandb_project: str = "agent0-curriculum",
    wandb_run_name: Optional[str] = None,
    save_steps: int = 5,
    logging_steps: int = 1,
    lambda_unc: float = 1.0,
    lambda_tool: float = 0.6,
    gamma_tool: float = 0.6,
    cap_tool: int = 4,
    executor_k: int = 4,
):
    """
    Train the Curriculum Agent using GRPO.
    
    Implements Algorithm 1, Lines 3-10 (Curriculum Evolution) from the paper.
    The curriculum agent learns to generate tasks that maximize the executor's
    uncertainty while encouraging tool use.
    
    Paper Reference:
        Algorithm 1: Self-Evolutionary Framework Agent0
        Section 3.2: Curriculum Agent Training
        Table 8: Training hyperparameters
    
    Args:
        model_name: Base model to train
        executor_model_path: Path to frozen executor (None = use base model)
        output_dir: Directory for outputs and checkpoints
        prompt_preset: Prompt configuration from prompts.toml
        num_prompts: Number of curriculum prompts per training batch
        num_generations: Number of GRPO rollouts per prompt
        max_steps: Maximum training steps (paper: 5)
        learning_rate: Learning rate (paper: 1e-6)
        per_device_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        use_wandb: Whether to log to W&B
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        save_steps: Steps between checkpoints
        logging_steps: Steps between logs
        lambda_unc: R_unc weight (paper: 1.0)
        lambda_tool: R_tool weight (paper: 0.6)
        gamma_tool: Tool reward scale (paper: 0.6)
        cap_tool: Max rewarded tool calls (paper: 4)
        executor_k: Number of executor samples for p̂ (paper: 10)
    
    Returns:
        Trained GRPOTrainer instance
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts from TOML
    prompt_config = load_prompts(prompt_preset)
    
    logger.info("=" * 60)
    logger.info("🚀 CURRICULUM AGENT TRAINING")
    logger.info(f"   Prompt Preset: {prompt_config.name}")
    logger.info("=" * 60)
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        run_name = wandb_run_name or f"curriculum_{prompt_preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model_name": model_name,
                "prompt_preset": prompt_preset,
                "num_prompts": num_prompts,
                "num_generations": num_generations,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "lambda_unc": lambda_unc,
                "lambda_tool": lambda_tool,
                "executor_k": executor_k,
            }
        )
        logger.success(f"📊 W&B initialized: {wandb_project}/{run_name}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️ wandb requested but not available")
        use_wandb = False
    
    # Load curriculum model (trainable)
    logger.info("=" * 60)
    logger.info("📦 Loading curriculum model (trainable)...")
    curriculum_model, tokenizer = load_model_and_tokenizer(
        model_name,
        use_lora=use_lora,
        lora_r=lora_r,
    )
    
    # Load executor model (frozen)
    logger.info("=" * 60)
    logger.info("📦 Loading executor model (frozen)...")
    executor_path = executor_model_path or model_name
    executor_model, executor_tokenizer = load_model_and_tokenizer(
        executor_path,
        use_lora=False,
    )
    executor_model.eval()
    for param in executor_model.parameters():
        param.requires_grad = False
    logger.info("🔒 Executor model frozen")
    
    # Create dataset with prompts from TOML
    logger.info("=" * 60)
    train_dataset = create_curriculum_prompts_dataset(
        num_prompts=num_prompts,
        system_prompt=prompt_config.curriculum_system,
        user_prompt=prompt_config.curriculum_user,
        tokenizer=tokenizer,
    )
    
    # Create reward computer
    logger.info("=" * 60)
    logger.info("⚙️ Creating reward computer...")
    reward_computer = CurriculumRewardComputer(
        executor_model=executor_model,
        executor_tokenizer=executor_tokenizer,
        k=executor_k,
        lambda_unc=lambda_unc,
        lambda_tool=lambda_tool,
        gamma_tool=gamma_tool,
        cap_tool=cap_tool,
        output_dir=str(output_dir),
        executor_system_prompt=prompt_config.executor_system,
    )
    
    # Create GRPO config
    config = Agent0GRPOConfig(
        output_dir=str(output_dir),
        num_generations=num_generations,
        max_new_tokens=2048,
        temperature=1.0,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if use_wandb else "none",
        weight_decay=0.01,
    )
    
    trainer = create_grpo_trainer(
        model=curriculum_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=[reward_computer],
        config=config,
    )
    
    # Train
    logger.info("=" * 60)
    logger.info("🏋️ Starting training...")
    logger.info(f"  📦 Model: {model_name}")
    logger.info(f"  📝 Preset: {prompt_config.name}")
    logger.info(f"  🔢 Steps: {max_steps}")
    logger.info(f"  🎲 Rollouts: {num_generations}")
    logger.info("=" * 60)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        reward_computer.save_logs("_final")
        
        final_path = output_dir / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.success(f"💾 Saved final model to {final_path}")
        
        training_info = {
            "model_name": model_name,
            "prompt_preset": prompt_preset,
            "prompt_config_name": prompt_config.name,
            "executor_model_path": executor_path,
            "output_dir": str(output_dir),
            "num_prompts": num_prompts,
            "max_steps": max_steps,
            "lambda_unc": lambda_unc,
            "lambda_tool": lambda_tool,
            "executor_k": executor_k,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    logger.info("=" * 60)
    logger.success("✅ Curriculum agent training complete!")
    logger.info("=" * 60)
    
    return trainer


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Agent0 Curriculum Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--executor_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/curriculum")
    
    # Prompt configuration
    available_presets = get_available_presets() if PROMPTS_FILE.exists() else ["math", "data_scientist"]
    parser.add_argument("--prompt_preset", type=str, 
                        default=os.getenv("PROMPT_PRESET", "data_scientist"),
                        choices=available_presets,
                        help=f"Prompt preset from prompts.toml")
    
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="agent0-curriculum")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--lambda_unc", type=float, default=1.0)
    parser.add_argument("--lambda_tool", type=float, default=0.6)
    parser.add_argument("--gamma_tool", type=float, default=0.6)
    parser.add_argument("--cap_tool", type=int, default=4)
    parser.add_argument("--executor_k", type=int, default=4)
    
    return parser.parse_args()


# Path to prompts file for CLI validation
PROMPTS_FILE = Path(__file__).parent / "prompts.toml"


def main():
    """Main entry point."""
    args = parse_args()
    
    train_curriculum_agent(
        model_name=args.model_name,
        executor_model_path=args.executor_model_path,
        output_dir=args.output_dir,
        prompt_preset=args.prompt_preset,
        num_prompts=args.num_prompts,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        lambda_unc=args.lambda_unc,
        lambda_tool=args.lambda_tool,
        gamma_tool=args.gamma_tool,
        cap_tool=args.cap_tool,
        executor_k=args.executor_k,
    )


if __name__ == "__main__":
    main()
