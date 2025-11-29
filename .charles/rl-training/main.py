"""
Agent0 Co-Evolution Orchestrator
Runs the full co-evolutionary training loop.

Paper Reference: Algorithm 1

Usage:
    uv run main.py                          # Run full co-evolution
    uv run main.py --iterations 3           # Run 3 iterations
    uv run main.py --curriculum_only        # Train only curriculum agent
    uv run main.py --executor_only          # Train only executor agent
    uv run main.py --help
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import torch
from loguru import logger

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def run_curriculum_training(
    model_name: str,
    executor_model_path: Optional[str],
    output_dir: str,
    iteration: int,
    **kwargs
) -> str:
    """
    Run curriculum agent training (Algorithm 1, Lines 3-10).
    
    Returns:
        Path to trained curriculum model
    """
    from train_curriculum import train_curriculum_agent
    
    iter_output_dir = Path(output_dir) / f"curriculum_iter{iteration}"
    
    # Get wandb_project from kwargs or env
    wandb_project = kwargs.get("wandb_project") or os.getenv("WANDB_PROJECT", "rl-hackathon-agent1")
    run_prefix = os.getenv("WANDB_RUN_PREFIX", "agent1")
    
    logger.info(f"=" * 60)
    logger.info(f"CURRICULUM AGENT TRAINING - Iteration {iteration}")
    logger.info(f"=" * 60)
    
    train_curriculum_agent(
        model_name=model_name,
        executor_model_path=executor_model_path,
        output_dir=str(iter_output_dir),
        num_prompts=kwargs.get("num_prompts", 100),
        num_generations=kwargs.get("curriculum_k_rollouts", 4),
        max_steps=kwargs.get("curriculum_max_steps", 5),
        learning_rate=kwargs.get("learning_rate", 1e-6),
        per_device_batch_size=kwargs.get("curriculum_batch_size", 2),
        gradient_accumulation_steps=kwargs.get("curriculum_grad_accum", 4),
        use_lora=kwargs.get("use_lora", False),
        lora_r=kwargs.get("lora_r", 32),
        use_wandb=kwargs.get("use_wandb", True),
        wandb_project=wandb_project,
        wandb_run_name=f"{run_prefix}-curriculum-iter{iteration}",
        lambda_unc=kwargs.get("lambda_unc", 1.0),
        lambda_tool=kwargs.get("lambda_tool", 0.6),
        gamma_tool=kwargs.get("gamma_tool", 0.6),
        cap_tool=kwargs.get("cap_tool", 4),
        executor_k=kwargs.get("executor_k_samples", 4),
    )
    
    return str(iter_output_dir / "final_model")


def run_executor_training(
    model_name: str,
    curriculum_model_path: str,
    output_dir: str,
    iteration: int,
    **kwargs
) -> str:
    """
    Run executor agent training (Algorithm 1, Lines 11-24).
    
    Returns:
        Path to trained executor model
    """
    from train_executor import train_executor_agent
    
    iter_output_dir = Path(output_dir) / f"executor_iter{iteration}"
    
    # Get wandb_project from kwargs or env
    wandb_project = kwargs.get("wandb_project") or os.getenv("WANDB_PROJECT", "rl-hackathon-agent1")
    run_prefix = os.getenv("WANDB_RUN_PREFIX", "agent1")
    
    logger.info(f"=" * 60)
    logger.info(f"EXECUTOR AGENT TRAINING - Iteration {iteration}")
    logger.info(f"=" * 60)
    
    train_executor_agent(
        model_name=model_name,
        curriculum_model_path=curriculum_model_path,
        output_dir=str(iter_output_dir),
        num_tasks=kwargs.get("num_tasks", 200),
        k_samples=kwargs.get("executor_k_samples", 4),
        delta=kwargs.get("delta", 0.25),
        num_generations=kwargs.get("executor_k_rollouts", 4),
        max_steps=kwargs.get("executor_max_steps", 40),
        learning_rate=kwargs.get("learning_rate", 1e-6),
        per_device_batch_size=kwargs.get("executor_batch_size", 2),
        gradient_accumulation_steps=kwargs.get("executor_grad_accum", 4),
        use_lora=kwargs.get("use_lora", False),
        lora_r=kwargs.get("lora_r", 32),
        use_wandb=kwargs.get("use_wandb", True),
        wandb_project=wandb_project,
        wandb_run_name=f"{run_prefix}-executor-iter{iteration}",
        use_adpo_scaling=kwargs.get("use_adpo_scaling", False),
    )
    
    return str(iter_output_dir / "final_model")


def run_coevolution(
    model_name: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "./outputs",
    num_iterations: int = 1,
    curriculum_only: bool = False,
    executor_only: bool = False,
    **kwargs
):
    """
    Run the full Agent0 co-evolutionary loop.
    
    Model Weight Flow:
    ==================
    Iteration 1:
      - Curriculum: base_model → trained → curriculum_iter1/final_model
      - Executor:   base_model → trained → executor_iter1/final_model
    
    Iteration 2+:
      - Curriculum: curriculum_iter(t-1) → trained → curriculum_iter(t)
      - Executor:   executor_iter(t-1)   → trained → executor_iter(t)
    
    During training:
      - The agent being trained has TRAINABLE weights
      - The other agent is FROZEN (used for reward/task generation)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("AGENT1 CO-EVOLUTION FRAMEWORK")
    logger.info("=" * 70)
    logger.info(f"Base Model: {model_name}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Iterations: {num_iterations}")
    logger.info("=" * 70)
    
    # Track model paths across iterations
    # These start as base model and get updated after each phase
    curriculum_model_path = model_name  # Start from base
    executor_model_path = model_name    # Start from base
    
    # Training history
    history = {
        "model_name": model_name,
        "num_iterations": num_iterations,
        "iterations": [],
        "start_time": datetime.now().isoformat(),
    }
    
    for iteration in range(1, num_iterations + 1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# ITERATION {iteration}/{num_iterations}")
        logger.info(f"{'#' * 70}")
        logger.info(f"  Curriculum weights from: {curriculum_model_path}")
        logger.info(f"  Executor weights from:   {executor_model_path}")
        logger.info("")
        
        iter_info = {
            "iteration": iteration,
            "curriculum_model_input": curriculum_model_path,
            "executor_model_input": executor_model_path,
            "curriculum_model_output": None,
            "executor_model_output": None,
        }
        
        # ====================================================================
        # Phase 1: Curriculum Evolution
        # - Curriculum: TRAINABLE (starts from curriculum_model_path)
        # - Executor: FROZEN (used for computing p̂ rewards)
        # ====================================================================
        if not executor_only:
            logger.info("─" * 50)
            logger.info("PHASE 1: Curriculum Evolution")
            logger.info(f"  Training curriculum from: {curriculum_model_path}")
            logger.info(f"  Frozen executor from:     {executor_model_path}")
            logger.info("─" * 50)
            
            new_curriculum_path = run_curriculum_training(
                model_name=curriculum_model_path,  # Continue from previous curriculum
                executor_model_path=executor_model_path,  # Frozen executor for rewards
                output_dir=str(output_dir),
                iteration=iteration,
                **kwargs
            )
            curriculum_model_path = new_curriculum_path
            iter_info["curriculum_model_output"] = new_curriculum_path
            logger.success(f"✅ Curriculum model saved: {new_curriculum_path}")
        
        # ====================================================================
        # Phase 2: Executor Evolution
        # - Curriculum: FROZEN (used to generate tasks)
        # - Executor: TRAINABLE (starts from executor_model_path)
        # ====================================================================
        if not curriculum_only:
            logger.info("─" * 50)
            logger.info("PHASE 2: Executor Evolution")
            logger.info(f"  Frozen curriculum from: {curriculum_model_path}")
            logger.info(f"  Training executor from: {executor_model_path}")
            logger.info("─" * 50)
            
            new_executor_path = run_executor_training(
                model_name=executor_model_path,  # Continue from previous executor
                curriculum_model_path=curriculum_model_path,  # Frozen curriculum for task generation
                output_dir=str(output_dir),
                iteration=iteration,
                **kwargs
            )
            executor_model_path = new_executor_path
            iter_info["executor_model_output"] = new_executor_path
            logger.success(f"✅ Executor model saved: {new_executor_path}")
        
        history["iterations"].append(iter_info)
    
    # Save training history
    history["end_time"] = datetime.now().isoformat()
    history_path = output_dir / "coevolution_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("CO-EVOLUTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final Curriculum Model: {curriculum_model_path}")
    logger.info(f"Final Executor Model: {executor_model_path}")
    logger.info("=" * 70)
    
    return {
        "curriculum_model": curriculum_model_path,
        "executor_model": executor_model_path,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agent0 Co-Evolution Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model and output
    # NOTE: Use instruct model (Qwen/Qwen3-0.6B), NOT base model (Qwen/Qwen3-0.6B-Base)
    parser.add_argument("--model_name", type=str, 
                        default=os.getenv("MODEL_ID", "Qwen/Qwen3-0.6B"),
                        help="Base model name (use instruct models, not *-Base)")
    parser.add_argument("--output_dir", type=str,
                        default=os.getenv("OUTPUT_DIR", "./outputs"),
                        help="Output directory")
    
    # Iterations
    parser.add_argument("--iterations", type=int,
                        default=int(os.getenv("NUM_ITERATIONS", "1")),
                        help="Number of co-evolution iterations")
    
    # Mode selection
    parser.add_argument("--curriculum_only", action="store_true",
                        help="Train only curriculum agent")
    parser.add_argument("--executor_only", action="store_true",
                        help="Train only executor agent")
    
    # Curriculum agent settings
    parser.add_argument("--curriculum_max_steps", type=int,
                        default=int(os.getenv("CURRICULUM_MAX_STEPS", "5")),
                        help="Max steps for curriculum training")
    parser.add_argument("--curriculum_batch_size", type=int,
                        default=int(os.getenv("CURRICULUM_BATCH_SIZE", "2")),
                        help="Batch size for curriculum training")
    parser.add_argument("--curriculum_grad_accum", type=int,
                        default=int(os.getenv("CURRICULUM_GRAD_ACCUM", "4")),
                        help="Gradient accumulation for curriculum")
    parser.add_argument("--curriculum_k_rollouts", type=int,
                        default=int(os.getenv("CURRICULUM_K_ROLLOUTS", "4")),
                        help="GRPO rollouts for curriculum")
    parser.add_argument("--num_prompts", type=int,
                        default=int(os.getenv("NUM_CURRICULUM_PROMPTS", "100")),
                        help="Number of curriculum prompts")
    
    # Executor agent settings
    parser.add_argument("--executor_max_steps", type=int,
                        default=int(os.getenv("EXECUTOR_MAX_STEPS", "40")),
                        help="Max steps for executor training")
    parser.add_argument("--executor_batch_size", type=int,
                        default=int(os.getenv("EXECUTOR_BATCH_SIZE", "2")),
                        help="Batch size for executor training")
    parser.add_argument("--executor_grad_accum", type=int,
                        default=int(os.getenv("EXECUTOR_GRAD_ACCUM", "4")),
                        help="Gradient accumulation for executor")
    parser.add_argument("--executor_k_rollouts", type=int,
                        default=int(os.getenv("EXECUTOR_K_ROLLOUTS", "4")),
                        help="GRPO rollouts for executor")
    parser.add_argument("--executor_k_samples", type=int,
                        default=int(os.getenv("EXECUTOR_K_SAMPLES", "4")),
                        help="Samples for p̂ computation")
    parser.add_argument("--num_tasks", type=int,
                        default=int(os.getenv("NUM_TASKS_TO_GENERATE", "200")),
                        help="Number of tasks to generate")
    
    # Shared settings
    parser.add_argument("--learning_rate", type=float,
                        default=float(os.getenv("LEARNING_RATE", "1e-6")),
                        help="Learning rate")
    parser.add_argument("--delta", type=float,
                        default=float(os.getenv("DELTA", "0.25")),
                        help="Frontier filtering threshold")
    
    # Reward parameters
    parser.add_argument("--lambda_unc", type=float,
                        default=float(os.getenv("LAMBDA_UNC", "1.0")),
                        help="R_unc weight")
    parser.add_argument("--lambda_tool", type=float,
                        default=float(os.getenv("LAMBDA_TOOL", "0.6")),
                        help="R_tool weight")
    parser.add_argument("--gamma_tool", type=float,
                        default=float(os.getenv("GAMMA_TOOL", "0.6")),
                        help="Tool reward scale")
    parser.add_argument("--cap_tool", type=int,
                        default=int(os.getenv("CAP_TOOL", "4")),
                        help="Tool reward cap")
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true",
                        default=os.getenv("USE_LORA", "false").lower() == "true",
                        help="Use LoRA adapters")
    parser.add_argument("--lora_r", type=int,
                        default=int(os.getenv("LORA_R", "32")),
                        help="LoRA rank")
    
    # W&B settings
    parser.add_argument("--use_wandb", action="store_true",
                        default=os.getenv("USE_WANDB", "true").lower() == "true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str,
                        default=os.getenv("WANDB_PROJECT", "rl-hackathon-agent1"),
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Post-process args
    # - Convert boolean flags from strings
    # - Expand user home directory in paths
    for key, value in vars(args).items():
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            setattr(args, key, value.lower() == "true")
        elif isinstance(value, str) and value.startswith("~/"):
            expanded_path = str(Path.home() / value[2:])
            setattr(args, key, expanded_path)
    
    return args


def main():
    args = parse_args()
    
    # Note: Environment variables are already handled in parse_args() via os.getenv() defaults
    # No need for additional override logic here
    
    # Convert to dict for easier access
    train_settings = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "num_iterations": args.iterations,
        "curriculum_only": args.curriculum_only,
        "executor_only": args.executor_only,
        "num_prompts": args.num_prompts,
        "curriculum_k_rollouts": args.curriculum_k_rollouts,
        "curriculum_max_steps": args.curriculum_max_steps,
        "learning_rate": args.learning_rate,
        "curriculum_batch_size": args.curriculum_batch_size,
        "curriculum_grad_accum": args.curriculum_grad_accum,
        "executor_batch_size": args.executor_batch_size,
        "executor_grad_accum": args.executor_grad_accum,
        "executor_k_rollouts": args.executor_k_rollouts,
        "executor_k_samples": args.executor_k_samples,
        "executor_max_steps": args.executor_max_steps,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "num_tasks": args.num_tasks,
        "delta": args.delta,
        "lambda_unc": args.lambda_unc,
        "lambda_tool": args.lambda_tool,
        "gamma_tool": args.gamma_tool,
        "cap_tool": args.cap_tool,
    }
    
    # Log final settings
    logger.info("=" * 70)
    logger.info("FINAL TRAINING SETTINGS")
    logger.info("=" * 70)
    for key, value in train_settings.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)
    
    # Run the co-evolutionary training loop
    # Pass all settings via **train_settings (no duplicate positional args)
    run_coevolution(**train_settings)


if __name__ == "__main__":
    main()
