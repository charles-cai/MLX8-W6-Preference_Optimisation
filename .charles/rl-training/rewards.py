"""
Reward functions for Agent0 Curriculum Agent training.
Implements R_unc (Eq 2), R_tool (Eq 3), and R_C (Eq 5) from the paper.
"""

import re
from typing import List, Optional
from loguru import logger


def compute_r_unc(p_hat: float) -> float:
    """
    Compute uncertainty reward R_unc (Eq 2).
    
    Maximized when p̂ = 0.5 (highest executor uncertainty).
    Penalizes tasks that are too easy (p̂ → 1) or too hard (p̂ → 0).
    
    R_unc(x; π_φ) = 1 - 2|p̂(x; π_φ) - 0.5|
    
    Args:
        p_hat: Self-consistency score (proportion voting for majority answer)
               Range: [0, 1], where 0.5 = maximum uncertainty
    
    Returns:
        Uncertainty reward in range [0, 1]
    """
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def compute_r_tool(
    completion: str,
    tool_marker: str = "```python",
    output_marker: str = "```output",
    gamma: float = 0.6,
    cap: int = 4,
) -> float:
    """
    Compute tool use reward R_tool (Eq 3).
    
    Rewards tasks that prompt the executor to use code interpreter.
    Capped to prevent rewarding excessive/spurious tool use.
    
    R_tool(x; π_φ) = γ · min(N_tool(y), C)
    
    Args:
        completion: The executor's response text
        tool_marker: Marker indicating code block start (e.g., ```python)
        output_marker: Marker indicating tool execution output
        gamma: Scaling hyperparameter (default 0.6 from paper)
        cap: Maximum number of rewarded tool calls (default 4 from paper)
    
    Returns:
        Tool use reward (γ * min(count, cap))
    """
    # Count tool invocations by looking for output markers
    # The paper uses 'output' markers within complete predictions
    n_tool = len(re.findall(re.escape(output_marker), completion, re.IGNORECASE))
    
    # Also count python code blocks as potential tool uses
    if n_tool == 0:
        n_tool = len(re.findall(re.escape(tool_marker), completion, re.IGNORECASE))
    
    return gamma * min(n_tool, cap)


def check_format(task: str) -> bool:
    """
    Check if generated task has valid format.
    
    Valid format requires:
    - <question> tags with content
    - \\boxed{} with an answer
    
    Args:
        task: Generated task string from curriculum agent
    
    Returns:
        True if format is valid, False otherwise
    """
    # Check for question tags
    has_question = bool(re.search(r'<question>.*?</question>', task, re.DOTALL))
    
    # Check for boxed answer
    has_boxed = bool(re.search(r'\\boxed\{[^}]+\}', task))
    
    return has_question and has_boxed


def compute_r_curriculum(
    task: str,
    p_hat: float,
    executor_completion: str,
    lambda_unc: float = 1.0,
    lambda_tool: float = 0.6,
    gamma_tool: float = 0.6,
    cap_tool: int = 4,
) -> float:
    """
    Compute composite curriculum reward R_C (Eq 5).
    
    R_C(x_i) = R_format(x_i) · max(0, (λ_unc·R_unc + λ_tool·R_tool) - R_rep(x_i))
    
    Note: R_rep (repetition penalty) is skipped for MVP - requires batch-level
    BLEU computation which adds complexity. Can be added in Phase 2.
    
    Args:
        task: Generated task from curriculum agent
        p_hat: Self-consistency score from executor responses
        executor_completion: Sample completion from executor for tool counting
        lambda_unc: Weight for uncertainty reward (default 1.0)
        lambda_tool: Weight for tool reward (default 0.6)
        gamma_tool: Scaling for tool reward (default 0.6)
        cap_tool: Cap on tool calls (default 4)
    
    Returns:
        Composite curriculum reward
    """
    # Format gate - zero reward if format is invalid
    if not check_format(task):
        logger.debug(f"Task failed format check, returning 0 reward")
        return 0.0
    
    # Compute component rewards
    r_unc = compute_r_unc(p_hat)
    r_tool = compute_r_tool(
        executor_completion, 
        gamma=gamma_tool, 
        cap=cap_tool
    )
    
    # Composite reward (without repetition penalty for MVP)
    composite = lambda_unc * r_unc + lambda_tool * r_tool
    
    # Ensure non-negative
    reward = max(0.0, composite)
    
    logger.debug(f"R_C components: r_unc={r_unc:.3f}, r_tool={r_tool:.3f}, total={reward:.3f}")
    
    return reward


def reward_fn_for_grpo(
    prompts: List[str],
    completions: List[str],
    p_hats: Optional[List[float]] = None,
    **kwargs
) -> List[float]:
    """
    Reward function compatible with TRL's GRPOTrainer interface.
    
    For curriculum agent training, this wraps compute_r_curriculum.
    
    Args:
        prompts: List of prompts (curriculum generation prompts)
        completions: List of generated tasks from curriculum agent
        p_hats: List of self-consistency scores (must be provided externally)
        **kwargs: Additional arguments (e.g., executor_completions)
    
    Returns:
        List of reward scores
    """
    if p_hats is None:
        logger.warning("p_hats not provided, using default 0.5 (neutral uncertainty)")
        p_hats = [0.5] * len(completions)
    
    executor_completions = kwargs.get('executor_completions', [''] * len(completions))
    
    rewards = []
    for task, p_hat, exec_comp in zip(completions, p_hats, executor_completions):
        r = compute_r_curriculum(task, p_hat, exec_comp)
        rewards.append(r)
    
    return rewards


if __name__ == "__main__":
    # Quick sanity checks
    print("Testing reward functions...")
    
    # Test R_unc
    print("\n--- R_unc (Uncertainty Reward) ---")
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r = compute_r_unc(p)
        print(f"  p̂={p:.2f} → R_unc={r:.3f}")
    
    # Test R_tool
    print("\n--- R_tool (Tool Use Reward) ---")
    test_completions = [
        "No code here",
        "```python\nprint('hello')\n```\n```output\nhello\n```",
        "```python\nx=1\n```\n```output\n1\n```\n```python\ny=2\n```\n```output\n2\n```",
    ]
    for comp in test_completions:
        r = compute_r_tool(comp)
        print(f"  '{comp[:30]}...' → R_tool={r:.3f}")
    
    # Test format check
    print("\n--- Format Check ---")
    valid_task = "<question>What is 2+2?</question>\n\\boxed{4}"
    invalid_task = "What is 2+2? Answer: 4"
    print(f"  Valid format: {check_format(valid_task)}")
    print(f"  Invalid format: {check_format(invalid_task)}")
    
    # Test composite reward
    print("\n--- R_C (Composite Curriculum Reward) ---")
    r = compute_r_curriculum(
        task=valid_task,
        p_hat=0.5,
        executor_completion="```python\nprint(2+2)\n```\n```output\n4\n```"
    )
    print(f"  Composite reward: {r:.3f}")
    
    print("\n✅ Rewards module ready!")
