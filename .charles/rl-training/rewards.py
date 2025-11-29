"""
Reward functions for Agent0 curriculum and executor training.

Paper Reference:
    - Eq 2: R_unc = 1 - 2|p̂ - 0.5| (Uncertainty reward)
    - Eq 3: R_tool = γ · min(N_tool, C) (Tool use reward)
    - Eq 5: R_C = R_format · max(0, λ_unc·R_unc + λ_tool·R_tool - R_rep)
"""

import re
from typing import Optional
from loguru import logger


def compute_r_unc(p_hat: float) -> float:
    """
    Compute uncertainty reward R_unc (Eq 2).
    
    R_unc(x; π_φ) = 1 - 2|p̂(x; π_φ) - 0.5|
    
    Maximized when p̂ = 0.5 (highest uncertainty).
    Returns 0 when p̂ = 0 or p̂ = 1 (no uncertainty).
    """
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def compute_r_tool(
    response: str,
    gamma: float = 0.6,
    cap: int = 4,
) -> float:
    """
    Compute tool use reward R_tool (Eq 3).
    
    R_tool(x; π_φ) = γ · min(N_tool(y), C)
    
    Per paper: "identified by the tool response marker, i.e., 'output'"
    N_tool counts the number of tool execution outputs, not code blocks.
    """
    # Count tool invocations by looking for output markers (tool responses)
    output_markers = len(re.findall(r"'''output|```output", response, re.IGNORECASE))
    n_tool = output_markers
    return gamma * min(n_tool, cap)


def check_format(task: str) -> bool:
    """
    Check if generated task has valid format (R_format gate).
    
    Valid format requires <question>...</question> tags.
    """
    if task is None or not task.strip():
        return False
    
    has_question = bool(re.search(r'<question>.*?</question>', task, re.DOTALL))
    return has_question


def compute_r_c(
    p_hat: float,
    response: str,
    task: str,
    lambda_unc: float = 1.0,
    lambda_tool: float = 0.6,
    gamma_tool: float = 0.6,
    cap_tool: int = 4,
    r_rep: float = 0.0,
) -> float:
    """
    Compute composite curriculum reward R_C (Eq 5).
    
    R_C(x_i) = R_format(x_i) · max(0, (λ_unc·R_unc + λ_tool·R_tool) - R_rep(x_i))
    """
    if not check_format(task):
        logger.debug("Task failed format check, returning 0 reward")
        return 0.0
    
    r_unc = compute_r_unc(p_hat)
    r_tool = compute_r_tool(response, gamma=gamma_tool, cap=cap_tool)
    reward = lambda_unc * r_unc + lambda_tool * r_tool - r_rep
    
    logger.debug(f"R_C: r_unc={r_unc:.3f}, r_tool={r_tool:.3f}, r_rep={r_rep:.3f} → {max(0.0, reward):.3f}")
    return max(0.0, reward)


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    logger.info("Testing reward functions...")
    
    # Test R_unc
    logger.info("\n📊 R_unc (Uncertainty Reward):")
    for p_hat in [0.0, 0.25, 0.5, 0.75, 1.0]:
        r_unc = compute_r_unc(p_hat)
        logger.info(f"  p̂={p_hat:.2f} → R_unc={r_unc:.3f}")
    
    # Test R_tool
    logger.info("\n🔧 R_tool (Tool Use Reward):")
    responses = [
        "Just text, no code",
        "Here's code:\n```python\nprint('hello')\n```",
        "With output:\n'''python\nprint(1)\n'''\n'''output\n1\n'''",
    ]
    for resp in responses:
        r_tool = compute_r_tool(resp)
        preview = resp[:40].replace('\n', ' ')
        logger.info(f"  '{preview}...' → R_tool={r_tool:.3f}")
    
    # Test R_format
    logger.info("\n📝 R_format (Format Check):")
    tasks = [
        "<question>What is 2+2?</question>\n\\boxed{4}",
        "No tags here",
        "",
    ]
    for task in tasks:
        valid = check_format(task)
        preview = task[:30].replace('\n', ' ') if task else "(empty)"
        logger.info(f"  '{preview}...' → valid={valid}")
    
    logger.success("\n✅ Rewards module ready!")
