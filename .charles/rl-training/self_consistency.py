"""
Self-consistency utilities for Agent0.
Implements p̂ computation, answer extraction, and frontier task filtering.
"""

import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from datetime import datetime
from loguru import logger


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{...} format.
    
    Handles nested braces and common LaTeX formatting.
    
    Args:
        text: Model response containing \\boxed{answer}
    
    Returns:
        Extracted answer string, or None if not found
    """
    if text is None:
        return None
    
    # Pattern for \boxed{...} with nested brace handling
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last boxed answer (final answer convention)
        answer = matches[-1].strip()
        # Normalize whitespace
        answer = ' '.join(answer.split())
        return answer
    
    # Fallback: try simpler pattern
    simple_pattern = r'\\boxed\{([^}]+)\}'
    simple_matches = re.findall(simple_pattern, text)
    
    if simple_matches:
        return simple_matches[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Handles common variations in mathematical answers.
    """
    if answer is None:
        return ""
    
    # Remove whitespace
    normalized = answer.strip().lower()
    
    # Remove common LaTeX formatting
    normalized = normalized.replace('\\', '')
    normalized = normalized.replace(' ', '')
    normalized = normalized.replace(',', '')
    
    # Handle fractions: \frac{a}{b} -> a/b
    frac_pattern = r'frac\{(\d+)\}\{(\d+)\}'
    normalized = re.sub(frac_pattern, r'\1/\2', normalized)
    
    return normalized


def compute_p_hat(
    responses: List[str],
    return_majority: bool = True,
) -> Tuple[float, Optional[str], Dict[str, int]]:
    """
    Compute self-consistency score p̂ (Eq 6).
    
    p̂(x) = (1/k) * Σ I(σ_i = ỹ)
    
    where ỹ = argmax_y Σ I(σ_i = y) is the majority answer.
    
    Args:
        responses: List of k model responses
        return_majority: Whether to return the majority answer
    
    Returns:
        Tuple of (p_hat, majority_answer, vote_counts)
    """
    # Extract and normalize answers
    answers = []
    for resp in responses:
        raw_answer = extract_boxed_answer(resp)
        normalized = normalize_answer(raw_answer) if raw_answer else "__NO_ANSWER__"
        answers.append(normalized)
    
    # Count votes
    vote_counts = Counter(answers)
    
    # Find majority answer
    if not vote_counts:
        return 0.0, None, {}
    
    majority_answer, majority_count = vote_counts.most_common(1)[0]
    
    # Compute p̂
    k = len(responses)
    p_hat = majority_count / k if k > 0 else 0.0
    
    # Get original (non-normalized) majority answer for pseudo-label
    original_majority = None
    if return_majority and majority_answer != "__NO_ANSWER__":
        for resp in responses:
            raw = extract_boxed_answer(resp)
            if raw and normalize_answer(raw) == majority_answer:
                original_majority = raw
                break
    
    return p_hat, original_majority, dict(vote_counts)


def filter_frontier_tasks(
    tasks_with_scores: List[Dict[str, Any]],
    delta: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Filter tasks to keep only frontier tasks (Eq 7).
    
    D^(t) = {x ∈ X_pool | |p̂(x) - 0.5| ≤ δ}
    
    This keeps tasks that are neither too easy nor too hard.
    
    Args:
        tasks_with_scores: List of dicts with 'task', 'p_hat', 'majority_answer'
        delta: Threshold controlling difficulty band (default 0.25)
               Keeps tasks with p̂ ∈ [0.25, 0.75]
    
    Returns:
        Filtered list of frontier tasks
    """
    frontier = []
    
    for item in tasks_with_scores:
        p_hat = item.get('p_hat', 0.0)
        
        # Check if in frontier band
        if abs(p_hat - 0.5) <= delta:
            frontier.append(item)
            logger.debug(f"Task accepted: p̂={p_hat:.3f}")
        else:
            logger.debug(f"Task filtered out: p̂={p_hat:.3f}")
    
    logger.info(f"Frontier filtering: {len(frontier)}/{len(tasks_with_scores)} tasks retained (δ={delta})")
    
    return frontier


def save_generations_to_csv(
    generations: List[Dict[str, Any]],
    output_path: str,
    mode: str = 'a',  # append by default
) -> None:
    """
    Save generated tasks/responses to CSV for inspection.
    
    Args:
        generations: List of dicts with generation data
        output_path: Path to CSV file
        mode: 'w' for write, 'a' for append
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if we need to write header
    write_header = mode == 'w' or not output_path.exists()
    
    if not generations:
        logger.warning("No generations to save")
        return
    
    fieldnames = list(generations[0].keys())
    
    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        for gen in generations:
            # Truncate long text fields for readability
            row = {}
            for k, v in gen.items():
                if isinstance(v, str) and len(v) > 1000:
                    row[k] = v[:1000] + "..."
                else:
                    row[k] = v
            writer.writerow(row)
    
    logger.info(f"Saved {len(generations)} generations to {output_path}")


def save_generations_to_jsonl(
    generations: List[Dict[str, Any]],
    output_path: str,
    mode: str = 'a',
) -> None:
    """
    Save generated tasks/responses to JSONL for processing.
    
    Args:
        generations: List of dicts with generation data
        output_path: Path to JSONL file
        mode: 'w' for write, 'a' for append
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, mode, encoding='utf-8') as f:
        for gen in generations:
            # Add timestamp
            gen_with_ts = {**gen, 'timestamp': datetime.now().isoformat()}
            f.write(json.dumps(gen_with_ts, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(generations)} generations to {output_path}")


def load_generations_from_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """Load generations from JSONL file."""
    generations = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line))
    return generations


if __name__ == "__main__":
    # Test answer extraction
    print("Testing answer extraction...")
    
    test_cases = [
        r"The answer is \boxed{42}.",
        r"Therefore, $x = \boxed{\frac{1}{2}}$",
        r"First \boxed{wrong}, then \boxed{correct}",
        r"No boxed answer here",
        r"Complex: \boxed{x^2 + y^2 = 1}",
    ]
    
    for text in test_cases:
        answer = extract_boxed_answer(text)
        print(f"  '{text[:40]}...' → {answer}")
    
    # Test self-consistency
    print("\nTesting self-consistency (p̂)...")
    
    responses = [
        r"Working... \boxed{42}",
        r"I think \boxed{42}",
        r"The answer is \boxed{42}",
        r"Maybe \boxed{41}",  # Wrong answer
    ]
    
    p_hat, majority, votes = compute_p_hat(responses)
    print(f"  Responses: {len(responses)}")
    print(f"  Votes: {votes}")
    print(f"  p̂ = {p_hat:.3f}")
    print(f"  Majority answer: {majority}")
    
    # Test frontier filtering
    print("\nTesting frontier filtering...")
    
    tasks = [
        {'task': 'Easy task', 'p_hat': 0.9, 'majority_answer': '1'},
        {'task': 'Frontier task 1', 'p_hat': 0.5, 'majority_answer': '2'},
        {'task': 'Frontier task 2', 'p_hat': 0.6, 'majority_answer': '3'},
        {'task': 'Hard task', 'p_hat': 0.2, 'majority_answer': '4'},
    ]
    
    frontier = filter_frontier_tasks(tasks, delta=0.25)
    print(f"  Input: {len(tasks)} tasks")
    print(f"  Frontier (δ=0.25): {len(frontier)} tasks")
    for t in frontier:
        print(f"    - {t['task']}: p̂={t['p_hat']}")
    
    print("\n✅ Self-consistency module ready!")
