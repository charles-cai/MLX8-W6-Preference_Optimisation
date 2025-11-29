# Plan: Reproduce Agent0 with Qwen3-0.6B for Hackathon Weekend

Implement the Agent0 co-evolutionary framework (Curriculum Agent ↔ Executor Agent) from scratch, adapting the 8B-scale paper to Qwen3-0.6B. Leverage existing TRL/GRPO infrastructure from `.charles/trainings/` while building new curriculum generation and self-consistency components. Focus on single-iteration MVP with simplified tool integration.

## Architecture

```
Qwen3-0.6B-Base
      │
      ├──► Curriculum Agent (π_θ) + LoRA adapter
      │    - Trained via GRPO to generate challenging tasks
      │    - Reward: R_C (uncertainty + tool use)
      │
      └──► Executor Agent (π_φ) + LoRA adapter  
           - Trained via GRPO to solve tasks
           - Reward: Binary (matches majority vote pseudo-label)
```

**Training approach**: Full fine-tuning (48GB A6000 available) or LoRA (r=32) for faster iteration.
**RL algorithm**: GRPO (Group Relative Policy Optimization) - policy gradient without critic.

## Steps

1. **Create GRPO trainer module** in [`grpo.py`] - implement the clipped policy loss (Eq 1) with group-relative advantage normalization using TRL's GRPOTrainer.

2. **Build reward functions module** in [`rewards.py`] - implement `R_unc` (uncertainty reward, Eq 2), `R_tool` (tool use reward, Eq 3), and composite `R_C` (Eq 5); skip BLEU-based repetition penalty for MVP.

3. **Implement self-consistency utilities** in [`self_consistency.py`] - `compute_p_hat()` for majority voting over k=4 rollouts, answer extraction from `\boxed{}` format, and frontier task filtering where `|p̂ - 0.5| ≤ δ=0.25`.

4. **Create Curriculum Agent training script** in [`train_curriculum.py`] - GRPO training loop to generate math problems using prompts from Table 7; includes W&B logging, model checkpointing, and CSV/JSONL output of generated tasks.

5. **Create Executor Agent training script** in [`train_executor.py`] - filter tasks by self-consistency, train with GRPO using pseudo-labels from majority voting.

6. **Build minimal co-evolution orchestrator** in [`agent0_main.py`] - single iteration loop (T=1): freeze executor → train curriculum → freeze curriculum → filter dataset → train executor → evaluate on GSM8K subset.

## Further Considerations

1. **Tool integration depth?** Start with mock sandbox (binary reward for `<code>` tags), upgrade to `subprocess` execution with 5s timeout if time permits.

2. **Training scale trade-offs?** With 48GB A6000: full fine-tuning feasible, batch_size=8, k=4-10 rollouts, 40 steps per agent.

3. **Skip ADPO for MVP?** The dynamic advantage scaling adds complexity; use standard GRPO first, port ADPO in Phase 2 if metrics plateau.

4. **Data Science Direction?** Can customize curriculum prompts to generate data science tasks instead of pure math.

## Steps Checkboxes

- [x] Step 1: Create GRPO trainer module (`grpo.py`)
- [x] Step 2: Build reward functions module (`rewards.py`)
- [x] Step 3: Implement self-consistency utilities (`self_consistency.py`)
- [x] Step 4: Create Curriculum Agent training script (`train_curriculum.py`)
- [x] Step 5: Create Executor Agent training script (`train_executor.py`)
- [x] Step 6: Build co-evolution orchestrator (`main.py`)
