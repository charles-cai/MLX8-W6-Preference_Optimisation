# Agent0: Self-Evolving Agents via Tool-Integrated Reasoning

## Hackathon and Technical Challenges Description

- **Iterate - London Hackathon**: https://oil-delivery-e5d.notion.site/Iterate-London-Hackathon-2ae6aa77016a806db260ef41aade73e7

- **Technical Challenges Description**: https://oil-delivery-e5d.notion.site/Technical-Challenges-Description-2b86aa77016a805f9a1af76b52c8e98e


### Why RL? Why Now?

2025 is the year (or decade) AI moves from "talking" to "doing."
For the last two years, we've optimised for *plausibility* (does it sound right?). Now, we optimise for **verifiability** (did it work?).

Reinforcement Learning is the engine of this shift. It is crucial for problems where:

1. **Multiple solutions exist** (Creativity > Pattern Matching).
2. **No training data exists** (We can't clone human behaviour; we must discover new strategies).
3. **The environment is non-differentiable** (Black-box software, compilers, games, biology).

Every project in this hackathon should address some part of the loop:
`Agent` → `Action` → `Environment` → `Reward` → `Update`

Importantly, RL isn’t just about training. Rich environments with realistic and verifiable tasks are the new “gold” for data, and research and development in these areas is just as, or perhaps even more valuable. As such we’ve organised around 3 themes/tracks: Environments, Tasks, and Training.


> **Hackathon Project**: Reproducing Agent0 with Qwen3-0.6B for the MLX8 RL Hackathon


## 🎯 Hackathon Track Alignment

This project spans **all three hackathon tracks**:

| Track | How We Address It |
|-------|------------------|
| **Track 1: Building Environments** | Code interpreter sandbox for tool-integrated reasoning |
| **Track 2: Building Task Curricula** | Curriculum Agent learns to generate progressively harder tasks |
| **Track 3: Training Agents** | GRPO-based RL training of both Curriculum and Executor agents |

## 📄 Paper Summary

**Agent0** ([arXiv](https://arxiv.org/abs/2505.03335)) introduces a fully autonomous framework for evolving LLM agents **without any human-curated data**.

### Key Innovation: Co-Evolution Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    ITERATION t                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PHASE 1: Curriculum Evolution                        │   │
│  │ • Freeze Executor Agent                              │   │
│  │ • Train Curriculum Agent via GRPO                    │   │
│  │ • Reward: R_C = uncertainty + tool_use               │   │
│  │ • Goal: Generate maximally confusing tasks           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PHASE 2: Executor Evolution                          │   │
│  │ • Freeze Curriculum Agent                            │   │
│  │ • Generate task pool, filter frontier tasks          │   │
│  │ • Train Executor Agent via GRPO                      │   │
│  │ • Reward: Binary (matches majority vote pseudo-label)│   │
│  │ • Goal: Solve increasingly difficult tasks           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Two Agents from Same Base Model

Both agents are initialized from the **same base LLM** (Qwen3-0.6B in our case):

- **Curriculum Agent (π_θ)**: Generates challenging math/reasoning problems
- **Executor Agent (π_φ)**: Learns to solve those problems

### Asymmetric Training: Curriculum vs Executor

**Important**: The two agents are trained **asymmetrically**, not symmetrically:

| Aspect | Curriculum Agent | Executor Agent | Ratio |
|--------|------------------|----------------|-------|
| **Steps per iteration** | 5 | 40 | 1:8 |
| **Rollouts (k)** | 4 | 16 | 1:4 |
| **Batch size** | 128 | 128 | 1:1 |
| **Purpose** | Generate tasks | Solve tasks | - |
| **Training order** | First | Second | - |

**Why asymmetric?**
- **Curriculum Agent** only needs to learn a "style" of generating challenging problems
- **Executor Agent** must develop actual reasoning capabilities to solve problems
- Solving is harder than generating → needs more training

### Iteration Flow (T=3 in paper, T=1 for MVP)

```
Iteration 1:
├── Phase 1: Train Curriculum (5 steps) using frozen Executor⁰
├── Phase 2: Train Executor (40 steps) using frozen Curriculum¹
│   └── Filter frontier tasks where |p̂ - 0.5| ≤ 0.25
└── Output: Curriculum¹, Executor¹

Iteration 2:
├── Phase 1: Train Curriculum (5 steps) using frozen Executor¹
├── Phase 2: Train Executor (40 steps) using frozen Curriculum²
└── Output: Curriculum², Executor²

Iteration 3:
├── Phase 1: Train Curriculum (5 steps) using frozen Executor²
├── Phase 2: Train Executor (40 steps) using frozen Curriculum³
└── Output: Curriculum³, Executor³ (final)
```

### Reward Functions

| Reward | Equation | Purpose |
|--------|----------|---------|
| **R_unc** | `1 - 2|p̂ - 0.5|` | Maximized when executor is maximally uncertain (p̂ = 0.5) |
| **R_tool** | `γ · min(N_tool, C)` | Rewards tasks that prompt tool/code use |
| **R_C** | `R_format · max(0, λ_unc·R_unc + λ_tool·R_tool)` | Composite curriculum reward |

### Self-Consistency (p̂)

For each task, sample k responses from executor and compute majority vote:
- **p̂** = proportion voting for majority answer
- **Frontier tasks**: Keep only tasks where `|p̂ - 0.5| ≤ δ` (neither too easy nor too hard)

### Paper Results

On Qwen3-8B-Base:
- **+18%** on mathematical reasoning (MATH, GSM8K, AIME, etc.)
- **+24%** on general reasoning (MMLU-Pro, BBEH, SuperGPQA)

## 🔧 Our Implementation

### Hardware

- **GPU**: NVIDIA A6000 (48GB VRAM)
- **Model**: Qwen3-0.6B-Base (fits comfortably, allows full fine-tuning)

### Architecture

```
Qwen/Qwen3-0.6B (Base)
      │
      ├──► Curriculum Agent (π_θ)
      │    └── Trained to generate challenging tasks
      │
      └──► Executor Agent (π_φ)
           └── Trained to solve tasks with tool use
```

## 📊 Hyperparameter Comparison: Paper vs Hackathon

### Global Settings

| Parameter | Paper (8B) | Hackathon (0.6B) | Notes |
|-----------|------------|------------------|-------|
| Base Model | Qwen3-8B-Base | Qwen3-0.6B-Base | 13x smaller |
| Iterations (T) | 3 | 1 | MVP single iteration |
| Learning Rate | 1e-6 | 1e-6 | Same |
| Weight Decay | 1e-2 | 1e-2 | Same |
| KL Penalty (β) | 1e-2 | 1e-2 | Same |
| Rollout Temperature | 1.0 | 1.0 | Same |
| Rollout Top-p | 0.99 | 0.99 | Same |

### Curriculum Agent Training

| Parameter | Paper | Hackathon | Notes |
|-----------|-------|-----------|-------|
| Global Batch Size | 128 | 8 | Reduced for single GPU |
| Max Steps | 5 | 5-10 | Similar |
| Number of Rollouts (k) | 4 | 4 | Same |
| Tool Reward Scale (γ) | 0.6 | 0.6 | Same |
| Tool Reward Cap (C) | 4 | 4 | Same |
| λ_unc | 1.0 | 1.0 | Same |
| λ_tool | 0.6 | 0.6 | Same |

### Executor Agent Training

| Parameter | Paper | Hackathon | Notes |
|-----------|-------|-----------|-------|
| Global Batch Size | 128 | 8 | Reduced for single GPU |
| Max Steps | 40 | 20-40 | Similar |
| Number of Rollouts (k) | 16 | 4-8 | Reduced for speed |
| Frontier Filter (δ) | 0.25 | 0.25 | Same (p̂ ∈ [0.25, 0.75]) |
| ADPO | Yes | No | Skip for MVP |

### Reward Function Parameters

| Parameter | Paper | Hackathon | Notes |
|-----------|-------|-----------|-------|
| R_unc weight (λ_unc) | 1.0 | 1.0 | Same |
| R_tool weight (λ_tool) | 0.6 | 0.6 | Same |
| R_tool scale (γ) | 0.6 | 0.6 | Same |
| R_tool cap (C) | 4 | 4 | Same |
| R_rep (repetition) | BLEU-based | Skipped | Simplification |
| R_format | Required | Required | Same |

### What We Skip for MVP

| Feature | Paper | Hackathon | Reason |
|---------|-------|-----------|--------|
| ADPO | Ambiguity-scaled advantages | Standard GRPO | Adds ~4hrs implementation |
| R_rep | BLEU-based repetition penalty | None | Requires batch-level computation |
| Multi-turn | Up to 4 turns | Single turn | Simplification |
| Full Sandbox | subprocess execution | Mock (code block counting) | Safety/complexity |

## 📁 Files

| File | Description |
|------|-------------|
| `grpo.py` | GRPO trainer wrapper with Agent0 config defaults |
| `rewards.py` | R_unc, R_tool, R_C reward functions |
| `self_consistency.py` | p̂ computation, answer extraction, frontier filtering |
| `train_curriculum.py` | Curriculum Agent training with W&B logging |
| `train_executor.py` | Executor Agent training with frontier filtering |
| `main.py` | Co-evolution orchestrator |
| `agent0.py` | Package exports |

## 🚀 Quick Start

```bash
# Test individual modules
cd .charles/rl-training

uv run grpo.py              # Test GRPO trainer setup
uv run rewards.py           # Test reward functions
uv run self_consistency.py  # Test self-consistency utilities

# Run full co-evolution (1 iteration)
uv run main.py

# Run with custom settings
uv run main.py \
    --model_name Qwen/Qwen3-0.6B \
    --output_dir ./outputs \
    --iterations 1 \
    --curriculum_max_steps 5 \
    --executor_max_steps 40

# Train only curriculum agent
uv run main.py --curriculum_only

# Train only executor agent (requires trained curriculum)
uv run main.py --executor_only

# See all options
uv run main.py --help
```

### Individual Training Scripts

```bash
# Train curriculum agent directly
uv run train_curriculum.py \
    --model_name Qwen/Qwen3-0.6B \
    --output_dir ./outputs/curriculum \
    --max_steps 10

# Train executor agent directly
uv run train_executor.py \
    --curriculum_model_path ./outputs/curriculum/final_model \
    --output_dir ./outputs/executor \
    --max_steps 40
```

### Output Files

Training produces:
- `outputs/curriculum/curriculum_generations_*.jsonl` - Full generation logs
- `outputs/curriculum/curriculum_summary_*.csv` - Summary for inspection
- `outputs/curriculum/final_model/` - Trained curriculum model
- `outputs/executor/frontier_tasks_*.jsonl` - Filtered frontier tasks
- `outputs/executor/final_model/` - Trained executor model
- `outputs/coevolution_history.json` - Full training history

## 🎯 Hackathon Goals

### MVP (Weekend)
- [x] GRPO trainer module
- [x] Reward functions (R_unc, R_tool, R_C)
- [x] Self-consistency utilities
- [x] Curriculum Agent training script
- [x] Executor Agent training script
- [x] Single iteration co-evolution orchestrator

### Stretch Goals
- [ ] Real code sandbox with subprocess execution
- [ ] ADPO (Ambiguity-Dynamic Policy Optimization)
- [ ] Multi-turn interactions
- [ ] GSM8K evaluation
- [ ] Custom data science curriculum

## 📚 References

- **Paper**: [Agent0: Unleashing Self-Evolving Agents](https://arxiv.org/abs/2505.03335)
- **Code**: [github.com/aiming-lab/Agent0](https://github.com/aiming-lab/Agent0)
- **TRL**: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- **GRPO**: [DeepSeekMath paper](https://arxiv.org/abs/2402.03300)

## 🤝 Team

Built during MLX8 Week 6 - Preference Optimization & RL Hackathon.
