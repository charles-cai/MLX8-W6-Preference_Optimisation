# Plan: Agent1 - Mini Edge Data Scientist

> Evolved from Agent0 paper, adapted for Qwen3-0.6B with data science focus.

## Overview

Implement the Agent0 co-evolutionary framework (Curriculum Agent ↔ Executor Agent) for training a small data scientist model that can solve math, coding, and data analysis tasks.

**See [README.md](./README.md) for detailed usage instructions.**

## Architecture

```
Qwen3-0.6B-Base
      │
      ├──► Curriculum Agent (π_θ) - generates challenging tasks
      │
      └──► Executor Agent (π_φ) - learns to solve tasks
```

## Implementation Steps

- [x] Step 1: GRPO trainer module (`grpo.py`)
- [x] Step 2: Reward functions (`rewards.py`) - R_unc, R_tool, R_C
- [x] Step 3: Self-consistency utilities (`self_consistency.py`)
- [x] Step 4: Curriculum Agent training (`train_curriculum.py`)
- [x] Step 5: Executor Agent training (`train_executor.py`)
- [x] Step 6: Co-evolution orchestrator (`main.py`)
- [x] Step 7: Checkpoint management (`.env` config)
- [x] Step 8: Ablation study support (`ABLATION_MODE`)
- [ ] Step 9: Evaluation script (`eval.py`)
- [ ] Step 10: Real code sandbox with subprocess execution
- [ ] Step 11: Gradio demo (`demo.py`)

## Gradio Demo Plan

### Purpose
Demonstrate that co-evolution improves **both** agents:
1. **Executor** solves problems better than base model
2. **Curriculum** generates appropriately challenging tasks

### Demo Tabs

#### Tab 1: "Solve a Problem" (Main Demo)
Compare Base Model vs Trained Executor on the same problem.

```
┌─────────────────────────────────────────────────────────────┐
│  📝 Enter a math/data science problem:                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ [Text input for problem]                              │  │
│  └───────────────────────────────────────────────────────┘  │
│  [Solve] button                                             │
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │ 🔴 Base Model        │  │ 🟢 Trained Executor  │        │
│  │ (Qwen3-0.6B)         │  │ (After Co-Evolution) │        │
│  ├──────────────────────┤  ├──────────────────────┤        │
│  │ [Response]           │  │ [Response]           │        │
│  │ Answer: ___          │  │ Answer: ___          │        │
│  │ Confidence: ___%     │  │ Confidence: ___%     │        │
│  └──────────────────────┘  └──────────────────────┘        │
│                                                             │
│  ✅ Match / ❌ Mismatch    [Show diff in reasoning]         │
└─────────────────────────────────────────────────────────────┘
```

**What it shows**: Executor has improved reasoning ability over base model.

#### Tab 2: "Generate a Challenge" (Curriculum Demo)
Show Curriculum Agent generating frontier-difficulty tasks.

```
┌─────────────────────────────────────────────────────────────┐
│  🎯 Task Difficulty Target: [Slider: Easy → Hard]           │
│  📚 Domain: [Dropdown: Math / Data Science / Coding]        │
│  [Generate Task] button                                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Generated Task:                                        │  │
│  │ <question>...</question>                               │  │
│  │ \boxed{expected_answer}                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  📊 Task Analysis:                                          │
│  • Estimated p̂ (executor uncertainty): 0.52               │
│  • Tool calls expected: 2                                  │
│  • Difficulty: Frontier ✓                                  │
│                                                             │
│  [Send to Executor →] button                               │
└─────────────────────────────────────────────────────────────┘
```

**What it shows**: Curriculum learns to generate appropriately difficult tasks.

#### Tab 3: "Co-Evolution Loop" (End-to-End Demo)
Full pipeline: Curriculum generates → Executor solves → Show uncertainty.

```
┌─────────────────────────────────────────────────────────────┐
│  🔄 Co-Evolution Demo                                       │
│  [Run One Cycle] button                                     │
│                                                             │
│  Step 1: Curriculum generates task                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ [Generated Task]                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Step 2: Executor attempts (k=4 samples)                    │  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Response 1: \boxed{42}                                 │  │
│  │ Response 2: \boxed{42}                                 │  │
│  │ Response 3: \boxed{41}  ← different                    │  │
│  │ Response 4: \boxed{42}                                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Step 3: Compute metrics                                    │
│  • p̂ = 0.75 (3/4 agree on 42)                             │
│  • Majority answer: 42                                      │
│  • Is frontier? Yes (|0.75 - 0.5| ≤ 0.25)                  │
│  • R_unc = 1 - 2|0.75 - 0.5| = 0.5                         │
│                                                             │
│  💡 This task would be used for Executor training!          │
└─────────────────────────────────────────────────────────────┘
```

**What it shows**: The self-consistency mechanism and frontier filtering.

#### Tab 4: "Training Insights" (Results Dashboard)
Show training metrics and generated samples from logs.

```
┌─────────────────────────────────────────────────────────────┐
│  📈 Training History                                        │
│                                                             │
│  [Chart: Reward over training steps]                        │
│  [Chart: p̂ distribution over iterations]                   │
│  [Chart: Task difficulty progression]                       │
│                                                             │
│  📋 Sample Generated Tasks                                  │
│  [Table from curriculum_generations_*.jsonl]                │
│                                                             │
│  📋 Sample Executor Responses                               │
│  [Table from executor_generations_*.jsonl]                  │
└─────────────────────────────────────────────────────────────┘
```

### Technical Implementation

```python
# demo.py structure
import gradio as gr

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# Agent1: Mini Edge Data Scientist")
        
        with gr.Tabs():
            with gr.Tab("🧠 Solve a Problem"):
                # Base vs Executor comparison
                ...
            
            with gr.Tab("🎯 Generate a Challenge"):
                # Curriculum task generation
                ...
            
            with gr.Tab("🔄 Co-Evolution Loop"):
                # End-to-end demo
                ...
            
            with gr.Tab("📊 Training Insights"):
                # Results dashboard
                ...
    
    return demo
```

### Model Loading Strategy
- Load models lazily (on first use)
- Cache loaded models in memory
- Option to use CPU for demo (slower but works everywhere)

### Sample Problems (Pre-loaded)
Include 10-20 curated problems for quick demo:
- GSM8K samples (grade school math)
- MATH samples (competition math)  
- Data science problems (pandas/statistics)

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base Model | Qwen3-0.6B | Fits in 48GB, allows full fine-tuning |
| RL Algorithm | GRPO | Simpler than PPO, no critic needed |
| Iterations | T=1 (MVP) | Paper uses T=3, start simple |
| Tool Integration | Mock (code block counting) | Real sandbox adds complexity |
| ADPO | Skipped | Standard GRPO first, add later if needed |

## Ablation Study Plan

| Ablation | Variable | Values |
|----------|----------|--------|
| Baseline | - | Default config |
| No tool reward | `lambda_tool` | 0.0 |
| No uncertainty reward | `lambda_unc` | 0.0 |
| Higher k samples | `executor_k` | 4, 8, 16 |
| Math-only curriculum | `prompt_preset` | `math` vs `data_scientist` |
| Learning rate | `learning_rate` | 1e-6, 5e-7, 1e-5 |

## Timeline (Hackathon Weekend)

| Day | Focus |
|-----|-------|
| Day 1 | Core modules, curriculum training |
| Day 2 | Executor training, co-evolution loop |
| Day 3 | Ablations, evaluation, **Gradio demo**, presentation |

## References

- [Agent0 Paper](https://arxiv.org/abs/2505.03335)
- [README.md](./README.md) - Detailed usage
- [prompts.toml](./prompts.toml) - Prompt configurations
