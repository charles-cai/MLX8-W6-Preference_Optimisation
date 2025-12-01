# Agent1: Mini Edge Data Scientist

> Evolved from Agent0 paper - Self-evolving agents via tool-integrated reasoning

## 🏃 Quick Start

```bash
cd /workspace/_github/charles-cai/MLX8-W6-Preference_Optimisation/.charles/rl-training
```

### Step 1: Verify Modules Work (Required First!)

```bash
# Test individual modules - run ALL before any training
uv run rewards.py              # ✅ Tests R_unc, R_tool reward functions
uv run self_consistency.py     # ✅ Tests p̂ computation, answer extraction
uv run prompts.py              # ✅ Tests prompt loading from prompts.toml
uv run grpo.py                 # ✅ Tests GRPO trainer config (dry run)
```

Expected output: Each script should print "✅ ... module ready!" at the end.

### Step 2: Mini Test Run (Sanity Check)

**⚠️ Always run mini tests first to verify setup before formal training!**

```bash
# Mini test: Curriculum only (2 steps, 5 prompts) - ~5 min
uv run train_curriculum.py \
    --max_steps 2 \
    --num_prompts 5 \
    --num_generations 2 \
    --output_dir ./outputs/test_curriculum \
    --no_wandb

# Check outputs exist
ls -la ./outputs/test_curriculum/
cat ./outputs/test_curriculum/curriculum_generations_*.jsonl | head -2 | jq .

# Mini test: Executor only (2 steps, 10 tasks) - ~10 min
uv run train_executor.py \
    --max_steps 2 \
    --num_tasks 10 \
    --num_generations 2 \
    --k_samples 2 \
    --output_dir ./outputs/test_executor \
    --no_wandb

# Check outputs exist
ls -la ./outputs/test_executor/
```

### Step 3: Formal Training Runs

**Formal runs use default output directories and enable W&B logging.**

```bash
# Option A: Full co-evolution (recommended)
uv run main.py --iterations 1

# Option B: Train components separately
uv run train_curriculum.py --prompt_preset data_scientist
uv run train_executor.py --curriculum_model_path ./outputs/curriculum/final_model
```

---

## 💾 GPU Memory Requirements

### Model Size vs GPU Memory (A6000 48GB Reference)

| Model | Full Fine-Tune | With LoRA (r=32) | With LoRA + GradChkpt | Recommendation |
|-------|---------------|------------------|----------------------|----------------|
| **Qwen3-0.6B** | ~12-15 GB ✅ | ~8-10 GB | ~6-8 GB | Full FT works |
| **Qwen3-1.7B** | ~30-35 GB ✅ | ~18-22 GB | ~14-18 GB | Full FT (tight) |
| **Qwen3-4B** | ~65-75 GB ❌ | ~28-35 GB ✅ | ~22-28 GB ✅ | **Use LoRA** |
| **Qwen3-8B** | ~120+ GB ❌ | ~45-55 GB ⚠️ | ~35-45 GB ✅ | LoRA + GradChkpt |

### Memory Breakdown (Full Fine-Tuning)

```
Total = Model Weights + Optimizer States + Gradients + Activations
      = 2×params (bf16) + 8×params (AdamW) + 2×params + activations

Example for 4B model:
  Weights:     4B × 2 bytes = 8 GB
  Optimizer:   4B × 8 bytes = 32 GB (momentum + variance + master weights)
  Gradients:   4B × 2 bytes = 8 GB
  Activations: ~15-20 GB (depends on batch/seq length)
  Total:       ~65-75 GB → EXCEEDS 48GB!
```

### Commands for Different GPU Sizes

#### A6000 48GB - Qwen3-4B (LoRA Required)

```bash
# 4B model with LoRA (recommended for A6000)
uv run train_curriculum.py \
    --model_name Qwen/Qwen3-4B \
    --use_lora \
    --lora_r 32 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 8

# Monitor GPU memory
watch -n 1 nvidia-smi
```

#### A6000 48GB - Qwen3-8B (LoRA + Aggressive Optimization)

```bash
# 8B model - very tight, may OOM
uv run train_curriculum.py \
    --model_name Qwen/Qwen3-8B \
    --use_lora \
    --lora_r 16 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 16

# Set environment for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### Multi-GPU (2x A6000 = 96GB)

```bash
# 4B full fine-tuning becomes possible
accelerate launch --num_processes 2 train_curriculum.py \
    --model_name Qwen/Qwen3-4B \
    --per_device_batch_size 2
```

### Expected Training Times (A6000 48GB)

| Model | Sanity Test (2 steps) | Formal Curriculum (5 steps) | Formal Executor (40 steps) |
|-------|----------------------|-----------------------------|-----------------------------|
| **0.6B** | ~5 min | ~20 min | ~2-3 hours |
| **1.7B** | ~10 min | ~40 min | ~4-5 hours |
| **4B + LoRA** | ~15 min | ~1 hour | ~6-8 hours |
| **8B + LoRA** | ~25 min | ~2 hours | ~12-16 hours |

---

## 📊 Data Flow & Weight Progression

### Single Iteration Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ITERATION 1                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: Curriculum Evolution                                               │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│    Curriculum Model (TRAINABLE)        Executor Model (FROZEN)               │
│    ┌──────────────────────┐           ┌──────────────────────┐              │
│    │ Input: base_model    │           │ Input: base_model    │              │
│    │ (Qwen/Qwen3-0.6B)    │           │ (Qwen/Qwen3-0.6B)    │              │
│    └──────────┬───────────┘           └──────────┬───────────┘              │
│               │                                   │                          │
│               ▼                                   │                          │
│    ┌──────────────────────┐                      │                          │
│    │ Generate tasks       │◄─────────────────────┤                          │
│    │ (num_prompts=100)    │    Compute p̂ for    │                          │
│    └──────────┬───────────┘    each task         │                          │
│               │                                   │                          │
│               ▼                                   │                          │
│    ┌──────────────────────┐                                                 │
│    │ Compute R_C reward:  │                                                 │
│    │ • R_unc (uncertainty)│                                                 │
│    │ • R_tool (code use)  │                                                 │
│    └──────────┬───────────┘                                                 │
│               │                                                              │
│               ▼                                                              │
│    ┌──────────────────────┐                                                 │
│    │ GRPO Update          │                                                 │
│    │ (max_steps=5)        │                                                 │
│    └──────────┬───────────┘                                                 │
│               │                                                              │
│               ▼                                                              │
│    ┌──────────────────────┐                                                 │
│    │ OUTPUT:              │                                                 │
│    │ curriculum_iter1/    │                                                 │
│    │   final_model/       │◄──── Trained curriculum weights                 │
│    │   *.jsonl logs       │◄──── Generation logs                            │
│    └──────────────────────┘                                                 │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 2: Executor Evolution                                                 │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│    Curriculum Model (FROZEN)           Executor Model (TRAINABLE)            │
│    ┌──────────────────────┐           ┌──────────────────────┐              │
│    │ Input: curriculum_   │           │ Input: base_model    │              │
│    │ iter1/final_model    │           │ (Qwen/Qwen3-0.6B)    │              │
│    └──────────┬───────────┘           └──────────┬───────────┘              │
│               │                                   │                          │
│               ▼                                   │                          │
│    ┌──────────────────────┐                      │                          │
│    │ Generate task pool   │                      │                          │
│    │ (num_tasks=200)      │                      │                          │
│    └──────────┬───────────┘                      │                          │
│               │                                   │                          │
│               ▼                                   ▼                          │
│    ┌─────────────────────────────────────────────────────┐                  │
│    │ Compute p̂ for each task using Executor samples     │                  │
│    │ Filter frontier tasks: |p̂ - 0.5| ≤ δ               │                  │
│    └──────────┬──────────────────────────────────────────┘                  │
│               │                                                              │
│               ▼                                                              │
│    ┌──────────────────────┐                                                 │
│    │ Frontier Dataset     │                                                 │
│    │ with pseudo-labels   │                                                 │
│    │ (majority answers)   │                                                 │
│    └──────────┬───────────┘                                                 │
│               │                                                              │
│               ▼                                                              │
│    ┌──────────────────────┐                                                 │
│    │ GRPO Update Executor │                                                 │
│    │ (max_steps=40)       │                                                 │
│    │ Reward: correctness  │                                                 │
│    └──────────┬───────────┘                                                 │
│               │                                                              │
│               ▼                                                              │
│    ┌──────────────────────┐                                                 │
│    │ OUTPUT:              │                                                 │
│    │ executor_iter1/      │                                                 │
│    │   final_model/       │◄──── Trained executor weights                   │
│    │   frontier_tasks_*   │◄──── Filtered training tasks                    │
│    │   all_tasks_scored_* │◄──── All tasks with p̂ scores                   │
│    └──────────────────────┘                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Iteration Weight Flow

```
ITERATION 1:
  Curriculum: base_model ──► curriculum_iter1/final_model
  Executor:   base_model ──► executor_iter1/final_model

ITERATION 2:
  Curriculum: curriculum_iter1/final_model ──► curriculum_iter2/final_model
  Executor:   executor_iter1/final_model   ──► executor_iter2/final_model

ITERATION 3:
  Curriculum: curriculum_iter2/final_model ──► curriculum_iter3/final_model
  Executor:   executor_iter2/final_model   ──► executor_iter3/final_model
```

### Generated Artifacts by Phase

| Phase | File | Contents |
|-------|------|----------|
| **Curriculum** | `curriculum_generations_*.jsonl` | Tasks, executor responses, rewards, p̂ |
| **Curriculum** | `curriculum_summary_*.csv` | Condensed metrics per task |
| **Curriculum** | `final_model/` | Trained curriculum weights |
| **Curriculum** | `training_info.json` | Hyperparameters used |
| **Executor** | `all_tasks_scored_*.jsonl` | All tasks with p̂ scores |
| **Executor** | `frontier_tasks_*.jsonl` | Filtered frontier tasks |
| **Executor** | `task_summary_*.csv` | Task filtering summary |
| **Executor** | `executor_generations_*.jsonl` | Training logs with rewards |
| **Executor** | `final_model/` | Trained executor weights |
| **Executor** | `training_info.json` | Hyperparameters used |

---

## 🧪 Test vs Formal Runs

### Test Runs (Development/Debugging)

**Purpose**: Verify code works before committing to long training.

```bash
# ALWAYS use separate output_dir for test runs!
uv run train_curriculum.py \
    --output_dir ./outputs/test_curriculum \   # ← Separate directory!
    --max_steps 2 \
    --num_prompts 5 \
    --no_wandb                                  # ← No W&B for tests

uv run train_executor.py \
    --output_dir ./outputs/test_executor \     # ← Separate directory!
    --max_steps 2 \
    --num_tasks 10 \
    --no_wandb
```

### Formal Runs (Actual Training)

**Purpose**: Full training with proper logging.

```bash
# Default directories, W&B enabled
uv run train_curriculum.py                     # → ./outputs/curriculum/
uv run train_executor.py                       # → ./outputs/executor/

# Or via main.py for multi-iteration
uv run main.py --iterations 3                  # → ./outputs/{curriculum,executor}_iter{1,2,3}/
```

### Clean Test Outputs Before Formal Runs

```bash
# Remove test outputs to avoid confusion
rm -rf ./outputs/test_*

# Verify no leftover test files
ls ./outputs/
```

---

## 🔄 Full Pipeline Commands

### Option 1: Co-Evolution via main.py (Recommended)

```bash
# Single iteration (quick, ~1-2 hours)
uv run main.py --iterations 1

# Paper default: 3 iterations (~6-8 hours)
uv run main.py --iterations 3

# Custom settings
uv run main.py \
    --iterations 2 \
    --curriculum_max_steps 5 \
    --executor_max_steps 40 \
    --num_prompts 100 \
    --num_tasks 200 \
    --learning_rate 1e-6
```

### Option 2: Train Components Separately

```bash
# Step 1: Train curriculum (generates task-setting ability)
uv run train_curriculum.py \
    --prompt_preset data_scientist \
    --max_steps 5 \
    --num_prompts 100

# Step 2: Verify curriculum output
ls ./outputs/curriculum/final_model/
cat ./outputs/curriculum/curriculum_generations_*.jsonl | jq -s 'length'

# Step 3: Train executor (uses trained curriculum)
uv run train_executor.py \
    --curriculum_model_path ./outputs/curriculum/final_model \
    --max_steps 40 \
    --num_tasks 200

# Step 4: Verify executor output
ls ./outputs/executor/final_model/
cat ./outputs/executor/frontier_tasks_*.jsonl | jq -s 'length'
```

---

## 🔬 Ablation Studies

### Running Ablations

Ablations use separate output directories to avoid conflicts.

```bash
# Baseline (default config)
uv run train_curriculum.py --output_dir ./outputs/ablations/baseline/curriculum

# No tool reward (λ_tool = 0)
uv run train_curriculum.py \
    --lambda_tool 0.0 \
    --output_dir ./outputs/ablations/no_tool_reward/curriculum

# No uncertainty reward (λ_unc = 0)
uv run train_curriculum.py \
    --lambda_unc 0.0 \
    --output_dir ./outputs/ablations/no_unc_reward/curriculum

# Higher k samples for self-consistency
uv run train_curriculum.py \
    --executor_k 8 \
    --output_dir ./outputs/ablations/k8_samples/curriculum

# Math-only curriculum (paper default prompts)
uv run train_curriculum.py \
    --prompt_preset math \
    --output_dir ./outputs/ablations/math_only/curriculum

# Different learning rate
uv run train_curriculum.py \
    --learning_rate 5e-7 \
    --output_dir ./outputs/ablations/lr_5e7/curriculum
```

### Alternative: Use Environment Variables

```bash
ABLATION_MODE=true ABLATION_NAME=no_tool_reward \
    uv run train_curriculum.py --lambda_tool 0.0
# Output: ./outputs/ablations/no_tool_reward/curriculum/
```

---

## 📂 Output Directory Structure

```
./outputs/
├── test_curriculum/              # ← Test runs (DELETE before formal)
├── test_executor/                # ← Test runs (DELETE before formal)
│
├── curriculum/                   # ← Single-run curriculum output
│   ├── final_model/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer files...
│   ├── checkpoints/
│   │   └── checkpoint-{step}/
│   ├── curriculum_generations_*.jsonl
│   ├── curriculum_summary_*.csv
│   └── training_info.json
│
├── executor/                     # ← Single-run executor output
│   ├── final_model/
│   ├── checkpoints/
│   ├── all_tasks_scored_*.jsonl
│   ├── frontier_tasks_*.jsonl
│   ├── task_summary_*.csv
│   ├── executor_generations_*.jsonl
│   └── training_info.json
│
├── curriculum_iter1/             # ← main.py iteration 1
├── executor_iter1/
├── curriculum_iter2/             # ← main.py iteration 2
├── executor_iter2/
├── curriculum_iter3/             # ← main.py iteration 3
├── executor_iter3/
├── coevolution_history.json      # ← main.py summary
│
└── ablations/                    # ← Ablation experiments
    ├── baseline/
    ├── no_tool_reward/
    ├── no_unc_reward/
    ├── k8_samples/
    ├── math_only/
    └── lr_5e7/
```

---

## 🔍 Inspecting Outputs

### View Generated Tasks (Curriculum)

```bash
# See first 3 generated tasks
cat ./outputs/curriculum/curriculum_generations_*.jsonl | head -3 | jq .

# Count total tasks
cat ./outputs/curriculum/curriculum_generations_*.jsonl | jq -s 'length'

# See reward distribution
cat ./outputs/curriculum/curriculum_summary_*.csv | head -10
```

### View Frontier Tasks (Executor)

```bash
# Count frontier tasks (used for training)
cat ./outputs/executor/frontier_tasks_*.jsonl | jq -s 'length'

# See p̂ distribution
cat ./outputs/executor/task_summary_*.csv | csvcut -c p_hat | tail -n +2 | sort -n | uniq -c

# View a sample frontier task
cat ./outputs/executor/frontier_tasks_*.jsonl | head -1 | jq '.question, .p_hat, .majority_answer'
```

### View Training Info

```bash
# Check curriculum training config
cat ./outputs/curriculum/training_info.json | jq .

# Check executor training config
cat ./outputs/executor/training_info.json | jq .

# Check co-evolution history
cat ./outputs/coevolution_history.json | jq '.iterations'
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Core settings
MODEL_ID=Qwen/Qwen3-0.6B
PROMPT_PRESET=data_scientist

# Training
CURRICULUM_MAX_STEPS=5
EXECUTOR_MAX_STEPS=40
LEARNING_RATE=1e-6

# Reward parameters (from paper Table 8)
LAMBDA_UNC=1.0          # Uncertainty reward weight
LAMBDA_TOOL=0.6         # Tool use reward weight
GAMMA_TOOL=0.6          # Tool reward scale
CAP_TOOL=4              # Max rewarded tool calls
DELTA=0.25              # Frontier threshold: |p̂ - 0.5| ≤ δ

# Sampling
EXECUTOR_K_SAMPLES=4    # k responses for p̂ computation (paper: 10)

# Checkpoints
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=3

# W&B
WANDB_PROJECT=rl-hackathon-agent1
WANDB_DIR=./.wandb
WANDB_RUN_PREFIX=agent1

# Output
OUTPUT_DIR=./outputs
```

### Prompt Presets (`prompts.toml`)

| Preset | Description | Use Case |
|--------|-------------|----------|
| `math` | Competition math (paper default) | Pure math reasoning |
| `data_scientist` | Math + data analysis + coding | **Hackathon focus** |
| `coding` | Algorithmic challenges | Code generation |
| `reasoning` | Logic puzzles | General reasoning |

---

## 📁 File Reference

| File | Description |
|------|-------------|
| `main.py` | Co-evolution orchestrator (runs both agents) |
| `train_curriculum.py` | Curriculum agent training |
| `train_executor.py` | Executor agent training |
| `grpo.py` | GRPO trainer configuration |
| `rewards.py` | R_unc, R_tool, R_C reward functions |
| `self_consistency.py` | p̂ computation, answer extraction |
| `prompts.py` | Prompt loading utilities |
| `prompts.toml` | Prompt configurations |
| `.env` | Environment configuration |

---

## 🎯 Hackathon Track Alignment

| Track | How We Address It |
|-------|-------------------|
| **Track 1: Building Environments** | Code interpreter sandbox for tool-integrated reasoning |
| **Track 2: Building Task Curricula** | Curriculum Agent learns to generate progressively harder tasks |
| **Track 3: Training Agents** | GRPO-based RL training of both Curriculum and Executor agents |

---

## 📄 Paper Reference

**Agent0** ([arXiv](https://arxiv.org/abs/2505.03335)) - Key equations:

| Reward | Formula | Purpose |
|--------|---------|---------|
| **R_unc** | `1 - 2|p̂ - 0.5|` | Maximize executor uncertainty |
| **R_tool** | `γ · min(N_tool, C)` | Reward tool/code use |
| **R_C** | `R_format · max(0, λ_unc·R_unc + λ_tool·R_tool)` | Composite curriculum reward |

### Training Asymmetry (Paper Table 8)

| Aspect | Curriculum | Executor | Ratio |
|--------|------------|----------|-------|
| Steps/iteration | 5 | 40 | 1:8 |
| Rollouts (k) | 4 | 16 | 1:4 |
| Purpose | Generate tasks | Solve tasks | - |

---

## Curriculum Agent Training: Sanity Check

This section documents a quick validation run to verify the curriculum training pipeline works correctly.

### Command Executed

```bash
uv run train_curriculum.py --max_steps 2 --num_prompts 8 --no_wandb
```

### Configuration Summary

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-1.7B` |
| Training Steps | 2 |
| Num Prompts | 8 |
| Num Generations | 2 |
| Per Device Batch Size | 4 |
| Gradient Accumulation Steps | 2 |
| Executor K (self-consistency samples) | 2 |
| Enable Thinking | `false` |
| Wandb | Disabled |

### Training Results

| Metric | Batch 1 | Batch 2 | Trend |
|--------|---------|---------|-------|
| **Avg Reward** | 0.215 | 0.270 | ↑ Improving |
| **Loss** | 0.127 | 0.037 | ↓ Good |
| **Grad Norm** | 2.92 | 1.77 | ↓ Stabilizing |
| **Completion Length (mean)** | 283 tokens | 368 tokens | ↑ More detailed |
| **Entropy** | 0.298 | 0.227 | ↓ More confident |

### Timing

| Phase | Duration |
|-------|----------|
| Model Loading (curriculum) | ~2s |
| Model Loading (executor) | ~2s |
| Step 1 (generation + rewards) | ~2m 22s |
| Step 2 (generation + rewards) | ~2m 53s |
| Model Saving | ~3s |
| **Total Runtime** | **~5m 28s** |

### Generated Tasks Sample

The curriculum agent successfully generated data science tasks. Example from batch 1:

```
<question>
A company collects data on customer satisfaction scores (1-10) and their 
purchase history (number of purchases in the last month). The data is stored 
in a CSV file named `customer_data.csv`. The task is to analyze the relationship 
between satisfaction scores and purchase history using statistical methods 
and Python.

**Data Description:**
- Satisfaction score (1-10) is a continuous variable.
- Purchase history (number of purchases) is a categorical variable...

**Steps to Solve:**
1. Load the data and perform EDA.
2. Handle missing values and outliers.
3. Perform a regression analysis...
</question>
```

### Reward Distribution Analysis

| Status | Count | Percentage |
|--------|-------|------------|
| Success (format valid) | 16 | 100% |
| Reward > 0 | 5 | 31% |
| Reward = 1.0 (max uncertainty) | 1 | 6% |
| Reward = 0 (executor confident) | 11 | 69% |

**Key Insight**: Most tasks received reward=0 because the executor was highly confident (p̂ = 1.0). The curriculum agent needs more training to generate tasks at the "frontier" of executor capability.

### Output Files

| File | Location | Content |
|------|----------|---------|
| Generation logs (JSONL) | `outputs/curriculum/curriculum_generations_*_final.jsonl` | Full task logs with executor responses |
| Summary (CSV) | `outputs/curriculum/curriculum_summary_*_final.csv` | Condensed metrics per task |
| Final model | `outputs/curriculum/final_model/` | Trained curriculum model weights |
| Training info | `outputs/curriculum/training_info.json` | Configuration and metadata |

### View Results

```bash
# View summary CSV
cat outputs/curriculum/curriculum_summary_*.csv

# View detailed JSONL (first entry, formatted)
head -1 outputs/curriculum/curriculum_generations_*.jsonl | jq .

# Check generated tasks
cat outputs/curriculum/curriculum_generations_*.jsonl | jq -r '.task' | head -100
```

### Console Output (Full)

<details>
<summary>Click to expand full training log</summary>

```
17:31:14 | INFO     | prompts:load_prompts - 📝 Loaded prompt preset: Mini Edge Data Scientist
17:31:14 | INFO     | __main__:train_curriculum_agent - ============================================================
17:31:14 | INFO     | __main__:train_curriculum_agent - 🚀 CURRICULUM AGENT TRAINING
17:31:14 | INFO     | __main__:train_curriculum_agent -    Prompt Preset: Mini Edge Data Scientist
17:31:14 | INFO     | __main__:train_curriculum_agent - ============================================================
17:31:16 | SUCCESS  | __main__:load_model_and_tokenizer - ✅ Model loaded: 1,720,574,976 total params
17:31:17 | INFO     | __main__:train_curriculum_agent - 🔒 Executor model frozen
17:31:17 | SUCCESS  | __main__:create_curriculum_prompts_dataset - ✅ Empty thinking tags detected
17:31:17 | INFO     | grpo:create_grpo_trainer - Config: k=2, batch=4, grad_accum=2, max_completion=2048
17:33:38 | INFO     | __main__:__call__ - 📊 Batch 1: avg_reward=0.215, samples=8
17:36:21 | INFO     | __main__:__call__ - 📊 Batch 2: avg_reward=0.270, samples=8
17:36:33 | SUCCESS  | __main__:save_logs - 💾 Saved 16 logs to outputs/curriculum
17:36:36 | SUCCESS  | __main__:train_curriculum_agent - ✅ Curriculum agent training complete!
```

</details>

### Validation Checklist

- [x] Model loads correctly (Qwen3-1.7B)
- [x] Thinking mode disabled (no `<think>` tags in output)
- [x] Tasks generated with correct format (`<question>` tags)
- [x] Executor samples responses for self-consistency
- [x] Rewards computed (R_unc + R_tool)
- [x] GRPO training step completes
- [x] Model saved successfully
- [x] Generation logs saved

---

## Executor Agent Training: Sanity Check

This section documents the first executor training iteration using tasks generated by the trained curriculum agent.

### What Gets Passed to Executor Training

Per the Agent0 paper (Algorithm 1, Lines 11-24), the executor receives:

| Input | Description |
|-------|-------------|
| **Curriculum Model** | Frozen model to generate task pool |
| **Frontier Tasks** | Tasks filtered by self-consistency (p̂ ∈ [δ, 1-δ]) |
| **Pseudo-Labels** | Majority-vote answers from k executor samples |
| **Self-Consistency Scores (p̂)** | Used for ADPO advantage scaling |

The training pipeline:
1. Curriculum generates task pool → 2. Score each task with k samples → 3. Filter frontier tasks → 4. Train executor on frontier tasks with pseudo-labels

### Command Executed

```bash
uv run train_executor.py \
    --curriculum_model_path ./outputs/curriculum/final_model \
    --max_steps 2 \
    --num_tasks 8 \
    --no_wandb
```

### Configuration Summary

| Setting | Value |
|---------|-------|
| Executor Model | `Qwen/Qwen3-1.7B` (trainable) |
| Curriculum Model | `./outputs/curriculum/final_model` (frozen) |
| Training Steps | 2 |
| Num Tasks | 8 |
| K Samples | 2 |
| Delta (frontier threshold) | 0.25 |
| Num Generations | 2 |
| Per Device Batch Size | 4 |
| Gradient Accumulation Steps | 2 |
| Enable Thinking | `false` |

### Training Results

| Metric | Batch 1 | Batch 2 | Notes |
|--------|---------|---------|-------|
| **Avg Reward** | 0.000 | 0.000 | No correct answers |
| **Accuracy** | 0.0% | 0.0% | Executor didn't match pseudo-labels |
| **Loss** | 0.0 | 0.0 | No gradient signal |
| **Completion Length** | 2048 | 2048 | Hit max length (clipped) |
| **Clipped Ratio** | 1.0 | 1.0 | All completions truncated |

### Timing

| Phase | Duration |
|-------|----------|
| Model Loading (executor) | ~2s |
| Model Loading (curriculum) | ~2s |
| Task Generation (8 tasks) | ~59s |
| Self-Consistency Scoring (p̂) | ~7m 48s |
| Training Step 1 | ~1m 22s |
| Training Step 2 | ~1m 32s |
| Model Saving | ~3s |
| **Total Runtime** | **~11m 55s** |

### Frontier Task Filtering Results

| Metric | Value |
|--------|-------|
| Total tasks generated | 8 |
| Tasks after frontier filtering | 8 |
| Retention rate | 100% |
| p̂ range observed | All tasks at p̂ ≈ 0.5 |

**Observation**: All 8 tasks had p̂ ≈ 0.5 (maximum uncertainty), meaning the executor was equally split on answers. This is expected for challenging data science tasks on first iteration.

### Why Reward = 0?

The logs show `avg_reward=0.000` and `accuracy=0.0%`. This happened because:

1. **Completions hit max length** (`completions/clipped_ratio: 1.0`) - responses were truncated at 2048 tokens
2. **No answer extraction** - truncated responses didn't contain `\boxed{...}` answers
3. **Pseudo-label mismatch** - complex LaTeX pseudo-labels are hard to match exactly

**Fix for future runs**: Increase `max_completion_length` or use simpler math tasks instead of data science tasks.

### Output Files

| File | Location | Content |
|------|----------|---------|
| All tasks scored | `outputs/executor/all_tasks_scored_*.jsonl` | All 8 tasks with p̂ and responses |
| Frontier tasks | `outputs/executor/frontier_tasks_*.jsonl` | Filtered tasks for training |
| Task summary | `outputs/executor/task_summary_*.csv` | Condensed view of tasks |
| Training logs | `outputs/executor/executor_generations_*_final.jsonl` | Full training outputs |
| Training summary | `outputs/executor/executor_summary_*_final.csv` | Metrics per sample |
| Final model | `outputs/executor/final_model/` | Trained executor weights |
| Training info | `outputs/executor/training_info.json` | Configuration metadata |

### View Results

```bash
# View task summary (frontier tasks)
cat outputs/executor/task_summary_*.csv

# View executor training summary
cat outputs/executor/executor_summary_*_final.csv

# View detailed frontier tasks
head -1 outputs/executor/frontier_tasks_*.jsonl | jq .

# Check pseudo-labels used
cat outputs/executor/frontier_tasks_*.jsonl | jq -r '.majority_answer' | head -5
```

### Console Output (Full)

<details>
<summary>Click to expand full training log</summary>

```
17:42:32 | INFO     | prompts:load_prompts - 📝 Loaded prompt preset: Mini Edge Data Scientist
17:42:32 | INFO     | __main__:train_executor_agent - 🚀 EXECUTOR AGENT TRAINING
17:42:34 | SUCCESS  | __main__:load_model_and_tokenizer - ✅ Model loaded: 1,720,574,976 total params
17:42:35 | INFO     | __main__:train_executor_agent - 🔒 Curriculum model frozen
17:43:34 | SUCCESS  | __main__:generate_task_pool - ✅ Generated 8 tasks
17:51:23 | INFO     | self_consistency:filter_frontier_tasks - Frontier filtering: 8/8 tasks retained (δ=0.25)
17:51:23 | INFO     | __main__:curate_frontier_dataset - 📊 Frontier filtering complete:
17:51:23 | INFO     | __main__:curate_frontier_dataset -   Total tasks processed: 8
17:51:23 | INFO     | __main__:curate_frontier_dataset -   Frontier tasks retained: 8
17:51:23 | INFO     | __main__:curate_frontier_dataset -   Retention rate: 100.0%
17:51:23 | SUCCESS  | __main__:create_executor_training_dataset - ✅ Created executor training dataset with 8 samples
17:51:23 | INFO     | __main__:set_pseudo_labels - 📋 Set 8 pseudo-labels for reward computation
17:51:23 | INFO     | grpo:create_grpo_trainer - 🔧 Override model generation_config: temperature=1.0, top_p=0.99
17:52:38 | INFO     | __main__:__call__ - 📊 Batch 1: avg_reward=0.000, accuracy=0.0%
17:54:02 | INFO     | __main__:__call__ - 📊 Batch 2: avg_reward=0.000, accuracy=0.0%
17:54:18 | SUCCESS  | __main__:save_logs - 💾 Saved 16 logs
17:54:21 | SUCCESS  | __main__:train_executor_agent - 💾 Saved final model to outputs/executor/final_model
17:54:21 | SUCCESS  | __main__:train_executor_agent - ✅ Executor agent training complete!
```

</details>

### Validation Checklist

- [x] Curriculum model loads from checkpoint
- [x] Task pool generated successfully (8 tasks)
- [x] Self-consistency computed (k=2 samples per task)
- [x] Frontier filtering applied (δ=0.25)
- [x] Pseudo-labels extracted via majority voting
- [x] Executor training dataset created
- [x] GRPO training completes
- [x] Model saved successfully
- [ ] Non-zero rewards (needs longer completions or simpler tasks)
- [ ] Accuracy > 0 (needs better answer extraction)

---

## Co-Evolution: Next Steps

After completing one iteration of both agents, continue the co-evolutionary loop:

```bash
# Iteration 2: Train curriculum with updated executor
uv run train_curriculum.py \
    --executor_model_path ./outputs/executor/final_model \
    --output_dir ./outputs/curriculum_iter2 \
    --max_steps 5 \
    --no_wandb

# Iteration 2: Train executor with updated curriculum
uv run train_executor.py \
    --curriculum_model_path ./outputs/curriculum_iter2/final_model \
    --output_dir ./outputs/executor_iter2 \
    --max_steps 10 \
    --no_wandb
```

Per the Agent0 paper, this co-evolutionary loop drives improvement in both agents over multiple iterations.

---

## Sanity Check Results

### Curriculum Agent Training (Iteration 1)

**Date:** 2024-11-29 17:31-17:36

#### Command

```bash
uv run train_curriculum.py --max_steps 2 --num_prompts 8 --no_wandb
```

#### Configuration

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3-1.7B` |
| Training Steps | 2 |
| Num Prompts | 8 |
| Num Generations | 2 |
| Per Device Batch Size | 4 |
| Gradient Accumulation | 2 |
| Executor K | 2 |
| Enable Thinking | `false` |

#### Training Results

| Metric | Batch 1 | Batch 2 | Trend |
|--------|---------|---------|-------|
| **Avg Reward** | 0.215 | 0.270 | ↑ Improving |
| **Loss** | 0.127 | 0.037 | ↓ Good |
| **Grad Norm** | 2.92 | 1.77 | ↓ Stabilizing |
| **Completion Length** | 283 | 368 | ↑ More detailed |
| **Entropy** | 0.298 | 0.227 | ↓ More confident |

#### Timing

| Phase | Duration |
|-------|----------|
| Model Loading (curriculum) | ~2s |
| Model Loading (executor) | ~2s |
| Step 1 (gen + rewards) | ~2m 22s |
| Step 2 (gen + rewards) | ~2m 53s |
| Model Saving | ~3s |
| **Total** | **~5m 28s** |

#### Reward Distribution

| Status | Count | % |
|--------|-------|---|
| Success (format valid) | 16 | 100% |
| Reward > 0 | 5 | 31% |
| Reward = 1.0 (max) | 1 | 6% |
| Reward = 0 | 11 | 69% |

**Insight:** Most tasks got reward=0 because executor was confident (p̂ = 1.0).

#### Sample Generated Task

```
<question>
A company collects data on customer satisfaction scores (1-10) and their 
purchase history. Analyze the relationship using statistical methods...
</question>
```

#### Output Files

| File | Content |
|------|---------|
| `outputs/curriculum/curriculum_generations_*_final.jsonl` | Full task logs |
| `outputs/curriculum/curriculum_summary_*_final.csv` | Condensed metrics |
| `outputs/curriculum/final_model/` | Trained weights |
| `outputs/curriculum/training_info.json` | Config |

#### View Results

```bash
cat outputs/curriculum/curriculum_summary_*.csv
head -1 outputs/curriculum/curriculum_generations_*.jsonl | jq .
cat outputs/curriculum/curriculum_generations_*.jsonl | jq -r '.task' | head -50
```

#### Validation Checklist

- [x] Model loads (Qwen3-1.7B)
- [x] Thinking mode disabled
- [x] Tasks have `<question>` tags
- [x] Executor samples responses
- [x] Rewards computed (R_unc + R_tool)
- [x] GRPO training completes
- [x] Model saved
- [x] Logs saved

---

### Executor Agent Training (Iteration 1)

**Date:** 2024-11-29 17:42-17:54

#### What Gets Passed to Executor Training

Per the Agent0 paper (Algorithm 1, Lines 11-24):

| Input | Description |
|-------|-------------|
| **Curriculum Model** | `./outputs/curriculum/final_model` (frozen) |
| **Frontier Tasks** | Tasks filtered by self-consistency (p̂ ∈ [0.25, 0.75]) |
| **Pseudo-Labels** | Majority-vote answers from k=2 executor samples |
| **Self-Consistency (p̂)** | Used for ADPO advantage scaling |

#### Command

```bash
uv run train_executor.py \
    --curriculum_model_path ./outputs/curriculum/final_model \
    --max_steps 2 \
    --num_tasks 8 \
    --no_wandb
```

#### Configuration

| Setting | Value |
|---------|-------|
| Executor Model | `Qwen/Qwen3-1.7B` (trainable) |
| Curriculum Model | `./outputs/curriculum/final_model` (frozen) |
| Training Steps | 2 |
| Num Tasks | 8 |
| K Samples | 2 |
| Delta (frontier threshold) | 0.25 |
| Num Generations | 2 |
| Per Device Batch Size | 4 |
| Gradient Accumulation | 2 |
| Enable Thinking | `false` |

#### Training Results

| Metric | Batch 1 | Batch 2 | Notes |
|--------|---------|---------|-------|
| **Avg Reward** | 0.000 | 0.000 | No correct answers |
| **Accuracy** | 0.0% | 0.0% | Executor didn't match pseudo-labels |
| **Loss** | 0.0 | 0.0 | No gradient signal |
| **Completion Length** | 2048 | 2048 | Hit max (clipped) |
| **Clipped Ratio** | 1.0 | 1.0 | All truncated |

#### Timing

| Phase | Duration |
|-------|----------|
| Model Loading (executor) | ~2s |
| Model Loading (curriculum) | ~2s |
| Task Generation (8 tasks) | ~59s |
| Self-Consistency Scoring | ~7m 48s |
| Training Step 1 | ~1m 22s |
| Training Step 2 | ~1m 32s |
| Model Saving | ~3s |
| **Total** | **~11m 55s** |

#### Frontier Task Filtering

| Metric | Value |
|--------|-------|
| Total tasks generated | 8 |
| Tasks after filtering | 8 |
| Retention rate | 100% |
| p̂ range observed | All at ~0.5 |

#### Why Reward = 0?

1. **Completions hit max length** (`clipped_ratio: 1.0`) - truncated at 2048 tokens
2. **No answer extraction** - truncated responses didn't contain `\boxed{...}`
3. **Pseudo-label mismatch** - complex LaTeX answers hard to match

**Fix:** Increase `MAX_COMPLETION_LENGTH` or use simpler tasks.

#### Output Files

| File | Content |
|------|---------|
| `outputs/executor/all_tasks_scored_*.jsonl` | All 8 tasks with p̂ and responses |
| `outputs/executor/frontier_tasks_*.jsonl` | Filtered tasks for training |
| `outputs/executor/task_summary_*.csv` | Condensed view |
| `outputs/executor/executor_generations_*_final.jsonl` | Training outputs |
| `outputs/executor/executor_summary_*_final.csv` | Metrics per sample |
| `outputs/executor/final_model/` | Trained weights |
| `outputs/executor/training_info.json` | Config metadata |

#### View Results

```bash
# Task summary
cat outputs/executor/task_summary_*.csv

# Training summary
cat outputs/executor/executor_summary_*_final.csv

# Frontier tasks (first entry)
head -1 outputs/executor/frontier_tasks_*.jsonl | jq .

# Pseudo-labels
cat outputs/executor/frontier_tasks_*.jsonl | jq -r '.majority_answer' | head -5
```

#### Console Output

<details>
<summary>Click to expand</summary>

```
17:42:32 | INFO     | 🚀 EXECUTOR AGENT TRAINING
17:42:34 | SUCCESS  | ✅ Model loaded: 1,720,574,976 total params
17:42:35 | INFO     | 🔒 Curriculum model frozen
17:43:34 | SUCCESS  | ✅ Generated 8 tasks
17:51:23 | INFO     | Frontier filtering: 8/8 tasks retained (δ=0.25)
17:51:23 | SUCCESS  | ✅ Created executor training dataset with 8 samples
17:52:38 | INFO     | 📊 Batch 1: avg_reward=0.000, accuracy=0.0%
17:54:02 | INFO     | 📊 Batch 2: avg_reward=0.000, accuracy=0.0%
17:54:21 | SUCCESS  | ✅ Executor agent training complete!
```

</details>

#### Validation Checklist

- [x] Curriculum model loads from checkpoint
- [x] Task pool generated (8 tasks)
- [x] Self-consistency computed (k=2)
- [x] Frontier filtering applied (δ=0.25)
- [x] Pseudo-labels extracted
- [x] Training dataset created
- [x] GRPO training completes
- [x] Model saved
- [ ] Non-zero rewards ❌
- [ ] Accuracy > 0 ❌

---

## Co-Evolution: Next Steps

After completing one iteration of both agents, continue the co-evolutionary loop:

```bash
# Iteration 2: Train curriculum with updated executor
uv run train_curriculum.py \
    --executor_model_path ./outputs/executor/final_model \
    --output_dir ./outputs/curriculum_iter2 \
    --max_steps 5 \
    --no_wandb

# Iteration 2: Train executor with updated curriculum
uv run train_executor.py \
    --curriculum_model_path ./outputs/curriculum_iter2/final_model \
    --output_dir ./outputs/executor_iter2 \
    --max_steps 10 \
    --no_wandb
```

Per the Agent0 paper, this co-evolutionary loop drives improvement in both agents over multiple iterations.

---
## Final Notes

1. Other teams' submissions to GitHub/HuggingFace saved at MacBook Pro: `/Users/charles/_Projects/_github/_rl_hack`
2. hackathon finalists info stored in [./hackathon-winners]