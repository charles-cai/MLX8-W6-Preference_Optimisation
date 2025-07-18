# MLX8 Week 6 Preference Optimisation - AI Coding Instructions

## Project Overview
This is a comprehensive RLHF (Reinforcement Learning from Human Feedback) pipeline implementing the full preference optimization workflow: SFT → Reward Model → DPO/PPO training using TRL (Transformers Reinforcement Learning).

## Architecture & Data Flow
1. **Data Pipeline**: `data_prep_unified.py` handles dataset loading with configurable train/eval splits for different training phases
2. **Training Stages**: Sequential pipeline in `.charles/trainings/`:
   - `train_sft.py` - Supervised Fine-Tuning on base models
   - `train_reward.py` - Reward model training using comparison data
   - `train_dpo.py` - Direct Preference Optimization using trained reward model
3. **Model Progression**: Base model → SFT model → Reward model → DPO-optimized model

## Key Patterns & Conventions

### Environment Configuration
- All training scripts use `.env` files for configuration
- Model paths follow pattern: `./.data/{sft_full_top_0,reward_model,dpo_model}`
- Wandb integration is standard across all training scripts

### Dataset Handling
```python
# Standard pattern for dataset preparation
dpo_train, dpo_eval, _ = load_and_prepare_datasets(
    dataset_name=DATASET_NAME_COMPARISON,
    train_on_percent=TRAIN_PERCENT,
    eval_on_percent=EVAL_PERCENT,
    train_percent_reward=REWARD_TRAIN_PERCENT,
    eval_precent_reward=REWARD_EVAL_PERCENT,
)
```

### TRL Integration Patterns
- **Reward Training**: Use `RewardTrainer` with custom `compute_reward_metrics()` to avoid TRL's default accuracy computation
- **Data Formatting**: Reward models expect `{input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected}` format
- **Model Loading**: Always check for tensor types and convert properly: `torch.tensor(data)` before `.unsqueeze(0).to(device)`

### Error Handling & Logging
- Use `logger_utils.setup_logger()` for consistent logging across modules
- Always wrap training in try/catch with model saving on failure
- Disable TRL's visualization to prevent tensor shape conflicts: `trainer.visualize_samples = lambda num_print_samples: None`

## Critical Development Workflows

### Training Pipeline Execution
```bash
# Sequential execution required:
cd .charles/trainings/
uv run train_sft.py      # Creates SFT model, default is TOP_K=5, you can --top_k 0 to overwrite, 0 for full model training
uv run train_dpo.py      # Uses reward model, creates final DPO model

uv run eval.py        # Evaluates DPO model, uses reward model for scoring
uv run eval.py --sft --top_k=0
uv run eval.py --dpo

uv run train_reward.py   # Uses SFT model, creates reward model  
uv run eval_reward.py

uv run train_ppo.py
uv run eval.py --ppo #(WIP)
```

### Common TRL Debugging
- **Tensor Issues**: Always convert list data to tensors before `.unsqueeze()`
- **Metrics Errors**: TRL's default metrics expect iterable predictions; use custom `compute_metrics` functions
- **Memory Issues**: Use `torch.no_grad()` for evaluation loops and proper device management

### Model Path Dependencies
- Reward training depends on: `{SFT_OUTPUT_DIR_PREFIX}0` (SFT_TOP_0 model)
- DPO training depends on: `REWARD_MODEL_DIR` output
- Update environment variables when changing model locations

## Integration Points
- **Wandb**: All training logs to shared project with run naming: `{stage}_{model_id}_{train_percent}_{eval_percent}_{epochs}`
- **HuggingFace**: Models saved in HF format for easy loading/sharing
- **TRL Library**: Core training framework - understanding TRL's expected data formats is crucial

## Testing & Validation
- Each training script includes post-training evaluation section
- Reward model validation checks preference accuracy on held-out examples
- Model info saved as `training_info.pt` for pipeline traceability