import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import Dataset
import wandb
from tqdm import tqdm

# Import our custom modules
from data_prep_unified import load_and_prepare_datasets
from logger_utils import setup_logger

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B-Base")
DATASET_NAME_COMPARISON = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")
TRAIN_PERCENT = float(os.environ.get("TRAIN_PERCENT", "0.20"))
EVAL_PERCENT = float(os.environ.get("EVAL_PERCENT", "0.15"))

# Reward model specific percentages
REWARD_TRAIN_PERCENT = float(os.environ.get("REWARD_TRAIN_PERCENT", "0.20"))
REWARD_EVAL_PERCENT = float(os.environ.get("REWARD_EVAL_PERCENT", "0.15"))

# Use the SFT_TOP_0 model as base for reward model
SFT_OUTPUT_DIR_PREFIX = os.environ.get("SFT_OUTPUT_DIR_PREFIX", "./.data/sft_full_top_")
BASE_MODEL_PATH = f"{SFT_OUTPUT_DIR_PREFIX}0"  # SFT_TOP_0 model
REWARD_MODEL_DIR = os.environ.get("REWARD_MODEL_DIR", "./.data/reward_model")

def format_reward_dataset(dataset, tokenizer, max_length=512):
    """
    Format dataset for TRL RewardTrainer with proper tokenization.
    TRL RewardTrainer expects datasets with 'input_ids_chosen', 'attention_mask_chosen', 
    'input_ids_rejected', 'attention_mask_rejected' columns.
    """
    def format_example(example):
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Create full text for chosen and rejected responses
        chosen_text = prompt + "\n\nSummary:\n" + chosen
        rejected_text = prompt + "\n\nSummary:\n" + rejected
        
        # Tokenize with consistent length
        chosen_tokens = tokenizer(
            chosen_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_tokens = tokenizer(
            rejected_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids_chosen': chosen_tokens["input_ids"].squeeze(),
            'attention_mask_chosen': chosen_tokens["attention_mask"].squeeze(),
            'input_ids_rejected': rejected_tokens["input_ids"].squeeze(),
            'attention_mask_rejected': rejected_tokens["attention_mask"].squeeze(),
        }
    
    logger = setup_logger("reward_data_formatting")
    logger.info(f"Formatting {len(dataset)} examples for reward training...")
    
    # Use tqdm for progress tracking
    formatted_dataset = []
    for example in tqdm(dataset, desc="Formatting reward dataset"):
        formatted_example = format_example(example)
        formatted_dataset.append(formatted_example)
    
    # Convert to HuggingFace dataset
    return Dataset.from_list(formatted_dataset)

def compute_reward_metrics(eval_pred):
    """
    Custom metrics function for reward model evaluation.
    TRL RewardTrainer expects this to handle reward predictions correctly.
    """
    predictions, labels = eval_pred
    
    # For reward models, we don't need traditional accuracy metrics
    # Instead, we can compute reward-specific metrics
    if predictions is not None:
        # Handle case where predictions might be scalars or arrays
        if hasattr(predictions, 'shape') and len(predictions.shape) > 0:
            mean_reward = float(predictions.mean())
            std_reward = float(predictions.std())
        else:
            mean_reward = float(predictions) if predictions is not None else 0.0
            std_reward = 0.0
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }
    else:
        return {"mean_reward": 0.0, "std_reward": 0.0}

if __name__ == "__main__":
    logger = setup_logger("reward_model_training_trl")
    
    # Initialize wandb
    model_id = os.environ.get("MODEL_ID", "unknown").split("/")[-1]
    wandb_run_name = f"reward_{model_id}_train_{TRAIN_PERCENT}_eval_{EVAL_PERCENT}_epochs_{os.environ.get('NUM_TRAIN_EPOCHS', '1')}"
    
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlx8-week6-preference-optimisation"),
        entity=os.environ.get("WANDB_ENTITY", "charles-cai"),
        name=wandb_run_name,
        config={
            "model_type": "reward_model",
            "base_model_path": BASE_MODEL_PATH,
            "dataset_name": DATASET_NAME_COMPARISON,
            "train_percent": TRAIN_PERCENT,
            "eval_percent": EVAL_PERCENT,
            "num_train_epochs": int(os.environ.get("NUM_TRAIN_EPOCHS", "1")),
            "per_device_train_batch_size": int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2")),
            "per_device_eval_batch_size": int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2")),
            "gradient_accumulation_steps": int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")),
            "learning_rate": float(os.environ.get("LEARNING_RATE", "2e-5")),
            "max_length": int(os.environ.get("MAX_SEQ_LENGTH", "512")),
            "base_model": os.environ.get("MODEL_ID", "unknown"),
        }
    )
    
    # --- 1. Data Preparation ---
    logger.info("--- Preparing Datasets for Reward Model ---")
    
    # Load datasets with reward model training configuration
    # We use the next 20% of training data for reward model training
    dpo_train, dpo_eval, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME_COMPARISON,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT,
        train_percent_reward=REWARD_TRAIN_PERCENT,  # Use dedicated reward model train percent
        eval_precent_reward=REWARD_EVAL_PERCENT,    # Use dedicated reward model eval percent
    )
    
    logger.success(f"Reward model datasets prepared:")
    logger.success(f"Train records: {len(dpo_train)}")
    logger.success(f"Eval records: {len(dpo_eval)}")
    
    # Log dataset info to wandb
    wandb.log({
        "train_samples": len(dpo_train),
        "eval_samples": len(dpo_eval),
    })
    
    # --- 2. Load Base Model and Tokenizer ---
    logger.info(f"Loading base model from: {BASE_MODEL_PATH}")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 3. Format datasets for TRL RewardTrainer ---
    logger.info("Formatting datasets for TRL RewardTrainer...")
    max_length = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
    train_dataset = format_reward_dataset(dpo_train, tokenizer, max_length)
    eval_dataset = format_reward_dataset(dpo_eval, tokenizer, max_length)
    
    # --- 4. Configure Reward Training ---
    logger.info("Setting up reward training configuration...")
    
    reward_config = RewardConfig(
        output_dir=REWARD_MODEL_DIR,
        run_name=f"reward_model_{model_id}_{TRAIN_PERCENT}_{EVAL_PERCENT}",
        num_train_epochs=int(os.environ.get("NUM_TRAIN_EPOCHS", "1")),
        per_device_train_batch_size=int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2")),
        per_device_eval_batch_size=int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2")),
        gradient_accumulation_steps=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "2e-5")),
        logging_strategy="steps",
        logging_steps=int(os.environ.get("LOGGING_STEPS", "100")),
        save_strategy="steps",
        save_steps=int(os.environ.get("SAVE_STEPS", "500")),
        eval_steps=int(os.environ.get("EVAL_STEPS", "500")),
        save_total_limit=int(os.environ.get("SAVE_TOTAL_LIMIT", "1")),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=max_length,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )
    
    # --- 5. Create RewardTrainer ---
    logger.info("Creating TRL RewardTrainer...")
    
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_reward_metrics,  # Add custom metrics function
    )
    
    # Disable visualization to prevent the rounding error
    trainer.visualize_samples = lambda num_print_samples: None
    
    # --- 6. Train the Reward Model ---
    logger.info("--- Starting Reward Model Training with TRL ---")
    
    # Remove tqdm wrapper that's conflicting with trainer progress
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Try to save whatever progress we have
        if hasattr(trainer, 'model'):
            logger.info("Attempting to save model state...")
            trainer.save_model(REWARD_MODEL_DIR)
            tokenizer.save_pretrained(REWARD_MODEL_DIR)
        raise e
    
    # --- 7. Save the Reward Model ---
    logger.info(f"Saving reward model to {REWARD_MODEL_DIR}")
    trainer.save_model(REWARD_MODEL_DIR)
    tokenizer.save_pretrained(REWARD_MODEL_DIR)
    
    logger.success(f"Reward model training completed and saved to {REWARD_MODEL_DIR}")
    
    # --- 8. Quick Evaluation ---
    logger.info("--- Quick Reward Model Evaluation ---")
    
    # Test on a few examples
    test_examples = eval_dataset.select(range(min(5, len(eval_dataset))))
    
    correct_preferences = 0
    total_examples = 0
    eval_results = []
    
    for i, example in enumerate(tqdm(test_examples, desc="Evaluating examples")):
        # Get tokenized inputs
        chosen_ids = example['input_ids_chosen']
        chosen_mask = example['attention_mask_chosen']
        rejected_ids = example['input_ids_rejected']
        rejected_mask = example['attention_mask_rejected']
        
        # Convert to tensors if they're not already
        if not isinstance(chosen_ids, torch.Tensor):
            chosen_ids = torch.tensor(chosen_ids)
        if not isinstance(chosen_mask, torch.Tensor):
            chosen_mask = torch.tensor(chosen_mask)
        if not isinstance(rejected_ids, torch.Tensor):
            rejected_ids = torch.tensor(rejected_ids)
        if not isinstance(rejected_mask, torch.Tensor):
            rejected_mask = torch.tensor(rejected_mask)
        
        # Ensure tensors are on correct device and have batch dimension
        chosen_ids = chosen_ids.unsqueeze(0).to(model.device)
        chosen_mask = chosen_mask.unsqueeze(0).to(model.device)
        rejected_ids = rejected_ids.unsqueeze(0).to(model.device)
        rejected_mask = rejected_mask.unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            model.eval()
            chosen_outputs = model(input_ids=chosen_ids, attention_mask=chosen_mask)
            rejected_outputs = model(input_ids=rejected_ids, attention_mask=rejected_mask)
            
            # Get rewards (assuming the model outputs rewards as logits)
            chosen_reward = chosen_outputs.logits.mean()
            rejected_reward = rejected_outputs.logits.mean()
        
        is_correct = chosen_reward.item() > rejected_reward.item()
        if is_correct:
            correct_preferences += 1
        total_examples += 1
        
        eval_result = {
            "example_id": i,
            "chosen_reward": chosen_reward.item(),
            "rejected_reward": rejected_reward.item(),
            "reward_difference": (chosen_reward - rejected_reward).item(),
            "correct_preference": is_correct
        }
        eval_results.append(eval_result)
        
        # Decode text for logging
        chosen_text = tokenizer.decode(chosen_ids.squeeze(), skip_special_tokens=True)
        rejected_text = tokenizer.decode(rejected_ids.squeeze(), skip_special_tokens=True)
        
        logger.info(f"Example {i+1}:")
        logger.info(f"Chosen text: {chosen_text[:100]}...")
        logger.info(f"Rejected text: {rejected_text[:100]}...")
        logger.info(f"Chosen reward: {chosen_reward.item():.4f}")
        logger.info(f"Rejected reward: {rejected_reward.item():.4f}")
        logger.info(f"Difference: {(chosen_reward - rejected_reward).item():.4f}")
        logger.info(f"Correct preference: {is_correct}")
        logger.info("---")
    
    # Log evaluation results to wandb
    accuracy = correct_preferences / total_examples if total_examples > 0 else 0
    avg_reward_diff = sum([r["reward_difference"] for r in eval_results]) / len(eval_results)
    
    wandb.log({
        "eval_accuracy": accuracy,
        "eval_correct_preferences": correct_preferences,
        "eval_total_examples": total_examples,
        "eval_avg_reward_difference": avg_reward_diff,
    })
    
    # Log detailed evaluation results as a table
    eval_table = wandb.Table(columns=["example_id", "chosen_reward", "rejected_reward", "reward_difference", "correct_preference"])
    for result in eval_results:
        eval_table.add_data(
            result["example_id"],
            result["chosen_reward"],
            result["rejected_reward"],
            result["reward_difference"],
            result["correct_preference"]
        )
    wandb.log({"eval_results": eval_table})
    
    # --- 9. Save additional model info for later use ---
    model_info = {
        'base_model_path': BASE_MODEL_PATH,
        'training_config': reward_config.to_dict(),
        'model_type': 'reward_model_trl',
        'dataset_name': DATASET_NAME_COMPARISON,
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
    }
    
    torch.save(model_info, os.path.join(REWARD_MODEL_DIR, 'training_info.pt'))
    
    logger.success("Reward model training pipeline completed successfully!")
    
    # Finish wandb run
    wandb.finish()
