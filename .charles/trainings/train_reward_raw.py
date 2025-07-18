import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np

# Import our custom modules
from data_prep_unified import load_and_prepare_datasets
from logger_utils import setup_logger

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B-Base")
DATASET_NAME = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")
TRAIN_PERCENT = float(os.environ.get("TRAIN_PERCENT", "0.20"))
EVAL_PERCENT = float(os.environ.get("EVAL_PERCENT", "0.15"))

# Use the SFT_TOP_0 model as base for reward model
SFT_OUTPUT_DIR_PREFIX = os.environ.get("SFT_OUTPUT_DIR_PREFIX", "./.data/sft_full_top_")
BASE_MODEL_PATH = f"{SFT_OUTPUT_DIR_PREFIX}0"  # SFT_TOP_0 model
REWARD_MODEL_DIR = os.environ.get("REWARD_MODEL_DIR", "./.data/reward_model")

class RewardModel(nn.Module):
    """
    A reward model that adds a scalar reward head to a pretrained language model.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get the base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state
        
        # Get the reward score from the last token
        # We use the last non-padding token for each sequence
        batch_size = last_hidden_state.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        
        # Gather the last hidden states
        last_hidden_states = last_hidden_state[torch.arange(batch_size), sequence_lengths]
        
        # Get reward scores
        rewards = self.reward_head(last_hidden_states)
        
        return rewards

class RewardDataset(Dataset):
    """
    Dataset for reward model training using preference pairs.
    """
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Create full text for chosen and rejected
        chosen_text = prompt + "\n\nSummary:\n" + chosen
        rejected_text = prompt + "\n\nSummary:\n" + rejected
        
        # Tokenize both
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(),
        }

class RewardTrainer(Trainer):
    """
    Custom trainer for reward model training.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get rewards for chosen and rejected responses
        chosen_rewards = model(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask']
        )
        
        rejected_rewards = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask']
        )
        
        # Compute preference loss (chosen should have higher reward than rejected)
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        if return_outputs:
            return loss, {
                'chosen_rewards': chosen_rewards,
                'rejected_rewards': rejected_rewards,
                'reward_difference': chosen_rewards - rejected_rewards
            }
        
        return loss

if __name__ == "__main__":
    logger = setup_logger("reward_model_training")
    
    # --- 1. Data Preparation ---
    logger.info("--- Preparing Datasets for Reward Model ---")
    
    # Load datasets with reward model training configuration
    # We use the next 20% of training data for reward model training
    dpo_train, dpo_eval, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT,
        train_percent_reward=TRAIN_PERCENT,  # Use next 20% for reward model
        eval_precent_reward=EVAL_PERCENT,    # Use next 15% for reward model eval
    )
    
    logger.success(f"Reward model datasets prepared:")
    logger.success(f"Train records: {len(dpo_train)}")
    logger.success(f"Eval records: {len(dpo_eval)}")
    
    # --- 2. Load Base Model and Tokenizer ---
    logger.info(f"Loading base model from: {BASE_MODEL_PATH}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 3. Create Reward Model ---
    logger.info("Creating reward model...")
    reward_model = RewardModel(base_model)
    
    # --- 4. Prepare Datasets ---
    logger.info("Preparing reward datasets...")
    train_dataset = RewardDataset(dpo_train, tokenizer, max_length=512)
    eval_dataset = RewardDataset(dpo_eval, tokenizer, max_length=512)
    
    # --- 5. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=REWARD_MODEL_DIR,
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
        report_to="none",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # --- 6. Create Trainer ---
    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # --- 7. Train the Reward Model ---
    logger.info("--- Starting Reward Model Training ---")
    trainer.train()
    
    # --- 8. Save the Reward Model ---
    logger.info(f"Saving reward model to {REWARD_MODEL_DIR}")
    trainer.save_model(REWARD_MODEL_DIR)
    tokenizer.save_pretrained(REWARD_MODEL_DIR)
    
    # Save the custom reward model class info
    torch.save({
        'model_class': 'RewardModel',
        'base_model_path': BASE_MODEL_PATH,
        'model_config': base_model.config,
    }, os.path.join(REWARD_MODEL_DIR, 'reward_model_info.pt'))
    
    logger.success(f"Reward model training completed and saved to {REWARD_MODEL_DIR}")
    
    # --- 9. Quick Evaluation ---
    logger.info("--- Quick Reward Model Evaluation ---")
    reward_model.eval()
    
    # Test on a few examples
    test_examples = dpo_eval.select(range(min(5, len(dpo_eval))))
    
    for i, example in enumerate(test_examples):
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Tokenize and get rewards
        chosen_text = prompt + "\n\nSummary:\n" + chosen
        rejected_text = prompt + "\n\nSummary:\n" + rejected
        
        chosen_tokens = tokenizer(chosen_text, return_tensors='pt', truncation=True, max_length=512)
        rejected_tokens = tokenizer(rejected_text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            chosen_reward = reward_model(chosen_tokens['input_ids'], chosen_tokens['attention_mask'])
            rejected_reward = reward_model(rejected_tokens['input_ids'], rejected_tokens['attention_mask'])
        
        logger.info(f"Example {i+1}:")
        logger.info(f"Chosen reward: {chosen_reward.item():.4f}")
        logger.info(f"Rejected reward: {rejected_reward.item():.4f}")
        logger.info(f"Difference: {(chosen_reward - rejected_reward).item():.4f}")
        logger.info(f"Correct preference: {chosen_reward.item() > rejected_reward.item()}")
        logger.info("---")
