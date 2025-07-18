import random
import os
from datasets import load_dataset, concatenate_datasets
from logger_utils import setup_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
DATASET_NAME_COMPARISON = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")

Prompts = [
    "Summarize the following post:", 
    "Summarize the following post in a few sentences:",
    "Please provide a concise summary of the content below:",
    "Can you give me a brief overview of the text that follows?",
    "Write a short recap of the main points in this post:",
    "Condense the key ideas from the following passage into a few sentences:",
    "Create a quick synopsis of the material below:",
    "Highlight the essential takeaways from this post:",
    "Craft a succinct digest of the following content:",
    "Explain the core message of this post in summary form:",
    "Summarize the central themes of the text provided:"
]

def format_number(num):
    # This is your useful utility function from before
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}m"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}k"
    else:
        return str(num)

def load_and_prepare_datasets(
    dataset_name: str, 
    train_on_percent: float = 1.0, 
    eval_on_percent: float = 1.0,
    train_percent_reward: float = 0.0,
    eval_percent_reward: float = 0.0,
    ):
    
    """
    Loads and prepares the comparison dataset for both SFT and DPO.
    
    Args:
        dataset_name (str): The name of the dataset on Hugging Face hub.
        train_on_percent (float): The percentage of the training set to use (0.0 to 1.0).
        eval_on_percent (float): The percentage of the validation set to use (0.0 to 1.0).
        train_percent_reward (float): Additional percentage for reward model training.
        eval_precent_reward (float): Additional percentage for reward model evaluation.
    """
    # --- 1. Load all necessary splits ---
    train_dataset = load_dataset(dataset_name, split="train")
    valid1_dataset = load_dataset(dataset_name, split="valid1")
    valid2_dataset = load_dataset(dataset_name, split="valid2")
    test_dataset = load_dataset(dataset_name, split="test")

    # --- 2. Combine validation splits for a more robust evaluation set ---
    validation_dataset = concatenate_datasets([valid1_dataset, valid2_dataset])

    # --- 3. Handle reward model data splits ---
    if train_percent_reward > 0.0:
        # For reward model training, use the next portion of data
        total_train_samples = len(train_dataset)
        
        # Calculate indices for different splits
        sft_end_idx = int(total_train_samples * train_on_percent)
        reward_end_idx = int(total_train_samples * (train_on_percent + train_percent_reward))
        
        # Select reward model training data (next 20% after SFT data)
        train_dataset = train_dataset.select(range(sft_end_idx, min(reward_end_idx, total_train_samples)))
        
        # Similar for validation
        total_eval_samples = len(validation_dataset)
        eval_sft_end_idx = int(total_eval_samples * eval_on_percent)
        eval_reward_end_idx = int(total_eval_samples * (eval_on_percent + eval_percent_reward))
        
        validation_dataset = validation_dataset.select(range(eval_sft_end_idx, min(eval_reward_end_idx, total_eval_samples)))
    else:
        # --- 3. Apply percentage-based subsetting for regular training ---
        train_on_percent_new = train_on_percent + train_percent_reward if train_percent_reward > 0.0 else train_on_percent    
        eval_on_percent_new = eval_on_percent + eval_percent_reward if eval_percent_reward > 0.0 else eval_on_percent

        if train_on_percent_new < 1.0:
            num_train_samples = int(len(train_dataset) * train_on_percent)
            train_dataset = train_dataset.select(range(num_train_samples))

        if eval_on_percent_new < 1.0:
            num_eval_samples = int(len(validation_dataset) * eval_on_percent)
            validation_dataset = validation_dataset.select(range(num_eval_samples))
    
    # --- 4. Define th#e formatting function ---
    # Standard format for PPO / DPO: ["prompt", "chosen", "rejected"]
    def format_data(example):
        # Add a random instruction to the original prompt for variety
        prompt = random.choice(Prompts) + "\n\n" + example['prompt']
        chosen_summary = example['chosen']
        rejected_summary = example['rejected']
        
        return {
            "prompt": prompt,
            "chosen": chosen_summary,
            "rejected": rejected_summary,
        }

    # --- 5. Apply formatting to all splits ---
    train_dataset = train_dataset.map(format_data, remove_columns=train_dataset.column_names)
    validation_dataset = validation_dataset.map(format_data, remove_columns=validation_dataset.column_names)
    # We can also format the test set if we plan to use it for final evaluation
    test_dataset = test_dataset.map(format_data, remove_columns=test_dataset.column_names)

    return train_dataset, validation_dataset, test_dataset

def create_sft_format(example):
    """
    Formats an example from the DPO dataset for use with SFTTrainer.
    The SFTTrainer expects a single 'text' column.
    """
    example["text"] = example["prompt"] + "\n\nSummary:\n" + example["chosen"]
    return example

def create_reward_model_format(example, tokenizer, max_length=512):
    """
    Formats an example for reward model training with proper tokenization.
    Ensures chosen and rejected sequences have the same length.
    """
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    # Tokenize prompt + chosen
    chosen_text = prompt + "\n\nSummary:\n" + chosen
    chosen_tokens = tokenizer(
        chosen_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Tokenize prompt + rejected
    rejected_text = prompt + "\n\nSummary:\n" + rejected
    rejected_tokens = tokenizer(
        rejected_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "input_ids_chosen": chosen_tokens["input_ids"].squeeze(),
        "attention_mask_chosen": chosen_tokens["attention_mask"].squeeze(),
        "input_ids_rejected": rejected_tokens["input_ids"].squeeze(),
        "attention_mask_rejected": rejected_tokens["attention_mask"].squeeze(),
    }


# --- Main execution block to demonstrate usage ---
if __name__ == "__main__":
    logger = setup_logger("dataset_preparation")
    
    # --- SCENARIO: Use environment variables for percentages ---
    TRAIN_PERCENT = float(os.environ.get("TRAIN_PERCENT", "0.20"))
    EVAL_PERCENT = float(os.environ.get("EVAL_PERCENT", "0.15"))
    TRAIN_PERCENT_REWARD = 0.0
    EVAL_PERCENT_REWARD = 0.0

    logger.info(f"Loading {DATASET_NAME_COMPARISON} with {TRAIN_PERCENT*100}% of train and {EVAL_PERCENT*100}% of validation data.")
    logger.info(f"Train percent for reward model: {TRAIN_PERCENT_REWARD*100}%, Eval percent for reward model: {EVAL_PERCENT_REWARD*100}%")

    # Load the base datasets formatted for DPO
    dpo_train_dataset, dpo_validation_dataset, dpo_test_dataset = load_and_prepare_datasets(
        dataset_name=DATASET_NAME_COMPARISON,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT,
        train_percent_reward=TRAIN_PERCENT_REWARD,
        eval_precent_reward=EVAL_PERCENT_REWARD,
    )
    
    logger.success("DPO/RM datasets prepared:\n" +
                   f"Train records: {format_number(len(dpo_train_dataset))}\n" +
                   f"Validation records: {format_number(len(dpo_validation_dataset))}\n" +
                   f"Test records: {format_number(len(dpo_test_dataset))}")

    # Now, create the SFT-formatted versions from the DPO datasets
    sft_train_dataset = dpo_train_dataset.map(create_sft_format)
    sft_validation_dataset = dpo_validation_dataset.map(create_sft_format)

    logger.success("SFT datasets created from the DPO data:\n" +
                   f"Train records: {format_number(len(sft_train_dataset))}\n" +
                   f"Validation records: {format_number(len(sft_validation_dataset))}")
    
    # You can now inspect an example
    logger.success("--- DPO Example ---")
    logger.info(f"Prompt: {dpo_train_dataset[0]['prompt']}\n")
    logger.success(f"Chosen Summary: {dpo_train_dataset[0]['chosen']}\n")
    logger.warning(f"Rejected Summary: {dpo_train_dataset[0]['rejected']}")

    # print(dpo_train_dataset[0])
    
    logger.success("\n--- SFT Example (from the same base data) ---")
    #print(sft_train_dataset[0])
    logger.info(f"prompt: {sft_train_dataset[0]['prompt']}")
    logger.success(f"Chosen Summary: {sft_train_dataset[0]['chosen']}\n")
    logger.warning(f"Rejected Summary (no need): {sft_train_dataset[0]['rejected']}")