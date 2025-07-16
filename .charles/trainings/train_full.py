# /qwen_finetuning_project/train_full.py

import torch
from trl import SFTTrainer, DPOTrainer, SFTConfig, SFTTrainer

# Import our custom modules
from data_prep_unified import load_and_prepare_datasets, create_sft_format
from model import get_model_and_tokenizer
from logger_utils import setup_logger

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "CarperAI/openai_summarize_comparisons"
TRAIN_PERCENT = 0.20  # Use 25% of training data for a faster run
EVAL_PERCENT = 0.15   # Use 50% of validation data

TOP_K = 4  # Number of top layers to fine tune

SFT_OUTPUT_DIR = "./.data/sft_full_results"
DPO_OUTPUT_DIR = "./.data/dpo_full_results"
FINAL_MODEL_PATH = f"./.data/Qwen3_0.6B_tuned_topk{TOP_K})"

def freeze_bottom_layers(model, num_layers_to_tune):
    """
    Freezes the bottom layers of the model, leaving the top layers trainable.
    """
    # Qwen2's layers are in `model.model.layers`
    all_layers = model.model.layers
    num_total_layers = len(all_layers)
    layers_to_freeze = num_total_layers - num_layers_to_tune

    logger = setup_logger("layer_freezing")
    logger.info(f"Total layers: {num_total_layers}")
    logger.info(f"Freezing the bottom {layers_to_freeze} layers.")

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the parameters in the top N layers
    for i in range(layers_to_freeze, num_total_layers):
        layer = all_layers[i]
        for param in layer.parameters():
            param.requires_grad = True
            
    # Also leave the normalization layer and the output head trainable
    for param in model.model.norm.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Log the trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.success(
        f"Trainable params: {trainable_params:,} || "
        f"Total params: {total_params:,} || "
        f"Trainable %: {100 * trainable_params / total_params:.2f}"
    )
    return model


if __name__ == "__main__":
    logger = setup_logger("full_training_pipeline")

    # --- 1. Data Preparation ---
    logger.info("--- Preparing Datasets ---")
    dpo_train, dpo_eval, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT
    )
    sft_train = dpo_train.rename_column("chosen", "completion")
    sft_eval = dpo_eval.rename_column("chosen", "completion")
    sft_train = sft_train.remove_columns("rejected")
    sft_eval = sft_eval.remove_columns("rejected")
    logger.success("Datasets prepared for SFT and DPO.\n")

    # --- 2. Stage 1: Supervised Fine-Tuning (SFT) ---
    logger.info("--- STAGE 1: SUPERVISED FINE-TUNING (SFT) ---")
    
    sft_model, tokenizer = get_model_and_tokenizer(MODEL_ID)
    if TOP_K > 0:
        sft_model = freeze_bottom_layers(sft_model, TOP_K)

    training_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        max_seq_length=1024,
        completion_only_loss=True, 
    )

    sft_trainer = SFTTrainer(
        model=sft_model,
        args=training_args,
        train_dataset=sft_train,
        eval_dataset=sft_eval,
        processing_class=tokenizer,
    )

    sft_trainer.train()
    logger.success(f"SFT finished. Model saved to {SFT_OUTPUT_DIR}")

    # Save the final SFT model and tokenizer
    sft_trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)

    # Clean up to free VRAM
    del sft_model
    del sft_trainer
    torch.cuda.empty_cache()
    logger.info("Cleaned up SFT resources.")

    # --- 3. Stage 2: Direct Preference Optimization (DPO) ---
    logger.info("--- STAGE 2: DIRECT PREFERENCE OPTIMIZATION (DPO) ---")

    # Load the SFT-tuned model from disk
    dpo_model, tokenizer = get_model_and_tokenizer(SFT_OUTPUT_DIR)

    dpo_training_args = TrainingArguments(
        output_dir=DPO_OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1, # DPO is more memory-intensive
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6, # Use a lower learning rate for preference tuning
        logging_steps=50,
        save_steps=200,
        do_eval=True,
        eval_steps=200,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=dpo_model,
        ref_model=None, # TRL will handle creating the reference model copy
        args=dpo_training_args,
        train_dataset=dpo_train,
        eval_dataset=dpo_eval,
        tokenizer=tokenizer,
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
        # IMPORTANT: No peft_config is passed, so it performs full fine-tuning
    )

    dpo_trainer.train()
    logger.success(f"DPO finished. Final model saved to {DPO_OUTPUT_DIR}")
    
    # Save the final DPO model
    dpo_trainer.save_model(FINAL_MODEL_PATH)
    tokenizer.save_pretrained(FINAL_MODEL_PATH)
    logger.success(f"Fully tuned final model saved to {FINAL_MODEL_PATH}")