# /qwen_finetuning_project/train_full.py

import torch
import argparse
import os
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer

# Import our custom modules
from data_prep_unified import load_and_prepare_datasets, create_sft_format
from model import get_model_and_tokenizer
from logger_utils import setup_logger

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B-Base")
DATASET_NAME_COMPARISON = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")
TRAIN_PERCENT = float(os.environ.get("TRAIN_PERCENT", "0.20"))
EVAL_PERCENT = float(os.environ.get("EVAL_PERCENT", "0.15"))

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for Qwen model")
    parser.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K", "4")), 
                        help="Number of top layers to fine-tune (default from .env or 0 for full model training)")
    args = parser.parse_args()
    
    logger = setup_logger("full_training_pipeline")

    TOP_K = args.top_k  # Command line argument overwrites .env configuration

    SFT_OUTPUT_DIR = f"{os.environ.get('SFT_OUTPUT_DIR_PREFIX', './.data/sft_full_top_')}{TOP_K}"
    DPO_OUTPUT_DIR = os.environ.get("DPO_OUTPUT_DIR", "./.data/dpo_full_results")

    # --- 1. Data Preparation ---
    logger.info("--- Preparing Datasets ---")
    dpo_train, dpo_eval, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME_COMPARISON,
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
        run_name=f"sft_top_{TOP_K}_{MODEL_ID.split('/')[-1]}",
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
        max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "512")),
        completion_only_loss=True,
        dataloader_drop_last=True,  # Ensure consistent batch sizes
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

    # # Clean up to free VRAM
    # del sft_model
    # del sft_trainer
    # torch.cuda.empty_cache()
    # logger.info("Cleaned up SFT resources.")