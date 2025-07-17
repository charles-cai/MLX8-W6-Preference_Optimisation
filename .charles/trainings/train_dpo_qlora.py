# /qwen_finetuning_project/train_lora.py

import torch
import os
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer
from trl.trainer.dpo_config import DPOConfig

# Import our custom modules
from data_prep_unified import load_and_prepare_datasets
from model import get_model_and_tokenizer # We use this to get the tokenizer
from logger_utils import setup_logger

# We need the base model class for loading with quantization
from transformers import AutoModelForCausalLM

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B-Base")
DATASET_NAME = os.environ.get("DATASET_NAME", "CarperAI/openai_summarize_comparisons")
TRAIN_PERCENT = float(os.environ.get("TRAIN_PERCENT", "0.20"))
EVAL_PERCENT = float(os.environ.get("EVAL_PERCENT", "0.15"))

DPO_ADAPTER_DIR = os.environ.get("DPO_ADAPTER_DIR", "./.data/dpo_qlora_adapter")
DPO_MERGED_DIR = os.environ.get("DPO_MERGED_DIR", "./.data/dpo_qlora_merged")

if __name__ == "__main__":
    logger = setup_logger("dpo_qlora_training_pipeline")

    # --- 1. Data Preparation (Identical to full tuning) ---
    logger.info("--- Preparing Datasets ---")
    dpo_train, dpo_eval, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT
    )
    # Remove unused SFT dataset creation
    # sft_train = dpo_train.map(create_sft_format)
    # sft_eval = dpo_eval.map(create_sft_format)
    logger.success("Datasets prepared for DPO.")

    # --- 2. Define QLoRA Configuration ---
    logger.info("--- Defining QLoRA Configuration ---")

    # QLoRA quantizes the base model to 4-bits
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config specifies which layers to adapt
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ---3. Load the base model with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Get the tokenizer
    _, tokenizer = get_model_and_tokenizer(MODEL_ID)

    # --- 4. Stage 2: Direct Preference Optimization (DPO) with QLoRA ---
    logger.info("--- STAGE 2: DIRECT PREFERENCE OPTIMIZATION (DPO) with QLoRA ---")
    
    dpo_training_args = DPOConfig(
        output_dir=DPO_ADAPTER_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2, # DPO is more memory intensive
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5, # Lower LR for preference tuning
        logging_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        max_prompt_length=1024,
        max_completion_length=512,
    )


    # The DPOTrainer needs the base model path and the adapter path
    dpo_trainer = DPOTrainer(
        model=MODEL_ID,
        args=dpo_training_args,
        train_dataset=dpo_train,
        eval_dataset=dpo_eval,
        processing_class=tokenizer,
        peft_config=peft_config,    
    )

    dpo_trainer.train()
    logger.success(f"DPO with QLoRA finished. Final adapter saving to {DPO_ADAPTER_DIR}")
    dpo_trainer.save_model(DPO_ADAPTER_DIR)

    # --- 6. Stage 3: Merge Adapter and Save Final Model ---
    logger.info("--- STAGE 3: MERGING THE FINAL ADAPTER ---")
    
    # Reload the base model in full precision (e.g., bfloat16) to merge
    merged_model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_DIR) 
    merged_model.save_pretrained(DPO_MERGED_DIR)
    tokenizer.save_pretrained(DPO_MERGED_DIR)
    
    logger.success(f"Successfully merged adapter. Final model saved to {DPO_MERGED_DIR}")