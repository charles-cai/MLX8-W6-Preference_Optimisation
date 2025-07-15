# /qwen_finetuning_project/train_lora.py

import torch
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DPOTrainer

# Import our custom modules
from data_prep_unified import load_and_prepare_datasets, create_sft_format
from model import get_model_and_tokenizer # We use this to get the tokenizer
from logger_utils import setup_logger

# We need the base model class for loading with quantization
from transformers import AutoModelForCausalLM

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2-0.5B-Base"
DATASET_NAME = "CarperAI/openai_summarize_comparisons"
TRAIN_PERCENT = 0.25  # Use 25% of training data for a faster run
EVAL_PERCENT = 0.50   # Use 50% of validation data

SFT_ADAPTER_DIR = "./sft_qlora_adapter"
DPO_ADAPTER_DIR = "./dpo_qlora_adapter"
FINAL_MERGED_MODEL_PATH = "./Qwen3_0.6B_qlora_merged"

if __name__ == "__main__":
    logger = setup_logger("qlora_training_pipeline")

    # --- 1. Data Preparation (Identical to full tuning) ---
    logger.info("--- Preparing Datasets ---")
    dpo_train, dpo_eval, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT
    )
    sft_train = dpo_train.map(create_sft_format)
    sft_eval = dpo_eval.map(create_sft_format)
    logger.success("Datasets prepared for SFT and DPO.")

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

    # --- 3. Stage 1: Supervised Fine-Tuning (SFT) with QLoRA ---
    logger.info("--- STAGE 1: SUPERVISED FINE-TUNING (SFT) with QLoRA ---")
    
    # Load the base model with 4-bit quantization
    sft_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Get the tokenizer
    _, tokenizer = get_model_and_tokenizer(MODEL_ID)

    sft_training_args = TrainingArguments(
        output_dir="./sft_qlora_training_output",
        num_train_epochs=1,
        per_device_train_batch_size=4, # Can use a larger batch size now
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4, # A higher learning rate is common for LoRA
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    sft_trainer = SFTTrainer(
        model=sft_model,
        train_dataset=sft_train,
        eval_dataset=sft_eval,
        peft_config=peft_config, # Pass the LoRA config here!
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=sft_training_args,
    )

    sft_trainer.train()
    logger.success(f"SFT with QLoRA finished. Adapter saved to {SFT_ADAPTER_DIR}")
    sft_trainer.save_model(SFT_ADAPTER_DIR)

    # Clean up to free VRAM
    del sft_model
    del sft_trainer
    torch.cuda.empty_cache()
    logger.info("Cleaned up SFT resources.")

    # --- 4. Stage 2: Direct Preference Optimization (DPO) with QLoRA ---
    logger.info("--- STAGE 2: DIRECT PREFERENCE OPTIMIZATION (DPO) with QLoRA ---")
    
    dpo_training_args = TrainingArguments(
        output_dir="./dpo_qlora_training_output",
        num_train_epochs=1,
        per_device_train_batch_size=2, # DPO is more memory intensive
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5, # Lower LR for preference tuning
        logging_steps=50,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    # For DPO, we load the SFT-tuned adapter on top of the base model
    # The DPOTrainer needs the base model path and the adapter path
    dpo_trainer = DPOTrainer(
        # The model argument can be the path to the SFT adapter
        model=SFT_ADAPTER_DIR,
        # The base_model_name_or_path is needed if model is a path
        model_init_kwargs={"quantization_config": bnb_config, "device_map": "auto", "trust_remote_code": True},
        ref_model=None, # TRL will handle the reference model
        args=dpo_training_args,
        train_dataset=dpo_train,
        eval_dataset=dpo_eval,
        tokenizer=tokenizer,
        peft_config=peft_config, # Pass the same LoRA config
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
    )
    
    dpo_trainer.train()
    logger.success(f"DPO with QLoRA finished. Final adapter saved to {DPO_ADAPTER_DIR}")
    dpo_trainer.save_model(DPO_ADAPTER_DIR)
    
    # --- 5. Stage 3: Merge Adapter and Save Final Model ---
    logger.info("--- STAGE 3: MERGING THE FINAL ADAPTER ---")
    
    # Reload the base model in full precision (e.g., bfloat16) to merge
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load the DPO adapter onto the base model
    final_model = PeftModel.from_pretrained(base_model, DPO_ADAPTER_DIR)
    
    # Merge the adapter into the base model
    merged_model = final_model.merge_and_unload()
    
    # Save the merged model for easy deployment
    merged_model.save_pretrained(FINAL_MERGED_MODEL_PATH)
    tokenizer.save_pretrained(FINAL_MERGED_MODEL_PATH)
    
    logger.success(f"Successfully merged adapter. Final model saved to {FINAL_MERGED_MODEL_PATH}")