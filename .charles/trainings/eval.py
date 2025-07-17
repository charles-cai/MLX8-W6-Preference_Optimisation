# /qwen_finetuning_project/evaluate.py

import torch
import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Import our data preparation function
from data_prep_unified import load_and_prepare_datasets
from logger_utils import setup_logger

# Load environment variables
load_dotenv()

# --- Configuration ---
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT or DPO model")
    parser.add_argument("--sft", action="store_true", help="Use SFT model for evaluation")
    parser.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K", "4")), 
                        help="Top K parameter for SFT model path")
    parser.add_argument("--dpo", action="store_true", help="Use DPO model for evaluation")
    
    args = parser.parse_args()
    
    # Default to DPO if no specific option is provided
    if not args.sft and not args.dpo:
        args.dpo = True
    
    return args

args = get_args()

TOP_K = args.top_k  # Command line argument overwrites .env configuration

# Determine model path based on arguments
if args.sft:
    SFT_OUTPUT_DIR_PREFIX = os.environ.get("SFT_OUTPUT_DIR_PREFIX", "./.data/sft_full_top_")
    EVAL_MODEL_PATH = SFT_OUTPUT_DIR_PREFIX + str(TOP_K)
elif args.dpo:
    EVAL_MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "./.data/dpo_qlora_merged")

DATASET_NAME = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")
NUM_TEST_SAMPLES = int(os.environ.get("NUM_TEST_SAMPLES", "20"))
NUM_SAMPLES_TO_SHOW = int(os.environ.get("NUM_SAMPLES_TO_SHOW", "20"))

if __name__ == "__main__":
    logger = setup_logger("evaluation_script")

    # --- 1. Load Model and Tokenizer ---
    model_type = "SFT" if args.sft else "DPO"
    logger.info(f"Loading {model_type} model from: {EVAL_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        EVAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_PATH)
    model.eval() # Set model to evaluation mode

    # --- 2. Load and Prepare Test Data ---
    logger.info(f"Loading and preparing test dataset...")
    # We don't need train/eval sets here, so we can ignore them with `_`
    _, _, test_dataset = load_and_prepare_datasets(dataset_name=DATASET_NAME)
    
    # For a quick evaluation, use a subset. For the final paper-ready result, use the full set.
    if NUM_TEST_SAMPLES > 0 and NUM_TEST_SAMPLES < len(test_dataset):
        test_dataset = test_dataset.select(range(NUM_TEST_SAMPLES))
    
    logger.info(f"Evaluating on {len(test_dataset)} samples from the test set.")

    # --- 3. Generate Summaries ---
    logger.info("Generating summaries for the test set...")
    predictions = []
    references = []

    # Show sample results for the first few examples

    shown = 0

    for sample in tqdm(test_dataset):
        # We use the prompt and the 'chosen' summary as the ground truth reference
        prompt_text = sample['prompt'] + "\n\nSummary:\n"
        reference_summary = sample['chosen']
        
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=512, # Limit the length of the generated summary
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode the generated summary, removing the prompt part
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_summary = full_output.split("Summary:")[-1].strip()
        
        predictions.append(generated_summary)
        references.append(reference_summary)

        # Show prompt, reference, and prediction for the first few samples
        if shown < NUM_SAMPLES_TO_SHOW:
            logger.info("--- Sample Result ---")
            logger.info(f"Prompt:{sample['prompt']}")
            logger.success(f"Reference Summary::: {reference_summary}")
            logger.warning(f"Generated Summary::: {generated_summary}\n")
            
            shown += 1

    # --- 4. Compute ROUGE Scores ---
    logger.info("Computing ROUGE scores...") 
    rouge = evaluate.load("rouge")
    
    results = rouge.compute(predictions=predictions, references=references)

    logger.success("--- FINAL EVALUATION RESULTS ---")
    logger.success(f"ROUGE-1: {results['rouge1'] * 100:.2f}")
    logger.success(f"ROUGE-2: {results['rouge2'] * 100:.2f}")
    logger.success(f"ROUGE-L: {results['rougeL'] * 100:.2f}")
    logger.success(f"ROUGE-Lsum: {results['rougeLsum'] * 100:.2f}")