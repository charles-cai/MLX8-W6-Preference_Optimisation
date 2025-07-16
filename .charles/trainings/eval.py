# /qwen_finetuning_project/evaluate.py

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Import our data preparation function
from data_prep_unified import load_and_prepare_datasets
from logger_utils import setup_logger

# --- Configuration ---
# Path to the final model you want to evaluate
# This could be the full-tuned one or the merged QLoRA one
FINAL_MODEL_PATH = "./.data/sft_full_results" 
DATASET_NAME = "CarperAI/openai_summarize_comparisons"
NUM_TEST_SAMPLES = 20 # 1000 Evaluate on a subset of the test set for speed. Use len(test_dataset) for full eval.
NUM_SAMPLES_TO_SHOW = 20

if __name__ == "__main__":
    logger = setup_logger("evaluation_script")

    # --- 1. Load Model and Tokenizer ---
    logger.info(f"Loading final model from: {FINAL_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        FINAL_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
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