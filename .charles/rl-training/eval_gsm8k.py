"""Quick evaluation script for hackathon - tests your trained models."""

# # Quick test (20 samples, ~2-3 min)
# uv run quick_eval.py --num_samples 20

# # Compare with base model (40 samples total, ~5-6 min)
# uv run quick_eval.py --num_samples 20 --compare_base

# # Test curriculum model's task generation
# uv run quick_eval.py --model ./outputs/curriculum/final_model --num_samples 10

# Other Quick Datasets
# Dataset	Size	Download	Best For
# GSM8K	1.3K test	openai/gsm8k	Math word problems ✅
# MATH	5K test	lighteval/MATH	Competition math
# MMLU	14K test	cais/mmlu	General knowledge
# TriviaQA	11K test	trivia_qa	Factual QA


import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from tqdm import tqdm
from loguru import logger

def extract_answer(text: str) -> str | None:
    """Extract answer from \\boxed{} or #### format."""
    # Try boxed format first
    if "\\boxed{" in text:
        try:
            return text.split("\\boxed{")[1].split("}")[0].strip()
        except IndexError:
            pass
    # Try GSM8K #### format
    if "####" in text:
        try:
            return text.split("####")[1].strip().split()[0]
        except IndexError:
            pass
    return None

def evaluate_model(model_path: str, num_samples: int = 20) -> dict:
    """Evaluate model on GSM8K subset."""
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {model_path}")
    logger.info(f"{'='*60}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load GSM8K test set
    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{num_samples}]")
    
    correct = 0
    results = []
    
    for i, sample in enumerate(tqdm(ds, desc="Evaluating", total=num_samples)):
        question = sample["question"]
        # GSM8K answers are after ####
        gold_answer = sample["answer"].split("####")[1].strip()
        
        # Format prompt
        prompt = f"""Solve step by step. Put your final answer in \\boxed{{}}.

Question: {question}

Solution:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):]
        
        # Extract and compare
        pred_answer = extract_answer(generated)
        
        # Normalize for comparison (remove commas, spaces)
        def normalize(x):
            if x is None:
                return None
            return re.sub(r'[,\s$]', '', str(x))
        
        is_correct = normalize(pred_answer) == normalize(gold_answer)
        if is_correct:
            correct += 1
            logger.success(f"[{i+1}/{num_samples}] Gold: {gold_answer}, Pred: {pred_answer}")
        else:
            logger.warning(f"[{i+1}/{num_samples}] Gold: {gold_answer}, Pred: {pred_answer}")
        
        results.append({
            "question": question[:50] + "...",
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": is_correct,
        })
    
    accuracy = correct / num_samples * 100
    logger.info(f"📊 Accuracy: {correct}/{num_samples} = {accuracy:.1f}%")
    
    return {"accuracy": accuracy, "correct": correct, "total": num_samples}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./outputs/executor_iter1/final_model", help="Model path")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--compare_base", action="store_true", help="Also eval base model")
    args = parser.parse_args()
    
    # Evaluate trained model
    trained_results = evaluate_model(args.model, args.num_samples)
    
    # Optionally compare with base models
    if args.compare_base:
        base_results = evaluate_model("Qwen/Qwen3-1.7B", args.num_samples)
        base_pretrain_results = evaluate_model("Qwen/Qwen3-1.7B-Base", args.num_samples)
        
        logger.info(f"{'='*60}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Qwen3-1.7B-Base (pretrain): {base_pretrain_results['accuracy']:.1f}%")
        logger.info(f"Qwen3-1.7B (instruct):      {base_results['accuracy']:.1f}%")
        logger.info(f"Trained Model:              {trained_results['accuracy']:.1f}%")
        logger.info(f"{'='*60}")
        logger.info(f"Improvement over pretrain:  {trained_results['accuracy'] - base_pretrain_results['accuracy']:+.1f}%")
        logger.info(f"Improvement over instruct:  {trained_results['accuracy'] - base_results['accuracy']:+.1f}%")