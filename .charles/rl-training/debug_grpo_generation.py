"""
Debug script to test generation with the same setup as train_curriculum.py
but without the full GRPO training loop.

Run: uv run debug_grpo_generation.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from prompts import load_prompts

def test_generation():
    model_name = "Qwen/Qwen3-1.7B"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load prompts same as train_curriculum.py
    prompt_config = load_prompts("data_scientist")
    print(f"\nPrompt preset: {prompt_config.name}")
    print(f"System prompt: {prompt_config.curriculum_system[:200]}...")
    print(f"User prompt: {prompt_config.curriculum_user[:200]}...")
    
    messages = [
        {"role": "system", "content": prompt_config.curriculum_system},
        {"role": "user", "content": prompt_config.curriculum_user},
    ]
    
    # Test with enable_thinking=False (same as train_curriculum.py)
    print("\n" + "="*60)
    print("TEST: Same prompt format as train_curriculum.py")
    print("="*60)
    
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    print(f"\nFull prompt:\n{prompt_str}")
    print(f"\nPrompt length: {len(prompt_str)} chars")
    
    # Tokenize
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    print(f"Input token count: {inputs['input_ids'].shape[1]}")
    
    # Generate with same params as GRPO would use
    print("\n" + "="*60)
    print("Generating with temperature=1.0, do_sample=True (GRPO defaults)")
    print("="*60)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=True,
        top_p=0.99,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )
    
    print(f"\nGenerated response:\n{response}")
    print(f"\nResponse length: {len(response)} chars")
    print(f"Contains <question>: {'<question>' in response}")
    print(f"Contains </question>: {'</question>' in response}")
    boxed_check = '\\boxed' in response or 'boxed' in response
    print(f"Contains \\boxed: {boxed_check}")
    
    # Test with lower temperature
    print("\n" + "="*60)
    print("Generating with temperature=0.7 (more focused)")
    print("="*60)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )
    
    print(f"\nGenerated response:\n{response}")

    # =========================================================================
    # TEST: Simulate what GRPOTrainer might be doing
    # =========================================================================
    print("\n" + "="*60)
    print("TEST: Simulating GRPOTrainer dataset format")
    print("="*60)
    
    # Create dataset exactly like train_curriculum.py does
    dataset = Dataset.from_list([{
        "prompt": prompt_str,
        "id": "curriculum_0",
    }])
    
    print(f"Dataset columns: {dataset.column_names}")
    print(f"First prompt from dataset (first 200 chars): {dataset[0]['prompt'][:200]}...")
    print(f"First prompt from dataset (last 200 chars): ...{dataset[0]['prompt'][-200:]}")
    
    # Check if the prompt is being tokenized correctly when loaded from dataset
    dataset_prompt = dataset[0]['prompt']
    dataset_inputs = tokenizer(dataset_prompt, return_tensors="pt").to(model.device)
    print(f"\nDataset prompt token count: {dataset_inputs['input_ids'].shape[1]}")
    print(f"Original prompt token count: {inputs['input_ids'].shape[1]}")
    print(f"Token counts match: {dataset_inputs['input_ids'].shape[1] == inputs['input_ids'].shape[1]}")
    
    # Generate from dataset prompt
    outputs = model.generate(
        **dataset_inputs,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=True,
        top_p=0.99,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(
        outputs[0][dataset_inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    )
    
    print(f"\nGenerated from dataset prompt:\n{response[:500]}...")
    
    # =========================================================================
    # TEST: Check what happens with TRL's expected format
    # =========================================================================
    print("\n" + "="*60)
    print("TEST: TRL GRPOTrainer expected formats")
    print("="*60)
    
    # TRL might expect different column names or formats
    # Check if it's looking for 'text' instead of 'prompt'
    dataset_text = Dataset.from_list([{
        "text": prompt_str,
        "id": "curriculum_0",
    }])
    print(f"Dataset with 'text' column: {dataset_text.column_names}")
    
    # TRL might also just expect raw messages, not pre-formatted
    dataset_messages = Dataset.from_list([{
        "messages": messages,
        "id": "curriculum_0",
    }])
    print(f"Dataset with 'messages' column: {dataset_messages.column_names}")


if __name__ == "__main__":
    test_generation()
