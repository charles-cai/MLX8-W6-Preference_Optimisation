"""
Debug script to understand Qwen3 thinking mode behavior.

Run: uv run debug_thinking.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_thinking_mode():
    model_name = "Qwen/Qwen3-1.7B"  # Or whatever model you're using
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    messages = [
        {"role": "system", "content": "You are a math problem generator."},
        {"role": "user", "content": "Generate a simple math problem."},
    ]
    
    # Test 1: With enable_thinking=True (default)
    print("\n" + "="*60)
    print("TEST 1: enable_thinking=True (default)")
    print("="*60)
    
    try:
        prompt_thinking = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        print(f"Prompt ends with: ...{prompt_thinking[-100:]}")
    except TypeError as e:
        print(f"TypeError: {e}")
        prompt_thinking = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(f"Fallback prompt ends with: ...{prompt_thinking[-100:]}")
    
    inputs = tokenizer(prompt_thinking, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nResponse (first 300 chars):\n{response[:300]}")
    print(f"\nContains <think>: {'<think>' in response}")
    
    # Test 2: With enable_thinking=False
    print("\n" + "="*60)
    print("TEST 2: enable_thinking=False")
    print("="*60)
    
    try:
        prompt_no_thinking = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        print(f"Prompt ends with: ...{prompt_no_thinking[-100:]}")
        
        # Check if prompts are different
        print(f"\nPrompts are different: {prompt_thinking != prompt_no_thinking}")
        if prompt_thinking != prompt_no_thinking:
            # Find the difference
            for i, (c1, c2) in enumerate(zip(prompt_thinking, prompt_no_thinking)):
                if c1 != c2:
                    print(f"First difference at position {i}:")
                    print(f"  thinking: ...{prompt_thinking[max(0,i-20):i+20]}...")
                    print(f"  no_think: ...{prompt_no_thinking[max(0,i-20):i+20]}...")
                    break
            if len(prompt_thinking) != len(prompt_no_thinking):
                print(f"Length difference: {len(prompt_thinking)} vs {len(prompt_no_thinking)}")
        
    except TypeError as e:
        print(f"TypeError: {e}")
        print("This tokenizer doesn't support enable_thinking!")
        return
    
    inputs = tokenizer(prompt_no_thinking, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nResponse (first 300 chars):\n{response[:300]}")
    print(f"\nContains <think>: {'<think>' in response}")
    
    # Test 3: Check tokenizer special tokens
    print("\n" + "="*60)
    print("TEST 3: Special tokens info")
    print("="*60)
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Has chat_template: {hasattr(tokenizer, 'chat_template')}")
    if hasattr(tokenizer, 'added_tokens_encoder'):
        think_tokens = [k for k in tokenizer.added_tokens_encoder.keys() if 'think' in k.lower()]
        print(f"Think-related tokens: {think_tokens}")


if __name__ == "__main__":
    test_thinking_mode()
