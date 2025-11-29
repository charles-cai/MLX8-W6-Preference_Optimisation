"""
Debug script to test TRL GRPOTrainer generation behavior.

This script creates a minimal GRPO setup to see exactly what the trainer
is doing with prompts and generation.

Run: uv run debug_grpo_trainer.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from prompts import load_prompts

def simple_reward(prompts, completions, **kwargs):
    """Simple reward function that just logs what it receives."""
    print("\n" + "="*60)
    print("REWARD FUNCTION CALLED")
    print("="*60)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Number of completions: {len(completions)}")
    
    if prompts:
        print(f"\nFirst prompt type: {type(prompts[0])}")
        print(f"First prompt (first 300 chars):\n{str(prompts[0])[:300]}...")
        print(f"First prompt (last 200 chars):\n...{str(prompts[0])[-200:]}")
    
    if completions:
        print(f"\nFirst completion type: {type(completions[0])}")
        print(f"First completion (full, max 1000 chars):\n{str(completions[0])[:1000]}")
        
        # Check for garbage patterns
        completion = str(completions[0])
        if '::::' in completion:
            print("\n⚠️ WARNING: Detected repeated colons (garbage pattern)")
        if '<question>' in completion:
            print("✅ Contains <question> tag")
        else:
            print("❌ Missing <question> tag")
    
    # Return dummy rewards
    return [0.5] * len(completions)

# Add __name__ attribute for TRL compatibility
simple_reward.__name__ = "simple_reward"


def test_grpo_trainer():
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
    
    # Check tokenizer settings
    print(f"\nTokenizer padding_side: {tokenizer.padding_side}")
    print(f"Tokenizer pad_token: {tokenizer.pad_token}")
    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    
    # Load prompts
    prompt_config = load_prompts("data_scientist")
    
    messages = [
        {"role": "system", "content": prompt_config.curriculum_system},
        {"role": "user", "content": prompt_config.curriculum_user},
    ]
    
    # Create formatted prompt (same as train_curriculum.py)
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    print(f"\nFormatted prompt (last 300 chars):\n...{prompt_str[-300:]}")
    
    # Create dataset with just 2 samples for quick test
    dataset = Dataset.from_list([
        {"prompt": prompt_str, "id": "test_0"},
        {"prompt": prompt_str, "id": "test_1"},
    ])
    
    print(f"\nDataset columns: {dataset.column_names}")
    print(f"Dataset size: {len(dataset)}")
    
    # Print available GRPOConfig parameters
    print("\n" + "="*60)
    print("Available GRPOConfig parameters:")
    print("="*60)
    import inspect
    sig = inspect.signature(GRPOConfig.__init__)
    for name, param in sig.parameters.items():
        if name not in ['self', 'kwargs']:
            print(f"  {name}: {param.default if param.default != inspect.Parameter.empty else 'REQUIRED'}")
    
    # Create minimal GRPO config with correct parameter names for TRL 0.19.1
    # Note: effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
    # This must be divisible by num_generations
    config = GRPOConfig(
        output_dir="./debug_grpo_output",
        num_generations=2,  # Minimum for GRPO
        max_completion_length=512,  # TRL 0.19+ uses this instead of max_new_tokens
        temperature=1.0,
        per_device_train_batch_size=2,  # Must be >= num_generations and divisible by it
        gradient_accumulation_steps=1,
        max_steps=1,  # Just one step for debugging
        logging_steps=1,
        report_to="none",
        use_vllm=False,
    )
    
    print(f"\nGRPO Config:")
    print(f"  num_generations: {config.num_generations}")
    print(f"  max_completion_length: {config.max_completion_length}")
    print(f"  temperature: {config.temperature}")
    
    # Check if config has any chat template settings
    for attr in dir(config):
        if 'chat' in attr.lower() or 'template' in attr.lower() or 'format' in attr.lower():
            val = getattr(config, attr, 'N/A')
            if not callable(val):
                print(f"  {attr}: {val}")
    
    print("\n" + "="*60)
    print("Creating GRPOTrainer...")
    print("="*60)
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[simple_reward],
        args=config,  # Changed from 'config' to 'args'
    )
    
    print("\n" + "="*60)
    print("Starting training (1 step)...")
    print("="*60)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Debug complete!")
    print("="*60)


if __name__ == "__main__":
    test_grpo_trainer()
