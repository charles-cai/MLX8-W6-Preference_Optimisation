# Verifying Executor Correctness in Agent0 Framework

Based on the Agent0 paper and your implementation, here's how verification works and how to test your trained models.

## How Agent0 Verifies Executor Correctness

### Paper Methodology (Section 3.3)

The paper uses **self-consistency with pseudo-labels** - there's no ground truth oracle:

1. **Pseudo-Label Generation**: For each task, sample k responses from executor, use **majority voting** to get pseudo-label ỹ
2. **Reward Signal**: `R = 𝟙(answer == ỹ)` - binary reward if executor matches majority answer
3. **Validation**: Performance is measured on **external benchmarks** (MATH, GSM8K, AIME, etc.)

```
Key Insight: The paper validates executor capability on HELD-OUT BENCHMARKS,
not on the self-generated curriculum tasks.
```

### What You Should Test

| Test Type | Purpose | Method |
|-----------|---------|--------|
| **Benchmark Eval** | True capability measurement | Run on GSM8K, MATH500, etc. |
| **Self-Consistency** | Internal coherence | Check p̂ stability across iterations |
| **Task Difficulty** | Curriculum evolution | Compare executor pass rates on curriculum tasks |
| **Tool Use** | TIR capability | Count code executions, verify outputs |

---

## Test Commands for Your Trained Models

### 1. Benchmark Evaluation (Most Important)

````bash
cd /workspace/_github/charles-cai/MLX8-W6-Preference_Optimisation/.charles/rl-training

# Create evaluation script
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Load your trained executor
model_path = './outputs/executor/final_model'
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Simple GSM8K-style test
test_problems = [
    {'question': 'Janet has 3 apples. She buys 2 more. How many apples does she have?', 'answer': '5'},
    {'question': 'A rectangle has length 4 and width 3. What is its area?', 'answer': '12'},
    {'question': 'If x + 5 = 12, what is x?', 'answer': '7'},
]

correct = 0
for p in test_problems:
    prompt = f'Solve step by step. Put final answer in \\\\boxed{{}}.\\n\\nQuestion: {p[\"question\"]}'
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if '\\\\boxed{' in response:
        answer = response.split('\\\\boxed{')[1].split('}')[0]
        if answer.strip() == p['answer']:
            correct += 1
            print(f'✅ Correct: {p[\"question\"]} -> {answer}')
        else:
            print(f'❌ Wrong: {p[\"question\"]} -> {answer} (expected {p[\"answer\"]})')
    else:
        print(f'❌ No boxed answer: {p[\"question\"]}')

print(f'\\nAccuracy: {correct}/{len(test_problems)} = {correct/len(test_problems)*100:.1f}%')
"
````

### 2. Self-Consistency Verification

````python
"""Verify executor self-consistency on held-out tasks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from self_consistency import compute_self_consistency, extract_answer
import json

def evaluate_self_consistency(
    model_path: str,
    test_tasks: list[str],
    k_samples: int = 4,
):
    """
    Test if executor produces consistent answers across samples.
    
    High p̂ (>0.8) = confident, consistent answers
    Low p̂ (<0.5) = uncertain, inconsistent answers
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    results = []
    for task in test_tasks:
        # Generate k responses
        responses = []
        for _ in range(k_samples):
            inputs = tokenizer(task, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=1.0,
                do_sample=True,
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(response)
        
        # Compute self-consistency
        answers = [extract_answer(r) for r in responses]
        p_hat, majority = compute_self_consistency(answers)
        
        results.append({
            "task": task[:100] + "...",
            "p_hat": p_hat,
            "majority_answer": majority,
            "unique_answers": len(set(a for a in answers if a)),
        })
        print(f"p̂={p_hat:.2f} | {len(set(answers))} unique answers | Task: {task[:50]}...")
    
    avg_p_hat = sum(r["p_hat"] for r in results) / len(results)
    print(f"\n📊 Average p̂: {avg_p_hat:.3f}")
    print(f"   (Higher = more consistent, target > 0.7 for easy tasks)")
    
    return results

if __name__ == "__main__":
    # Test tasks (use your curriculum-generated tasks or standard problems)
    test_tasks = [
        "Solve: What is 15 * 23? Put your answer in \\boxed{}.",
        "If a train travels 60 miles per hour for 2.5 hours, how far does it travel? Put answer in \\boxed{}.",
        "What is the derivative of x^3? Put answer in \\boxed{}.",
    ]
    
    evaluate_self_consistency(
        model_path="./outputs/executor/final_model",
        test_tasks=test_tasks,
        k_samples=4,
    )
````

### 3. Compare Base vs Trained Executor

````bash
# Create comparison script
cat > /workspace/_github/charles-cai/MLX8-W6-Preference_Optimisation/.charles/rl-training/compare_models.py << 'EOF'
"""Compare base model vs trained executor on same tasks."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def evaluate_model(model_path: str, test_problems: list[dict]) -> dict:
    """Evaluate model on test problems."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    correct = 0
    results = []
    
    for p in test_problems:
        prompt = f"Solve step by step. Put final answer in \\boxed{{}}.\n\nQuestion: {p['question']}"
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
        
        # Extract answer
        extracted = None
        if "\\boxed{" in response:
            try:
                extracted = response.split("\\boxed{")[1].split("}")[0].strip()
            except:
                pass
        
        is_correct = extracted == p["answer"] if extracted else False
        if is_correct:
            correct += 1
        
        results.append({
            "question": p["question"],
            "expected": p["answer"],
            "extracted": extracted,
            "correct": is_correct,
        })
    
    return {
        "accuracy": correct / len(test_problems),
        "correct": correct,
        "total": len(test_problems),
        "results": results,
    }

if __name__ == "__main__":
    # Test problems
    test_problems = [
        {"question": "What is 7 * 8?", "answer": "56"},
        {"question": "What is 144 / 12?", "answer": "12"},
        {"question": "If x = 5, what is 2x + 3?", "answer": "13"},
        {"question": "What is the square root of 81?", "answer": "9"},
        {"question": "What is 15% of 200?", "answer": "30"},
    ]
    
    print("=" * 60)
    print("BASE MODEL (Qwen/Qwen3-1.7B)")
    print("=" * 60)
    base_results = evaluate_model("Qwen/Qwen3-1.7B", test_problems)
    print(f"Accuracy: {base_results['accuracy']*100:.1f}% ({base_results['correct']}/{base_results['total']})")
    
    print("\n" + "=" * 60)
    print("TRAINED EXECUTOR (outputs/executor/final_model)")
    print("=" * 60)
    trained_results = evaluate_model("./outputs/executor/final_model", test_problems)
    print(f"Accuracy: {trained_results['accuracy']*100:.1f}% ({trained_results['correct']}/{trained_results['total']})")
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    delta = (trained_results['accuracy'] - base_results['accuracy']) * 100
    print(f"Improvement: {delta:+.1f}%")
EOF

uv run compare_models.py
````

### 4. Verify Curriculum-Executor Interaction

````bash
# Test that curriculum generates tasks and executor can attempt them
cat > /workspace/_github/charles-cai/MLX8-W6-Preference_Optimisation/.charles/rl-training/test_interaction.py << 'EOF'
"""Test curriculum-executor interaction after training."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import load_prompts

def test_interaction():
    """
    1. Curriculum generates a task
    2. Executor attempts to solve it
    3. Check if executor produces reasonable output
    """
    # Load models
    curriculum_path = "./outputs/curriculum/final_model"
    executor_path = "./outputs/executor/final_model"
    
    print("Loading curriculum model...")
    curriculum_model = AutoModelForCausalLM.from_pretrained(
        curriculum_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    curriculum_tokenizer = AutoTokenizer.from_pretrained(curriculum_path)
    
    print("Loading executor model...")
    executor_model = AutoModelForCausalLM.from_pretrained(
        executor_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    executor_tokenizer = AutoTokenizer.from_pretrained(executor_path)
    
    # Get curriculum prompt
    prompts = load_prompts("data_scientist")
    curriculum_prompt = prompts["curriculum_system"] + "\n" + prompts["curriculum_user"]
    
    # Generate task from curriculum
    print("\n" + "=" * 60)
    print("CURRICULUM GENERATING TASK...")
    print("=" * 60)
    
    inputs = curriculum_tokenizer(curriculum_prompt, return_tensors="pt").to(curriculum_model.device)
    with torch.no_grad():
        outputs = curriculum_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=1.0,
            do_sample=True,
        )
    task = curriculum_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract question
    if "<question>" in task and "</question>" in task:
        question = task.split("<question>")[1].split("</question>")[0].strip()
    else:
        question = task[len(curriculum_prompt):].strip()
    
    print(f"Generated Task:\n{question[:500]}...")
    
    # Executor attempts task
    print("\n" + "=" * 60)
    print("EXECUTOR ATTEMPTING TASK...")
    print("=" * 60)
    
    executor_prompt = f"Solve the following problem step by step. Put your final answer in \\boxed{{}}.\n\n{question}"
    inputs = executor_tokenizer(executor_prompt, return_tensors="pt").to(executor_model.device)
    
    with torch.no_grad():
        outputs = executor_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
        )
    response = executor_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Executor Response:\n{response[len(executor_prompt):][:800]}...")
    
    # Check for answer
    has_boxed = "\\boxed{" in response
    has_code = "```python" in response or "```" in response
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"✓ Has boxed answer: {has_boxed}")
    print(f"✓ Contains code: {has_code}")
    print(f"✓ Response length: {len(response)} chars")
    
    return {
        "task": question,
        "response": response,
        "has_boxed": has_boxed,
        "has_code": has_code,
    }

if __name__ == "__main__":
    test_interaction()
EOF

uv run test_interaction.py
````

---

## Expected Results After Training

### For Well-Trained Models (Per Paper):

| Metric | Iteration 1 | Iteration 2 | Iteration 3 |
|--------|-------------|-------------|-------------|
| **Math AVG** | 51.9% | 52.2% | 52.5% |
| **p̂ on curriculum tasks** | ~0.5 | ~0.5 | ~0.5 |
| **Executor pass rate on Iter 1 tasks** | 64% | - | - |
| **Executor pass rate on Iter 3 tasks** | - | - | 51% |

### For Your Sanity Check (Limited Training):

| What to Expect | Acceptable | Concerning |
|----------------|------------|------------|
| Curriculum generates valid tasks | `<question>` tags present | No structure |
| Executor produces responses | Any output | Empty/truncated |
| Some self-consistency | p̂ varies (0.3-0.8) | All p̂=1.0 or all p̂=0 |
| Benchmark accuracy | ≥ base model | < base model |

---

## Quick Verification Commands

````bash
cd /workspace/_github/charles-cai/MLX8-W6-Preference_Optimisation/.charles/rl-training

# 1. Check models exist
ls -la ./outputs/curriculum/final_model/
ls -la ./outputs/executor/final_model/

# 2. Check training logs
cat ./outputs/curriculum/training_info.json | jq .
cat ./outputs/executor/training_info.json | jq .

# 3. View sample curriculum tasks
cat ./outputs/curriculum/curriculum_generations_*.jsonl | head -1 | jq '.task'

# 4. View executor frontier tasks
cat ./outputs/executor/frontier_tasks_*.jsonl | head -1 | jq '{task: .task, p_hat: .p_hat, majority_answer: .majority_answer}'

# 5. Run quick test
uv run compare_models.py
````