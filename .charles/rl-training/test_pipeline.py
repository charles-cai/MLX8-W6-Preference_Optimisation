"""
Test script to validate Agent1 pipeline components.

Usage:
    uv run test_pipeline.py              # Run all tests
    uv run test_pipeline.py --quick      # Quick module tests only
    uv run test_pipeline.py --full       # Full integration test
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


def run_command(cmd: str, description: str, check: bool = True) -> bool:
    """Run a command and report success/failure."""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {description}")
    print(f"   Command: {cmd}")
    print("="*60)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=False,
            text=True,
            check=check,
        )
        print(f"✅ PASSED: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"   Error: {e}")
        return False


def test_modules():
    """Test individual modules can be imported and run."""
    print("\n" + "#"*60)
    print("# PHASE 1: Module Tests")
    print("#"*60)
    
    results = []
    
    # Test rewards.py
    results.append(run_command(
        "uv run rewards.py",
        "Reward functions (R_unc, R_tool, R_C)"
    ))
    
    # Test self_consistency.py
    results.append(run_command(
        "uv run self_consistency.py",
        "Self-consistency (p̂, answer extraction)"
    ))
    
    # Test prompts.py
    results.append(run_command(
        "uv run prompts.py",
        "Prompt loading from TOML"
    ))
    
    # Test grpo.py
    results.append(run_command(
        "uv run grpo.py",
        "GRPO trainer configuration"
    ))
    
    return all(results)


def test_curriculum_training(quick: bool = True):
    """Test curriculum agent training."""
    print("\n" + "#"*60)
    print("# PHASE 2: Curriculum Training Test")
    print("#"*60)
    
    output_dir = "./outputs/test_curriculum"
    
    # Clean up previous test
    if Path(output_dir).exists():
        import shutil
        shutil.rmtree(output_dir)
    
    cmd = (
        f"uv run train_curriculum.py "
        f"--prompt_preset data_scientist "
        f"--max_steps {'2' if quick else '5'} "
        f"--num_prompts {'5' if quick else '20'} "
        f"--executor_k 2 "
        f"--output_dir {output_dir} "
        f"--no_wandb"
    )
    
    success = run_command(cmd, "Curriculum agent training")
    
    if success:
        # Verify outputs
        final_model = Path(output_dir) / "final_model"
        generations = list(Path(output_dir).glob("curriculum_generations_*.jsonl"))
        
        checks = []
        
        if final_model.exists():
            print(f"   ✓ Final model saved: {final_model}")
            checks.append(True)
        else:
            print(f"   ✗ Final model NOT found: {final_model}")
            checks.append(False)
        
        if generations:
            print(f"   ✓ Generations logged: {generations[0].name}")
            # Count entries
            with open(generations[0]) as f:
                count = sum(1 for _ in f)
            print(f"   ✓ {count} generation entries")
            checks.append(True)
        else:
            print(f"   ✗ No generation logs found")
            checks.append(False)
        
        return all(checks)
    
    return False


def test_executor_training(curriculum_model_path: str, quick: bool = True):
    """Test executor agent training."""
    print("\n" + "#"*60)
    print("# PHASE 3: Executor Training Test")
    print("#"*60)
    
    output_dir = "./outputs/test_executor"
    
    # Clean up previous test
    if Path(output_dir).exists():
        import shutil
        shutil.rmtree(output_dir)
    
    cmd = (
        f"uv run train_executor.py "
        f"--curriculum_model_path {curriculum_model_path} "
        f"--max_steps {'2' if quick else '10'} "
        f"--num_tasks {'5' if quick else '20'} "
        f"--k_samples 2 "
        f"--output_dir {output_dir} "
        f"--no_wandb"
    )
    
    success = run_command(cmd, "Executor agent training")
    
    if success:
        # Verify outputs
        final_model = Path(output_dir) / "final_model"
        frontier_tasks = list(Path(output_dir).glob("frontier_tasks_*.jsonl"))
        
        checks = []
        
        if final_model.exists():
            print(f"   ✓ Final model saved: {final_model}")
            checks.append(True)
        else:
            print(f"   ✗ Final model NOT found: {final_model}")
            checks.append(False)
        
        if frontier_tasks:
            print(f"   ✓ Frontier tasks saved: {frontier_tasks[0].name}")
            with open(frontier_tasks[0]) as f:
                count = sum(1 for _ in f)
            print(f"   ✓ {count} frontier tasks")
            checks.append(True)
        else:
            print(f"   ✗ No frontier tasks found")
            checks.append(False)
        
        return all(checks)
    
    return False


def test_coevolution(quick: bool = True):
    """Test full co-evolution loop."""
    print("\n" + "#"*60)
    print("# PHASE 4: Co-Evolution Test")
    print("#"*60)
    
    output_dir = "./outputs/test_coevolution"
    
    # Clean up previous test
    if Path(output_dir).exists():
        import shutil
        shutil.rmtree(output_dir)
    
    cmd = (
        f"uv run main.py "
        f"--iterations 1 "
        f"--curriculum_max_steps {'2' if quick else '5'} "
        f"--executor_max_steps {'2' if quick else '10'} "
        f"--num_prompts {'5' if quick else '20'} "
        f"--num_tasks {'5' if quick else '20'} "
        f"--output_dir {output_dir} "
        f"--no_wandb"
    )
    
    success = run_command(cmd, "Full co-evolution loop")
    
    if success:
        # Verify outputs
        history_file = Path(output_dir) / "coevolution_history.json"
        curriculum_dir = Path(output_dir) / "curriculum_iter1"
        executor_dir = Path(output_dir) / "executor_iter1"
        
        checks = []
        
        if history_file.exists():
            print(f"   ✓ History saved: {history_file}")
            import json
            with open(history_file) as f:
                history = json.load(f)
            print(f"   ✓ Iterations completed: {len(history.get('iterations', []))}")
            checks.append(True)
        else:
            print(f"   ✗ History NOT found")
            checks.append(False)
        
        if curriculum_dir.exists():
            print(f"   ✓ Curriculum output dir: {curriculum_dir}")
            checks.append(True)
        else:
            print(f"   ✗ Curriculum dir NOT found")
            checks.append(False)
        
        if executor_dir.exists():
            print(f"   ✓ Executor output dir: {executor_dir}")
            checks.append(True)
        else:
            print(f"   ✗ Executor dir NOT found")
            checks.append(False)
        
        return all(checks)
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Test Agent1 pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick tests only (modules)")
    parser.add_argument("--full", action="store_true", help="Full integration test")
    parser.add_argument("--curriculum", action="store_true", help="Test curriculum only")
    parser.add_argument("--executor", action="store_true", help="Test executor only")
    parser.add_argument("--coevolution", action="store_true", help="Test co-evolution only")
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 AGENT1 PIPELINE TESTS")
    print(f"   Time: {datetime.now().isoformat()}")
    print("="*60)
    
    results = {}
    
    # Always run module tests
    results["modules"] = test_modules()
    
    if args.quick:
        # Quick mode: only module tests
        pass
    elif args.curriculum:
        results["curriculum"] = test_curriculum_training(quick=True)
    elif args.executor:
        # Need curriculum model first
        curriculum_path = "./outputs/test_curriculum/final_model"
        if not Path(curriculum_path).exists():
            print(f"⚠️ Curriculum model not found at {curriculum_path}")
            print("   Running curriculum test first...")
            results["curriculum"] = test_curriculum_training(quick=True)
        results["executor"] = test_executor_training(curriculum_path, quick=True)
    elif args.coevolution:
        results["coevolution"] = test_coevolution(quick=True)
    elif args.full or not any([args.quick, args.curriculum, args.executor, args.coevolution]):
        # Full test: all phases
        results["curriculum"] = test_curriculum_training(quick=True)
        if results["curriculum"]:
            results["executor"] = test_executor_training(
                "./outputs/test_curriculum/final_model", 
                quick=True
            )
        results["coevolution"] = test_coevolution(quick=True)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
