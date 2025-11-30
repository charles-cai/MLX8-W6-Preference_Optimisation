"""
Gradio Demo for Agent1: Mini Edge Data Scientist

Showcases the co-evolutionary training results:
1. Curriculum Agent generates challenging questions
2. Compare Base Model vs Trained Executor on solving them

Usage:
    uv run demo.py
    uv run demo.py --curriculum_model ./outputs/curriculum/final_model --executor_model ./outputs/executor/final_model
"""

import os
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Default paths
DEFAULT_BASE_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen3-1.7B")
DEFAULT_CURRICULUM_MODEL = "./outputs/curriculum_iter1/final_model"
DEFAULT_EXECUTOR_MODEL = "./outputs/executor_iter1/final_model"

# Model card information
MODEL_CARDS = {
    "curriculum": {
        "name": "curriculum_iter1",
        "base_model": "Qwen/Qwen3-1.7B",
        "description": "Fine-tuned with GRPO to generate challenging data science questions at the executor's frontier difficulty.",
        "training": "TRL GRPO training on self-generated curriculum",
    },
    "executor": {
        "name": "executor_iter1", 
        "base_model": "Qwen/Qwen3-1.7B",
        "description": "Fine-tuned with GRPO to solve challenging tasks using self-consistency pseudo-labels.",
        "training": "TRL GRPO training on frontier-filtered tasks",
    },
    "base": {
        "name": "Qwen3-1.7B",
        "base_model": "Qwen/Qwen3-1.7B",
        "description": "Base instruction-tuned model without Agent1 co-evolutionary training.",
        "training": "Original Qwen3 instruction tuning",
    },
}

# Global model cache
_model_cache = {}

# Vote counters
vote_counts = {"base": 0, "executor": 0}


def load_model(model_path: str, cache_key: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with caching - matches eval_gsm8k.py approach."""
    if cache_key in _model_cache:
        logger.info(f"Using cached model: {cache_key}")
        return _model_cache[cache_key]
    
    logger.info(f"Loading model: {model_path}")
    
    # Check if local path exists
    if model_path.startswith("./") or model_path.startswith("/"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Load model directly like eval_gsm8k.py does
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    _model_cache[cache_key] = (model, tokenizer)
    logger.success(f"Model loaded: {model_path}")
    
    return model, tokenizer


def generate_question(curriculum_model_path: str) -> str:
    """Generate a data scientist question using the curriculum model."""
    try:
        gr.Info("Loading curriculum model...")
        logger.info(f"Loading curriculum model from: {curriculum_model_path}")
        model, tokenizer = load_model(curriculum_model_path, "curriculum")
        
        gr.Info("Preparing prompt...")
        # Load prompts from TOML
        from prompts import load_prompts
        prompt_config = load_prompts("data_scientist")
        
        messages = [
            {"role": "system", "content": prompt_config.curriculum_system},
            {"role": "user", "content": prompt_config.curriculum_user},
        ]
        
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        gr.Info("Generating question (this may take a minute)...")
        logger.info("Starting generation...")
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.9,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        logger.info("Decoding response...")
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Strip thinking tags if present
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        gr.Info("✅ Question generated!")
        logger.success("Question generated successfully")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return f"❌ Error: Model not found at '{curriculum_model_path}'. Please check the path."
    except Exception as e:
        logger.error(f"Error generating question: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error generating question: {str(e)}"


def solve_question(question: str, model_path: str, cache_key: str) -> str:
    """Solve a question using the specified model."""
    try:
        logger.info(f"Loading {cache_key} model from: {model_path}")
        model, tokenizer = load_model(model_path, cache_key)
        
        # Extract just the question if in <question> tags
        question_match = re.search(r'<question>(.*?)</question>', question, re.DOTALL)
        if question_match:
            clean_question = question_match.group(1).strip()
        else:
            clean_question = question
        
        # Load executor prompt
        from prompts import load_prompts
        prompt_config = load_prompts("data_scientist")
        
        messages = [
            {"role": "system", "content": prompt_config.executor_system},
            {"role": "user", "content": clean_question},
        ]
        
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        logger.info(f"Generating {cache_key} response...")
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        # Strip thinking tags if present
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        logger.success(f"{cache_key} response generated")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        return f"❌ Error: Model not found at '{model_path}'. Please check the path."
    except Exception as e:
        logger.error(f"Error solving question with {cache_key}: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


def solve_with_both_models(
    question: str,
    base_model_path: str,
    executor_model_path: str,
) -> Tuple[str, str]:
    """Solve question with both base and executor models."""
    if not question or question.strip() == "":
        return "Please generate a question first.", "Please generate a question first."
    
    gr.Info("Solving with base model...")
    base_answer = solve_question(question, base_model_path, "base")
    
    gr.Info("Solving with executor model...")
    executor_answer = solve_question(question, executor_model_path, "executor")
    
    gr.Info("✅ Both models done!")
    return base_answer, executor_answer


def generate_voting_prompt(
    question: str,
    base_answer: str,
    executor_answer: str,
) -> str:
    """Generate a formatted prompt for voting comparison and return it."""
    if not question or question.strip() == "":
        return "Please generate a question and solve with both models first."
    
    if not base_answer or not executor_answer:
        return "Please solve with both models first."
    
    # Extract clean question if in tags
    question_match = re.search(r'<question>(.*?)</question>', question, re.DOTALL)
    if question_match:
        clean_question = question_match.group(1).strip()
    else:
        clean_question = question
    
    prompt = f"""Please vote which answer is better and provide a concise single line reason.

==========
QUESTION:
==========
{clean_question}

==========
BASE MODEL ANSWER (Qwen3-1.7B):
==========
{base_answer}

==========
FINE-TUNED MODEL ANSWER (Executor Agent):
==========
{executor_answer}

==========
YOUR VOTE:
==========
Which answer is better? (Base Model / Fine-tuned Model)
Reason: """
    
    return prompt


def copy_to_clipboard(text: str) -> str:
    """Return text for clipboard copy via JS."""
    if not text or text.strip() == "":
        gr.Info("Nothing to copy!")
        return ""
    gr.Info("📋 Copied to clipboard!")
    return text


def generate_and_copy_voting_prompt(
    question: str,
    base_answer: str,
    executor_answer: str,
) -> str:
    """Generate voting prompt and copy to clipboard."""
    prompt = generate_voting_prompt(question, base_answer, executor_answer)
    if prompt.startswith("Please"):
        gr.Warning(prompt)
        return ""
    gr.Info("📋 Voting prompt copied to clipboard!")
    return prompt


def vote_base() -> str:
    """Vote for base model."""
    vote_counts["base"] += 1
    return f"🔵 Base Model: {vote_counts['base']} votes | 🟢 Trained Executor: {vote_counts['executor']} votes"


def vote_executor() -> str:
    """Vote for executor model."""
    vote_counts["executor"] += 1
    return f"🔵 Base Model: {vote_counts['base']} votes | 🟢 Trained Executor: {vote_counts['executor']} votes"


def reset_votes() -> str:
    """Reset vote counters."""
    vote_counts["base"] = 0
    vote_counts["executor"] = 0
    return f"🔵 Base Model: {vote_counts['base']} votes | 🟢 Trained Executor: {vote_counts['executor']} votes"


def create_demo(
    base_model: str = DEFAULT_BASE_MODEL,
    curriculum_model: str = DEFAULT_CURRICULUM_MODEL,
    executor_model: str = DEFAULT_EXECUTOR_MODEL,
) -> gr.Blocks:
    """Create the Gradio demo interface."""
    
    with gr.Blocks(
        title="Agent1: Mini Edge Data Scientist",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # 🤖 Agent1: Mini Edge Data Scientist Demo
        
        This demo showcases the results of co-evolutionary training:
        1. **Curriculum Agent** generates challenging data science questions
        2. **Base Model** vs **Trained Executor** compete to solve them
        3. **You decide** which answer is better!
        
        ---
        """)
        
        # Model path configuration (collapsible)
        with gr.Accordion("⚙️ Model Configuration", open=False):
            with gr.Row():
                curriculum_path = gr.Textbox(
                    label=f"Curriculum Model Path ({MODEL_CARDS['curriculum']['name']})",
                    value=curriculum_model,
                    interactive=True,
                    info=MODEL_CARDS['curriculum']['description'],
                )
            with gr.Row():
                base_path = gr.Textbox(
                    label=f"Base Model Path ({MODEL_CARDS['base']['name']})",
                    value=base_model,
                    interactive=True,
                    info=MODEL_CARDS['base']['description'],
                )
                executor_path = gr.Textbox(
                    label=f"Trained Executor Model Path ({MODEL_CARDS['executor']['name']})",
                    value=executor_model,
                    interactive=True,
                    info=MODEL_CARDS['executor']['description'],
                )
        
        gr.Markdown("## 📝 Step 1: Generate a Question")
        
        # Row 1: Question generation
        with gr.Row():
            generate_btn = gr.Button(
                "🎯 Generate Data Science Question",
                variant="primary",
                size="lg",
            )
        
        question_output = gr.Textbox(
            label="Generated Question (from Curriculum Fine-tuned Model)",
            lines=15,
            max_lines=15,
            interactive=True,
            show_copy_button=True,  # Built-in copy button
            placeholder="Click 'Generate' to create a challenging data science question...",
        )
        
        gr.Markdown("## 🧠 Step 2: Compare Model Answers")
        
        solve_btn = gr.Button(
            "⚡ Solve with Both Models",
            variant="secondary",
            size="lg",
        )
        
        # Row 2-3: Side-by-side answers
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔵 Base Model (Qwen3-1.7B)")
                base_answer = gr.Textbox(
                    label="Base Model Answer",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True,  # Built-in copy button
                    placeholder="Base model answer will appear here...",
                )
            
            with gr.Column():
                gr.Markdown("### 🟢 Trained Executor (executor_iter1)")
                executor_answer = gr.Textbox(
                    label="Trained Executor Answer",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True,  # Built-in copy button
                    placeholder="Trained executor answer will appear here...",
                )
        
        gr.Markdown("## 🏆 Step 3: Vote for the Better Answer")
        
        # Row 4: Voting
        with gr.Row():
            vote_base_btn = gr.Button(
                "👍 Base Model is Better",
                variant="secondary",
            )
            vote_executor_btn = gr.Button(
                "👍 Trained Executor is Better",
                variant="primary",
            )
            reset_btn = gr.Button(
                "🔄 Reset Votes",
                variant="stop",
            )
            generate_prompt_btn = gr.Button(
                "📝 Generate Voting Prompt",
                variant="secondary",
            )
        
        vote_display = gr.Textbox(
            label="Vote Tally",
            value=f"🔵 Base Model: {vote_counts['base']} votes | 🟢 Trained Executor: {vote_counts['executor']} votes",
            interactive=False,
        )
        
        # Voting prompt output (with copy button)
        voting_prompt_output = gr.Textbox(
            label="Voting Prompt (click copy button on right)",
            lines=10,
            max_lines=15,
            interactive=False,
            show_copy_button=True,  # Built-in copy button
            visible=False,
        )
        
        # Event handlers
        generate_btn.click(
            fn=generate_question,
            inputs=[curriculum_path],
            outputs=[question_output],
        )
        
        solve_btn.click(
            fn=solve_with_both_models,
            inputs=[question_output, base_path, executor_path],
            outputs=[base_answer, executor_answer],
        )
        
        vote_base_btn.click(
            fn=vote_base,
            inputs=[],
            outputs=[vote_display],
        )
        
        vote_executor_btn.click(
            fn=vote_executor,
            inputs=[],
            outputs=[vote_display],
        )
        
        reset_btn.click(
            fn=reset_votes,
            inputs=[],
            outputs=[vote_display],
        )
        
        def show_voting_prompt(question, base_ans, exec_ans):
            prompt = generate_voting_prompt(question, base_ans, exec_ans)
            if prompt.startswith("Please"):
                gr.Warning(prompt)
                return gr.update(visible=False, value="")
            gr.Info("📋 Voting prompt generated! Use the copy button to copy.")
            return gr.update(visible=True, value=prompt)
        
        generate_prompt_btn.click(
            fn=show_voting_prompt,
            inputs=[question_output, base_answer, executor_answer],
            outputs=[voting_prompt_output],
        )
        
        gr.Markdown("""
        ---
        ### 📊 About This Demo
        
        This demo is part of the **Agent0-inspired co-evolutionary training** framework:
        
        | Model | Description | Training Method |
        |-------|-------------|-----------------|
        | **Curriculum Agent** | Generates challenging questions at the executor's "frontier" difficulty | GRPO with uncertainty reward |
        | **Executor Agent** | Trained to solve these challenging questions | GRPO with self-consistency pseudo-labels |
        | **Base Model** | Original Qwen3-1.7B without Agent1 training | Qwen instruction tuning |
        
        The goal is to show that co-evolution improves the executor's problem-solving abilities!
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Agent1 Gradio Demo")
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model path",
    )
    parser.add_argument(
        "--curriculum_model",
        type=str,
        default=DEFAULT_CURRICULUM_MODEL,
        help="Curriculum model path",
    )
    parser.add_argument(
        "--executor_model",
        type=str,
        default=DEFAULT_EXECUTOR_MODEL,
        help="Trained executor model path",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the demo on",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Agent1 Demo")
    logger.info(f"   Base Model: {args.base_model}")
    logger.info(f"   Curriculum Model: {args.curriculum_model}")
    logger.info(f"   Executor Model: {args.executor_model}")
    logger.info("=" * 60)
    
    # Check if model paths exist
    logger.info("🔍 Checking model paths...")
    for name, path in [
        ("Curriculum", args.curriculum_model),
        ("Executor", args.executor_model),
    ]:
        if path.startswith("./") or path.startswith("/"):
            exists = Path(path).exists()
            logger.info(f"   {name}: {path} - {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        else:
            logger.info(f"   {name}: {path} - (HuggingFace model)")
    
    logger.info("📦 Creating demo interface...")
    demo = create_demo(
        base_model=args.base_model,
        curriculum_model=args.curriculum_model,
        executor_model=args.executor_model,
    )
    
    logger.info("🔄 Enabling queue for async processing...")
    demo.queue()
    
    logger.info(f"🌐 Launching on http://0.0.0.0:{args.port}")
    logger.info("   If you're on a remote server, use SSH port forwarding:")
    logger.info(f"   ssh -L {args.port}:localhost:{args.port} user@server")
    logger.info("   Or run with --share for a public URL")
    
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
