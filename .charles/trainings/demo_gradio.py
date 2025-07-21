import torch
import os
import random
import gradio as gr
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prep_unified import load_and_prepare_datasets
from logger_utils import setup_logger

# Load environment variables
load_dotenv()

# Configuration
DATASET_NAME = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

# Model paths
SFT_OUTPUT_DIR_PREFIX = os.environ.get("SFT_OUTPUT_DIR_PREFIX", "./.data/sft_full_top_")
DPO_MODEL_PATH = os.environ.get("EVAL_MODEL_PATH", "./.data/dpo_qlora_merged")
PPO_MODEL_PATH = os.environ.get("POLICY_MODEL_DIR", "./.data/policy_model")

class ModelShowcase:
    def __init__(self):
        self.logger = setup_logger("gradio_eval")
        self.test_dataset = None
        self.models = {}
        self.tokenizers = {}
        self.model_info = {
            "sft_full": {"name": "SFT Full Model", "path": SFT_OUTPUT_DIR_PREFIX + "0", "params": "Full fine-tuning (top_k=0)"},
            "sft_top4": {"name": "SFT TOP-4 Model", "path": SFT_OUTPUT_DIR_PREFIX + "4", "params": "LoRA fine-tuning (top_k=4)"},
            "dpo": {"name": "DPO Model", "path": DPO_MODEL_PATH, "params": "Direct Preference Optimization"},
            "ppo": {"name": "PPO Model", "path": PPO_MODEL_PATH, "params": "Proximal Policy Optimization"}
        }
        self.load_dataset()
        self.load_models()
    
    def load_dataset(self):
        """Load test dataset for evaluation"""
        self.logger.info("Loading test dataset...")
        try:
            _, _, self.test_dataset = load_and_prepare_datasets(dataset_name=DATASET_NAME)
            self.logger.info(f"Loaded {len(self.test_dataset)} test samples")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            self.test_dataset = []
    
    def load_models(self):
        """Load all available models"""
        for model_key, info in self.model_info.items():
            try:
                if os.path.exists(info["path"]):
                    self.logger.info(f"Loading {info['name']} from {info['path']}")
                    model = AutoModelForCausalLM.from_pretrained(
                        info["path"],
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(info["path"])
                    model.eval()
                    
                    self.models[model_key] = model
                    self.tokenizers[model_key] = tokenizer
                    self.logger.info(f"Successfully loaded {info['name']}")
                else:
                    self.logger.warning(f"Model path not found: {info['path']}")
            except Exception as e:
                self.logger.error(f"Failed to load {info['name']}: {e}")
    
    def generate_response(self, model_key, prompt):
        """Generate response from a specific model"""
        if model_key not in self.models:
            return f"Model {model_key} not available"
        
        try:
            model = self.models[model_key]
            tokenizer = self.tokenizers[model_key]
            
            prompt_text = prompt + "\n\nSummary:\n"
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P
                )
            
            full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_summary = full_output.split("Summary:")[-1].strip()
            return generated_summary
            
        except Exception as e:
            self.logger.error(f"Error generating response with {model_key}: {e}")
            return f"Error: {str(e)}"
    
    def get_random_sample(self):
        """Get a random sample from test dataset and generate responses"""
        if not self.test_dataset:
            return ["No dataset loaded"] * 7
        
        sample = random.choice(self.test_dataset)
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        
        # Generate responses from all models
        responses = []
        for model_key in ["sft_full", "sft_top4", "dpo", "ppo"]:
            if model_key in self.models:
                response = self.generate_response(model_key, prompt)
                responses.append(response)
            else:
                responses.append("Model not available")
        
        return responses + [prompt, chosen, rejected]

    def reset_outputs(self):
        """Reset all text outputs to empty"""
        return [""] * 7

# Initialize the showcase
showcase = ModelShowcase()

def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Model Comparison Showcase", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Model Comparison Showcase")
        gr.Markdown("Compare responses from SFT Full, SFT TOP-4, DPO, and PPO models")
        
        # Model results in 4 columns
        with gr.Row():
            # SFT Full Model
            with gr.Column():
                gr.Markdown("### 🔵 SFT Full Model")
                gr.Markdown("**Parameters:** Full fine-tuning (top_k=0)")
                sft_full_output = gr.Textbox(
                    label="Generated Summary",
                    lines=15,
                    interactive=False,
                    placeholder="Model response will appear here..."
                )
            
            # SFT TOP-4 Model  
            with gr.Column():
                gr.Markdown("### 🟡 SFT TOP-4 Model")
                gr.Markdown("**Parameters:** LoRA fine-tuning (top_k=4)")
                sft_top4_output = gr.Textbox(
                    label="Generated Summary",
                    lines=15,
                    interactive=False,
                    placeholder="Model response will appear here..."
                )
            
            # DPO Model
            with gr.Column():
                gr.Markdown("### 🟢 DPO Model")
                gr.Markdown("**Parameters:** Direct Preference Optimization")
                dpo_output = gr.Textbox(
                    label="Generated Summary",
                    lines=15,
                    interactive=False,
                    placeholder="Model response will appear here..."
                )
            
            # PPO Model
            with gr.Column():
                gr.Markdown("### 🟣 PPO Model")
                gr.Markdown("**Parameters:** Proximal Policy Optimization")
                ppo_output = gr.Textbox(
                    label="Generated Summary",
                    lines=15,
                    interactive=False,
                    placeholder="Model response will appear here..."
                )
        
        # Separator
        gr.Markdown("---")
        
        # Input areas
        chosen_input = gr.Textbox(
            label="✅ Chosen (Reference)",
            lines=3,
            interactive=False,
            elem_classes=["green-text"]
        )
        
        prompt_input = gr.Textbox(
            label="📝 Prompt",
            lines=5,
            interactive=False,
            placeholder="Click the button below to load a random test sample..."
        )
        
        rejected_input = gr.Textbox(
            label="❌ Rejected",
            lines=3,
            interactive=False,
            elem_classes=["red-text"]
        )
        
        # Control buttons row with 30-30-30 split
        with gr.Row():
            with gr.Column(scale=30):
                random_btn = gr.Button(
                    "🎲 Pick Random Sample",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=30):
                data_source_dropdown = gr.Dropdown(
                    choices=["Test Dataset"],
                    value="Test Dataset",
                    label="Data Source",
                    interactive=True
                )
            
            with gr.Column(scale=30):
                reset_btn = gr.Button(
                    "🔄 Reset",
                    variant="secondary",
                    size="lg"
                )
        
        # Button click handlers
        random_btn.click(
            fn=showcase.get_random_sample,
            inputs=[],
            outputs=[
                sft_full_output,
                sft_top4_output, 
                dpo_output,
                ppo_output,
                prompt_input,
                chosen_input,
                rejected_input
            ]
        )
        
        reset_btn.click(
            fn=showcase.reset_outputs,
            inputs=[],
            outputs=[
                sft_full_output,
                sft_top4_output, 
                dpo_output,
                ppo_output,
                prompt_input,
                chosen_input,
                rejected_input
            ]
        )

        # Custom CSS for colored text
        demo.load(js="""
        function() {
            const style = document.createElement('style');
            style.textContent = `
                .green-text textarea {
                    border-color: #22c55e !important;
                    background-color: #f0fdf4 !important;
                }
                .red-text textarea {
                    border-color: #ef4444 !important;
                    background-color: #fef2f2 !important;
                }
            `;
            document.head.appendChild(style);
        }
        """)
    
    return demo

def print_model_status():
    """Print loaded model status for debugging"""
    print(f"\n=== Model Loading Status ===")
    print(f"Models successfully loaded: {len(showcase.models)}")
    print(f"Available model keys: {list(showcase.models.keys())}")
    
    for model_key, info in showcase.model_info.items():
        status = "✅ LOADED" if model_key in showcase.models else "❌ FAILED"
        print(f"{status} {info['name']}: {info['path']}")
    print("=" * 30)

# Call this to see the status
print_model_status()

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8086,
        share=False,
        show_error=True
    )
