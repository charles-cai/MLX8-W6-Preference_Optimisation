from sklearn import base
import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from datasets import Dataset
import wandb
from tqdm import tqdm
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BitsAndBytesConfig

# Import our custom modules with relative imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prep_unified import load_and_prepare_datasets
from logger_utils import setup_logger

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B-Base")
DATASET_NAME_COMPARISON = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")

TRAIN_PERCENT = float(os.environ.get("TRAIN_PERCENT", "0.20"))
EVAL_PERCENT = float(os.environ.get("EVAL_PERCENT", "0.15"))

# Reward Model Configuration
REWARD_TRAIN_PERCENT = float(os.environ.get("REWARD_TRAIN_PERCENT", "0.20"))
REWARD_EVAL_PERCENT = float(os.environ.get("REWARD_EVAL_PERCENT", "0.05"))

# Model paths
SFT_OUTPUT_DIR_PREFIX = os.environ.get("SFT_OUTPUT_DIR_PREFIX", "./.data/sft_full_top_")
BASE_MODEL_PATH = f"{SFT_OUTPUT_DIR_PREFIX}0"  # SFT_TOP_0 model
REWARD_MODEL_DIR = os.environ.get("REWARD_MODEL_DIR", "./.data/reward_model")
POLICY_MODEL_DIR = os.environ.get("POLICY_MODEL_DIR", "./.data/policy_model")

# PPO Configuration
PPO_EPOCHS = int(os.environ.get("PPO_EPOCHS", "4"))
PPO_BATCH_SIZE = int(os.environ.get("PPO_BATCH_SIZE", "32"))
PPO_MINI_BATCH_SIZE = int(os.environ.get("PPO_MINI_BATCH_SIZE", "4"))
PPO_LR = float(os.environ.get("PPO_LR", "1e-5"))
PPO_STEPS = int(os.environ.get("PPO_STEPS", "1000"))

# QLoRA Configuration
USE_QLORA = os.environ.get("USE_QLORA", "false").lower() == "true"
LORA_R = int(os.environ.get("LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))

# Generation parameters
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

def get_bnb_config():
    """Get BitsAndBytesConfig for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def get_lora_config():
    """Get LoRA configuration for PPO training."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

class RewardModel:
    """Wrapper for the trained reward model - fixed to match actual TRL reward model."""
    
    def __init__(self, model_path):
        # Load reward model with appropriate configuration
        if USE_QLORA:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=get_bnb_config(),
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
    
    def get_reward(self, texts, max_length=512):
        """Get reward scores for a batch of texts - fixed to match TRL reward model output."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize all texts with padding to match training format
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'  # Match training padding
        )
        
        # Move to device
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            
            # Extract rewards - TRL reward models output logits, use mean as reward
            rewards = outputs.logits.mean(dim=-1)  # Mean across sequence length
        
        return rewards.cpu().detach()

def prepare_policy_dataset(dataset, tokenizer, num_samples=None):
    """
    Prepare dataset for PPO training - simplified format for TRL PPO.
    TRL PPOTrainer expects a simple list of query strings.
    """
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    def format_for_ppo(example):
        # For PPO, we just need the query text - TRL handles tokenization
        prompt = example['prompt'] + "\n\nSummary:"
        return {'query': prompt}
    
    formatted_dataset = dataset.map(format_for_ppo, remove_columns=dataset.column_names)
    
    # Convert to simple list of query strings as expected by TRL
    return [item['query'] for item in formatted_dataset]

def evaluate_policy(policy_model, tokenizer, reward_model, eval_dataset, num_samples=10):
    """
    Evaluate the policy model on a few samples.
    """
    policy_model.eval()
    eval_results = []
    
    eval_samples = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
    
    for sample in eval_samples:
        query = sample['query']
        
        # Generate response
        inputs = tokenizer(query, return_tensors='pt').to(policy_model.device)
        
        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response[len(query):]
        
        # Get reward
        reward = reward_model.get_reward(full_response).item()
        
        eval_results.append({
            'query': query,
            'generated_text': generated_text,
            'reward': reward,
            'full_response': full_response
        })
    
    policy_model.train()
    return eval_results

if __name__ == "__main__":
    logger = setup_logger("ppo_training")
    
    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlx8-week6-preference-optimisation"),
        entity=os.environ.get("WANDB_ENTITY", "charles-cai"),
        name=f"ppo_policy_training_{PPO_STEPS}_{'qlora' if USE_QLORA else 'full'}",
        config={
            "base_model": BASE_MODEL_PATH,
            "reward_model": REWARD_MODEL_DIR,
            "use_qlora": USE_QLORA,
            "lora_r": LORA_R if USE_QLORA else None,
            "lora_alpha": LORA_ALPHA if USE_QLORA else None,
            "lora_dropout": LORA_DROPOUT if USE_QLORA else None,
            "ppo_epochs": PPO_EPOCHS,
            "ppo_batch_size": PPO_BATCH_SIZE,
            "ppo_mini_batch_size": PPO_MINI_BATCH_SIZE,
            "ppo_lr": PPO_LR,
            "ppo_steps": PPO_STEPS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }
    )
    
    # --- RLHF Pipeline Step 3: PPO Policy Training ---
    # The PPO-trained policy model is the FINAL model in the RLHF pipeline.
    # It starts from the SFT model and gets optimized via PPO to maximize
    # rewards from the reward model, resulting in human-preferred responses.
    
    # --- 1. Load Reward Model ---
    logger.info(f"Loading reward model from: {REWARD_MODEL_DIR}")
    reward_model = RewardModel(REWARD_MODEL_DIR)
    
    # --- 2. Load Policy Model (starts as SFT model, becomes final RLHF model) ---
    logger.info(f"Loading policy model from: {BASE_MODEL_PATH}")
    logger.info(f"Using QLoRA: {USE_QLORA}")
    
    if USE_QLORA:
        # Load BASE model with quantization first
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=get_bnb_config(),
            device_map="auto"
        )
        
        # Apply LoRA to base model
        lora_config = get_lora_config()
        base_model = get_peft_model(base_model, lora_config)
        
        # THEN wrap with value head
        policy_model = AutoModelForCausalLMWithValueHead(base_model)
        
        # Load reference model with quantization (no LoRA)
        ref_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=get_bnb_config(),
            device_map="auto"
        )
        
        logger.info(f"QLoRA configuration: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
        logger.info(f"Trainable parameters: {base_model.print_trainable_parameters()}")
    else:
        # Load model in full precision and wrap with value head
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        policy_model = AutoModelForCausalLMWithValueHead(base_model)
        
        # Load reference model (for KL divergence)
        ref_model = create_reference_model(policy_model)

    # # critic: share base weights but attach a scalar value head --> value_model for the PPOTrainer
    # critic = AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True)
    # critic.v_head = torch.nn.Linear(critic.config.hidden_size, 1, bias=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 3. Prepare Dataset ---
    logger.info("Preparing dataset for PPO training...")
    
    # Use different data split for PPO - avoid overlap with reward model training
    train_dataset, eval_dataset, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME_COMPARISON,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT,
        train_percent_reward=REWARD_TRAIN_PERCENT,
        eval_precent_reward=REWARD_EVAL_PERCENT,
    )
    
    # Prepare datasets as simple lists of query strings
    ppo_train_queries = prepare_policy_dataset(train_dataset, tokenizer, num_samples=PPO_STEPS)
    ppo_eval_queries = prepare_policy_dataset(eval_dataset, tokenizer, num_samples=100)
    
    logger.success(f"PPO datasets prepared:")
    logger.success(f"Train queries: {len(ppo_train_queries)}")
    logger.success(f"Eval queries: {len(ppo_eval_queries)}")
    
    # --- 4. Configure PPO ---
    logger.info("Setting up PPO configuration...")
    
    ppo_config = PPOConfig(
        learning_rate=PPO_LR,
        batch_size=PPO_BATCH_SIZE,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        #ppo_epochs=PPO_EPOCHS,  # Uncomment this
        seed=42,
        #stop_token_id=tokenizer.eos_token_id,
    )
    
    # --- 5. Create PPO Trainer ---
    logger.info("Creating PPO trainer...")    

    # Ensure the wrapped model has proper generation_config access
    if not hasattr(policy_model, 'generation_config'):
        policy_model.generation_config = policy_model.pretrained_model.generation_config
    
    # Ensure stop_token_id is set in generation_config
    if policy_model.generation_config.eos_token_id is None:
        policy_model.generation_config.eos_token_id = tokenizer.eos_token_id
    
    # Create reward function that matches TRL expectations
    def reward_function(samples):
        """Reward function that takes generated samples and returns rewards."""
        rewards = reward_model.get_reward(samples)
        # Convert to list of floats
        return [float(r.item()) for r in rewards]
    
    # Create PPO trainer with simplified approach
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=policy_model,
        value_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model.model,  # Use the reward model's underlying model
        processing_class=tokenizer,
        train_dataset=ppo_train_queries,  # Simple list of query strings
        
        #reward_fn=reward_function,  # Use reward function instead of model
    )

    # --- 6. Training Loop ---
    logger.info("--- Starting PPO Training ---")
    
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        if epoch >= PPO_STEPS:
            break
        
        # Extract queries and input_ids from batch
        if isinstance(batch, dict):
            queries = batch.get('query', [])
            query_tensors = batch.get('input_ids', [])
        else:
            # Handle list format
            queries = [item['query'] for item in batch]
            query_tensors = [item['input_ids'] for item in batch]
        
        # Ensure query_tensors are properly formatted
        if isinstance(query_tensors[0], torch.Tensor):
            query_tensors = [t.to(policy_model.device) for t in query_tensors]
        else:
            query_tensors = [torch.tensor(t).to(policy_model.device) for t in query_tensors]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )
        
        # Decode responses
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        # Create full texts for reward model
        full_texts = [q + r for q, r in zip(queries, responses)]
        
        # Get rewards - ensure single tensor per response
        rewards = reward_model.get_reward(full_texts)
        
        # Convert to list of scalar tensors
        rewards = [torch.tensor(r.item()).to(policy_model.device) for r in rewards]
        
        # PPO step
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        except Exception as e:
            logger.error(f"PPO step failed: {e}")
            logger.info("Skipping this batch and continuing...")
            continue
        
        # Log statistics
        if epoch % 10 == 0:
            mean_reward = np.mean([r.item() for r in rewards])
            logger.info(f"Epoch {epoch}: Mean reward = {mean_reward:.4f}")
            
            # Log to wandb with error handling
            try:
                wandb.log({
                    'epoch': epoch,
                    'mean_reward': mean_reward,
                    'std_reward': np.std([r.item() for r in rewards]),
                    'max_reward': np.max([r.item() for r in rewards]),
                    'min_reward': np.min([r.item() for r in rewards]),
                    **stats
                })
            except Exception as e:
                logger.warning(f"Wandb logging failed: {e}")
        
        # Periodic evaluation
        if epoch % 100 == 0 and epoch > 0:
            logger.info(f"--- Evaluation at epoch {epoch} ---")
            eval_results = evaluate_policy(
                policy_model, tokenizer, reward_model, ppo_eval_dataset, num_samples=5
            )
            
            for i, result in enumerate(eval_results):
                logger.info(f"Eval {i+1}:")
                logger.info(f"Query: {result['query'][:100]}...")
                logger.success(f"Generated: {result['generated_text'][:100]}...")
                logger.info(f"Reward: {result['reward']:.4f}")
                logger.info("---")
            
            # Log evaluation results
            eval_rewards = [r['reward'] for r in eval_results]
            wandb.log({
                f'eval_epoch_{epoch}_mean_reward': np.mean(eval_rewards),
                f'eval_epoch_{epoch}_std_reward': np.std(eval_rewards),
            })
    
    # --- 7. Save Policy Model (This is the FINAL RLHF model!) ---
    logger.info(f"Saving FINAL RLHF PPO-trained policy model to {POLICY_MODEL_DIR}")
    os.makedirs(POLICY_MODEL_DIR, exist_ok=True)
    
    if USE_QLORA:
        # Save LoRA adapters
        policy_model.save_pretrained(POLICY_MODEL_DIR)
        
        # Also save the merged model for easier inference
        merged_model_dir = f"{POLICY_MODEL_DIR}_merged"
        logger.info(f"Merging and saving full model to {merged_model_dir}")
        
        # Merge LoRA weights with base model
        merged_model = policy_model.merge_and_unload()
        merged_model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)
        
        logger.success(f"QLoRA adapters saved to: {POLICY_MODEL_DIR}")
        logger.success(f"Merged model saved to: {merged_model_dir}")
    else:
        policy_model.save_pretrained(POLICY_MODEL_DIR)
        
    tokenizer.save_pretrained(POLICY_MODEL_DIR)
    
    # --- 8. Final Evaluation ---
    logger.info("--- Final Policy Evaluation ---")
    final_eval_results = evaluate_policy(
        policy_model, tokenizer, reward_model, ppo_eval_dataset, num_samples=20
    )
    
    logger.success("--- Final Evaluation Results ---")
    final_rewards = [r['reward'] for r in final_eval_results]
    logger.success(f"Mean reward: {np.mean(final_rewards):.4f}")
    logger.success(f"Std reward: {np.std(final_rewards):.4f}")
    logger.success(f"Max reward: {np.max(final_rewards):.4f}")
    logger.success(f"Min reward: {np.min(final_rewards):.4f}")
    
    # Log final results
    wandb.log({
        'final_mean_reward': np.mean(final_rewards),
        'final_std_reward': np.std(final_rewards),
        'final_max_reward': np.max(final_rewards),
        'final_min_reward': np.min(final_rewards),
    })
    
    # Log sample final results as table
    sample_table = wandb.Table(columns=["query", "generated_text", "reward"])
    for result in final_eval_results[:10]:
        sample_table.add_data(
            result['query'][:200],
            result['generated_text'][:200],
            result['reward']
        )
    wandb.log({"final_sample_results": sample_table})
    
    # --- 9. Save training info ---
    training_info = {
        'base_model_path': BASE_MODEL_PATH,
        'reward_model_path': REWARD_MODEL_DIR,
        'use_qlora': USE_QLORA,
        'lora_config': lora_config.to_dict() if USE_QLORA else None,
        'ppo_config': ppo_config.to_dict(),
        'training_steps': PPO_STEPS,
        'final_mean_reward': np.mean(final_rewards),
        'dataset_name': DATASET_NAME_COMPARISON,
        'train_samples': len(ppo_train_dataset),
        'eval_samples': len(ppo_eval_dataset),
    }
    
    torch.save(training_info, os.path.join(POLICY_MODEL_DIR, 'training_info.pt'))
    
    logger.success(f"FINAL RLHF PPO policy model training completed and saved to {POLICY_MODEL_DIR}")
    logger.success("This PPO-trained policy model is the end result of the complete RLHF pipeline:")
    logger.success("  1. SFT → 2. Reward Model → 3. PPO Policy Model ✓")
    
    # Finish wandb run
    wandb.finish()