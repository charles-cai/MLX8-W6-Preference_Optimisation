# TRAINING PPO WITHOUT USING QLORA 

import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import Dataset
import wandb
from tqdm import tqdm
import numpy as np

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

# Generation parameters
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

class RewardModel:
    """Wrapper for the trained reward model - fixed to match actual TRL reward model."""
    
    def __init__(self, model_path):
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
            
            # Extract rewards - TRL reward models typically output a single reward score
            # Use the last token's logits as the reward score
            rewards = outputs.logits[:, -1, :].mean(dim=-1)
        
        return rewards.cpu().detach()

def prepare_policy_dataset(dataset, tokenizer, num_samples=None):
    """
    Prepare dataset for PPO training - updated for current TRL API.
    """
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    def format_for_ppo(example):
        # For PPO, we need the query text
        prompt = example['prompt'] + "\n\nSummary:"
        
        # Tokenize the query for TRL compatibility
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=512 - MAX_NEW_TOKENS,
            padding=False,
            return_tensors=None
        )
        
        return {
            "query": prompt,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }
    
    formatted_dataset = dataset.map(format_for_ppo, remove_columns=dataset.column_names)
    return formatted_dataset

def ppo_data_collator(batch):
    """
    Custom data collator for PPO that handles mixed tensor/string data.
    """
    # Separate queries (strings) from tokenized data (tensors)
    queries = [item["query"] for item in batch]
    
    # Extract and pad only the tensor fields
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    
    # Pad sequences manually
    max_length = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for ids, mask in zip(input_ids, attention_masks):
        padding_length = max_length - len(ids)
        
        # Pad input_ids with pad_token_id
        padded_ids = ids + [tokenizer.pad_token_id] * padding_length
        padded_input_ids.append(padded_ids)
        
        # Pad attention_mask with 0s
        padded_mask = mask + [0] * padding_length
        padded_attention_masks.append(padded_mask)
    
    return {
        "query": queries,
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(padded_attention_masks)
    }

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
        name=f"ppo_policy_training_{PPO_STEPS}",
        config={
            "base_model": BASE_MODEL_PATH,
            "reward_model": REWARD_MODEL_DIR,
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
    
    # Verify model paths exist
    if not os.path.exists(BASE_MODEL_PATH):
        logger.error(f"SFT model not found at {BASE_MODEL_PATH}")
        logger.error("Please run train_sft.py first!")
        exit(1)
    
    if not os.path.exists(REWARD_MODEL_DIR):
        logger.error(f"Reward model not found at {REWARD_MODEL_DIR}")
        logger.error("Please run train_reward.py first!")
        exit(1)
    
    # --- RLHF Pipeline Step 3: PPO Policy Training ---
    # The PPO-trained policy model is the FINAL model in the RLHF pipeline.
    # It starts from the SFT model and gets optimized via PPO to maximize
    # rewards from the reward model, resulting in human-preferred responses.
    
    # --- 1. Load Reward Model ---
    logger.info(f"Loading reward model from: {REWARD_MODEL_DIR}")
    reward_model = RewardModel(REWARD_MODEL_DIR)
    
    # --- 2. Load Policy Model (starts as SFT model, becomes final RLHF model) ---
    logger.info(f"Loading policy model from: {BASE_MODEL_PATH}")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load tokenizer first for dataset preparation and generation
    logger.info(f"Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    policy_model.generation_config = base_model.generation_config

    # Load reference model (for KL divergence)
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Get device from policy model for later use
    policy_device = next(policy_model.parameters()).device
    
    # --- 3. Prepare Dataset ---
    logger.info("Preparing dataset for PPO training...")
    
    # Use different data split for PPO - avoid overlap with reward model training
    train_dataset, eval_dataset, _ = load_and_prepare_datasets(
        dataset_name=DATASET_NAME_COMPARISON,
        train_on_percent=TRAIN_PERCENT,
        eval_on_percent=EVAL_PERCENT,
        train_percent_reward=REWARD_TRAIN_PERCENT,
        eval_percent_reward=REWARD_EVAL_PERCENT,
    )
    
    # Format for PPO with tokenizer
    ppo_train_dataset = prepare_policy_dataset(train_dataset, tokenizer, num_samples=PPO_STEPS)
    ppo_eval_dataset = prepare_policy_dataset(eval_dataset, tokenizer, num_samples=100)
    
    logger.success(f"PPO datasets prepared:")
    logger.success(f"Train samples: {len(ppo_train_dataset)}")
    logger.success(f"Eval samples: {len(ppo_eval_dataset)}")
    
    # --- 4. Configure PPO ---
    logger.info("Setting up PPO configuration...")
    
    ppo_config = PPOConfig(
        learning_rate=PPO_LR,
        batch_size=PPO_BATCH_SIZE,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        num_ppo_epochs=PPO_EPOCHS,
        seed=42,
    )
    
    # --- 5. Create PPO Trainer ---
    logger.info("Creating PPO trainer...")
    
    # Simplified PPOTrainer without reward model (we'll handle rewards manually)
    ppo_trainer = PPOTrainer(       
        args=ppo_config,  # Use args instead of config
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        model=policy_model, # actor / policy: the trainable copy (SFT initialized)
        ref_model=ref_model, # a frozen copy of the policy model for KL-divergence anchor
        reward_model=reward_model.model,  # a separately-trained, frozen scalar scorer
        value_model=base_model,  # Use policy model as value model
        train_dataset=ppo_train_dataset,  # Simple list of query strings
        data_collator=ppo_data_collator,
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
        queries = batch["query"]
        query_tensors = batch["input_ids"].to(policy_device)
        
        # Generate responses
        response_tensors = []
        with torch.no_grad():
            for i in range(query_tensors.size(0)):
                query_tensor = query_tensors[i:i+1]
                
                outputs = policy_model.generate(
                    input_ids=query_tensor,
                    attention_mask=torch.ones_like(query_tensor),
                    **generation_kwargs
                )
                
                # Extract only the generated part
                response = outputs[0][query_tensor.size(1):]
                response_tensors.append(response)
        
        # Decode responses for reward calculation
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        full_texts = [q + r for q, r in zip(queries, responses)]
        
        # Get rewards using our custom reward model
        rewards = reward_model.get_reward(full_texts)
        rewards = [torch.tensor(r.item()).to(policy_device) for r in rewards]
        
        # Convert query_tensors to list for TRL compatibility
        query_tensors_list = [query_tensors[i] for i in range(query_tensors.size(0))]
        
        # Manual PPO-style training implementation
        try:
            logger.debug("Running manual PPO training step")
            
            # Compute advantages and train
            all_logprobs = []
            all_ref_logprobs = []
            
            policy_model.train()
            
            for i, (query_tensor, response_tensor) in enumerate(zip(query_tensors_list, response_tensors)):
                full_tensor = torch.cat([query_tensor, response_tensor])
                
                # Get logprobs from policy model - handle tuple/tensor outputs
                outputs = policy_model(full_tensor.unsqueeze(0))
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, tuple):
                    logits = outputs[0]  # First element is logits
                else:
                    logits = outputs
                
                # Get logprobs for the response tokens
                response_start = query_tensor.size(0)
                response_logits = logits[0, response_start-1:-1]  # Shift by 1 for next token prediction
                logprobs = torch.nn.functional.log_softmax(response_logits, dim=-1)
                
                # Ensure response_tensor has correct shape for gathering
                if response_tensor.dim() == 1:
                    response_indices = response_tensor.unsqueeze(1)
                else:
                    response_indices = response_tensor
                
                # Gather logprobs for actual response tokens
                if response_indices.size(0) <= logprobs.size(0):
                    gathered_logprobs = logprobs[:response_indices.size(0)].gather(1, response_indices)
                    response_logprobs = gathered_logprobs.squeeze().sum()
                else:
                    # Handle case where response is longer than available logits
                    response_logprobs = torch.tensor(0.0, device=policy_device)
                
                all_logprobs.append(response_logprobs)
                
                # Get ref logprobs
                with torch.no_grad():
                    ref_outputs = ref_model(full_tensor.unsqueeze(0))
                    if hasattr(ref_outputs, 'logits'):
                        ref_logits = ref_outputs.logits
                    elif isinstance(ref_outputs, tuple):
                        ref_logits = ref_outputs[0]
                    else:
                        ref_logits = ref_outputs
                    
                    ref_response_logits = ref_logits[0, response_start-1:-1]
                    ref_logprobs = torch.nn.functional.log_softmax(ref_response_logits, dim=-1)
                    
                    if response_indices.size(0) <= ref_logprobs.size(0):
                        ref_gathered_logprobs = ref_logprobs[:response_indices.size(0)].gather(1, response_indices)
                        ref_response_logprobs = ref_gathered_logprobs.squeeze().sum()
                    else:
                        ref_response_logprobs = torch.tensor(0.0, device=policy_device)
                    
                    all_ref_logprobs.append(ref_response_logprobs)
            
            # Compute policy loss with KL penalty
            policy_loss = torch.tensor(0.0, device=policy_device)
            kl_div = torch.tensor(0.0, device=policy_device)
            
            for i, (logprob, ref_logprob, reward) in enumerate(zip(all_logprobs, all_ref_logprobs, rewards)):
                kl_penalty = 0.1 * (logprob - ref_logprob)
                advantage = reward - kl_penalty
                policy_loss -= logprob * advantage.detach()
                kl_div += (logprob - ref_logprob)
            
            policy_loss = policy_loss / len(all_logprobs)
            kl_div = kl_div / len(all_logprobs)
            
            # Backward pass with optimizer from trainer
            if hasattr(ppo_trainer, 'optimizer'):
                optimizer = ppo_trainer.optimizer
            else:
                # Create optimizer if trainer doesn't have one
                optimizer = torch.optim.AdamW(policy_model.parameters(), lr=PPO_LR)
            
            policy_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            stats = {
                "ppo/policy_loss": policy_loss.item(),
                "ppo/mean_scores": torch.stack(rewards).mean().item(),
                "ppo/kl_div": kl_div.item(),
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Simple fallback - just log the error and continue
            stats = {
                "ppo/policy_loss": 0.0,
                "ppo/mean_scores": torch.stack(rewards).mean().item(),
                "error": str(e)
            }
        
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
            
            # Prepare eval samples
            eval_samples = ppo_eval_dataset.select(range(min(5, len(ppo_eval_dataset))))
            eval_results = []
            
            policy_model.eval()
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
    
    policy_model.save_pretrained(POLICY_MODEL_DIR)
    tokenizer.save_pretrained(POLICY_MODEL_DIR)
    
    # --- 8. Final Evaluation ---
    logger.info("--- Final Policy Evaluation ---")
    
    # Prepare final eval samples
    final_eval_samples = ppo_eval_dataset.select(range(min(20, len(ppo_eval_dataset))))
    final_eval_results = []
    
    policy_model.eval()
    for sample in final_eval_samples:
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
        
        final_eval_results.append({
            'query': query,
            'generated_text': generated_text,
            'reward': reward,
            'full_response': full_response
        })
    
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