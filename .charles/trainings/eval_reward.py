import torch
import os
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our data preparation function
from data_prep_unified import load_and_prepare_datasets
from logger_utils import setup_logger

# Load environment variables
load_dotenv()

# --- Configuration ---
REWARD_MODEL_DIR = os.environ.get("REWARD_MODEL_DIR", "./.data/reward_model")
DATASET_NAME = os.environ.get("DATASET_NAME_COMPARISON", "CarperAI/openai_summarize_comparisons")
NUM_TEST_SAMPLES = int(os.environ.get("NUM_TEST_SAMPLES", "200"))
NUM_SAMPLES_TO_SHOW = int(os.environ.get("NUM_SAMPLES_TO_SHOW", "20"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "512"))

class RewardModelEvaluator:
    def __init__(self, model_path, tokenizer_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        
    def get_reward_score(self, text, max_length=None):
        """
        Get reward score for a given text.
        Updated to match training tokenization parameters.
        """
        if max_length is None:
            max_length = MAX_SEQ_LENGTH
            
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length'  # Match training padding
        )
        
        # Move to device
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = self.model(**tokens)
            
            # For TRL-trained reward models, extract reward similar to training evaluation
            # Use logits mean as reward score (matching training evaluation approach)
            reward = outputs.logits.mean()
        
        return reward.cpu().item()
    
    def evaluate_preferences(self, test_dataset, num_samples=None):
        """
        Evaluate reward model on preference pairs.
        """
        if num_samples and num_samples < len(test_dataset):
            test_dataset = test_dataset.select(range(num_samples))
        
        results = {
            'chosen_rewards': [],
            'rejected_rewards': [],
            'reward_differences': [],
            'preference_accuracies': [],
            'samples': []
        }
        
        logger.info(f"Evaluating on {len(test_dataset)} preference pairs...")
        
        for i, sample in enumerate(tqdm(test_dataset)):
            prompt = sample['prompt']
            chosen = sample['chosen']
            rejected = sample['rejected']
            
            # Create full texts
            chosen_text = prompt + "\n\nSummary:\n" + chosen
            rejected_text = prompt + "\n\nSummary:\n" + rejected
            
            # Get reward scores
            chosen_reward = self.get_reward_score(chosen_text)
            rejected_reward = self.get_reward_score(rejected_text)
            
            reward_diff = chosen_reward - rejected_reward
            preference_correct = chosen_reward > rejected_reward
            
            results['chosen_rewards'].append(chosen_reward)
            results['rejected_rewards'].append(rejected_reward)
            results['reward_differences'].append(reward_diff)
            results['preference_accuracies'].append(preference_correct)
            
            # Store samples for detailed analysis
            if i < NUM_SAMPLES_TO_SHOW:
                results['samples'].append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'chosen_reward': chosen_reward,
                    'rejected_reward': rejected_reward,
                    'reward_difference': reward_diff,
                    'preference_correct': preference_correct
                })
        
        return results
    
    def compute_metrics(self, results):
        """
        Compute evaluation metrics from results.
        """
        chosen_rewards = np.array(results['chosen_rewards'])
        rejected_rewards = np.array(results['rejected_rewards'])
        reward_differences = np.array(results['reward_differences'])
        preference_accuracies = np.array(results['preference_accuracies'])
        
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(chosen_rewards, rejected_rewards)
        spearman_corr, spearman_p = spearmanr(chosen_rewards, rejected_rewards)
        
        metrics = {
            # Preference accuracy
            'preference_accuracy': np.mean(preference_accuracies),
            'total_comparisons': len(preference_accuracies),
            'correct_preferences': np.sum(preference_accuracies),
            
            # Reward statistics
            'chosen_reward_mean': np.mean(chosen_rewards),
            'chosen_reward_std': np.std(chosen_rewards),
            'chosen_reward_min': np.min(chosen_rewards),
            'chosen_reward_max': np.max(chosen_rewards),
            'rejected_reward_mean': np.mean(rejected_rewards),
            'rejected_reward_std': np.std(rejected_rewards),
            'rejected_reward_min': np.min(rejected_rewards),
            'rejected_reward_max': np.max(rejected_rewards),
            
            # Reward difference statistics
            'reward_diff_mean': np.mean(reward_differences),
            'reward_diff_std': np.std(reward_differences),
            'reward_diff_median': np.median(reward_differences),
            'reward_diff_min': np.min(reward_differences),
            'reward_diff_max': np.max(reward_differences),
            
            # Separation metrics
            'positive_differences': np.sum(reward_differences > 0),
            'negative_differences': np.sum(reward_differences < 0),
            'zero_differences': np.sum(reward_differences == 0),
            'positive_diff_ratio': np.mean(reward_differences > 0),
            
            # Correlation metrics
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            
            # Distribution percentiles
            'chosen_reward_25th': np.percentile(chosen_rewards, 25),
            'chosen_reward_75th': np.percentile(chosen_rewards, 75),
            'rejected_reward_25th': np.percentile(rejected_rewards, 25),
            'rejected_reward_75th': np.percentile(rejected_rewards, 75),
            'reward_diff_25th': np.percentile(reward_differences, 25),
            'reward_diff_75th': np.percentile(reward_differences, 75),
        }
        
        return metrics
    
    def create_visualizations(self, results):
        """
        Create visualizations for reward model evaluation.
        """
        chosen_rewards = np.array(results['chosen_rewards'])
        rejected_rewards = np.array(results['rejected_rewards'])
        reward_differences = np.array(results['reward_differences'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reward distribution comparison
        axes[0, 0].hist(chosen_rewards, alpha=0.7, label='Chosen', bins=50, color='green')
        axes[0, 0].hist(rejected_rewards, alpha=0.7, label='Rejected', bins=50, color='red')
        axes[0, 0].set_xlabel('Reward Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution: Chosen vs Rejected')
        axes[0, 0].legend()
        
        # 2. Scatter plot of chosen vs rejected rewards
        axes[0, 1].scatter(chosen_rewards, rejected_rewards, alpha=0.6)
        axes[0, 1].plot([min(chosen_rewards), max(chosen_rewards)], 
                       [min(chosen_rewards), max(chosen_rewards)], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Chosen Reward')
        axes[0, 1].set_ylabel('Rejected Reward')
        axes[0, 1].set_title('Chosen vs Rejected Rewards')
        
        # 3. Reward difference distribution
        axes[1, 0].hist(reward_differences, bins=50, alpha=0.7, color='blue')
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Reward Difference (Chosen - Rejected)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reward Difference Distribution')
        
        # 4. Preference accuracy over batches
        batch_size = 50
        batch_accuracies = []
        for i in range(0, len(results['preference_accuracies']), batch_size):
            batch = results['preference_accuracies'][i:i+batch_size]
            batch_accuracies.append(np.mean(batch))
        
        axes[1, 1].plot(range(len(batch_accuracies)), batch_accuracies, 'o-')
        axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.8, label='Random')
        axes[1, 1].set_xlabel('Batch Index')
        axes[1, 1].set_ylabel('Preference Accuracy')
        axes[1, 1].set_title(f'Preference Accuracy Over Batches (size={batch_size})')
        axes[1, 1].legend()
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    logger = setup_logger("reward_model_evaluation")
    
    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlx8-week6-preference-optimisation"),
        entity=os.environ.get("WANDB_ENTITY", "charles-cai"),
        name=f"eval_reward_model_{NUM_TEST_SAMPLES}",
        config={
            "model_path": REWARD_MODEL_DIR,
            "dataset_name": DATASET_NAME,
            "num_test_samples": NUM_TEST_SAMPLES,
            "evaluation_type": "reward_model"
        }
    )
    
    # --- 1. Load Test Dataset ---
    logger.info("Loading test dataset...")
    _, _, test_dataset = load_and_prepare_datasets(dataset_name=DATASET_NAME)
    
    # --- 2. Initialize Evaluator ---
    logger.info(f"Loading reward model from: {REWARD_MODEL_DIR}")
    evaluator = RewardModelEvaluator(REWARD_MODEL_DIR)
    
    # --- 3. Evaluate Preferences ---
    logger.info("--- Starting Reward Model Evaluation ---")
    results = evaluator.evaluate_preferences(test_dataset, NUM_TEST_SAMPLES)
    
    # --- 4. Compute Metrics ---
    logger.info("Computing evaluation metrics...")
    metrics = evaluator.compute_metrics(results)
    
    # --- 5. Log Results ---
    logger.success("--- REWARD MODEL EVALUATION RESULTS ---")
    logger.success(f"Preference Accuracy: {metrics['preference_accuracy']:.4f}")
    logger.success(f"Correct Preferences: {metrics['correct_preferences']}/{metrics['total_comparisons']}")
    logger.success(f"Chosen Reward Mean: {metrics['chosen_reward_mean']:.4f} ± {metrics['chosen_reward_std']:.4f}")
    logger.success(f"Rejected Reward Mean: {metrics['rejected_reward_mean']:.4f} ± {metrics['rejected_reward_std']:.4f}")
    logger.success(f"Reward Difference Mean: {metrics['reward_diff_mean']:.4f} ± {metrics['reward_diff_std']:.4f}")
    
    # Enhanced wandb logging with organized metrics
    wandb.log({
        # Main performance metrics
        "accuracy/preference_accuracy": metrics['preference_accuracy'],
        "accuracy/correct_preferences": metrics['correct_preferences'],
        "accuracy/total_comparisons": metrics['total_comparisons'],
        "accuracy/positive_diff_ratio": metrics['positive_diff_ratio'],
        
        # Chosen reward statistics
        "chosen_rewards/mean": metrics['chosen_reward_mean'],
        "chosen_rewards/std": metrics['chosen_reward_std'],
        "chosen_rewards/min": metrics['chosen_reward_min'],
        "chosen_rewards/max": metrics['chosen_reward_max'],
        "chosen_rewards/25th_percentile": metrics['chosen_reward_25th'],
        "chosen_rewards/75th_percentile": metrics['chosen_reward_75th'],
        
        # Rejected reward statistics
        "rejected_rewards/mean": metrics['rejected_reward_mean'],
        "rejected_rewards/std": metrics['rejected_reward_std'],
        "rejected_rewards/min": metrics['rejected_reward_min'],
        "rejected_rewards/max": metrics['rejected_reward_max'],
        "rejected_rewards/25th_percentile": metrics['rejected_reward_25th'],
        "rejected_rewards/75th_percentile": metrics['rejected_reward_75th'],
        
        # Reward difference statistics
        "reward_differences/mean": metrics['reward_diff_mean'],
        "reward_differences/std": metrics['reward_diff_std'],
        "reward_differences/median": metrics['reward_diff_median'],
        "reward_differences/min": metrics['reward_diff_min'],
        "reward_differences/max": metrics['reward_diff_max'],
        "reward_differences/25th_percentile": metrics['reward_diff_25th'],
        "reward_differences/75th_percentile": metrics['reward_diff_75th'],
        
        # Separation metrics
        "separation/positive_differences": metrics['positive_differences'],
        "separation/negative_differences": metrics['negative_differences'],
        "separation/zero_differences": metrics['zero_differences'],
        
        # Correlation metrics
        "correlations/pearson_correlation": metrics['pearson_correlation'],
        "correlations/pearson_p_value": metrics['pearson_p_value'],
        "correlations/spearman_correlation": metrics['spearman_correlation'],
        "correlations/spearman_p_value": metrics['spearman_p_value'],
        
        # Configuration for reference
        "config/num_test_samples": NUM_TEST_SAMPLES,
        "config/max_seq_length": MAX_SEQ_LENGTH,
        "config/dataset_name": DATASET_NAME,
    })
    
    # --- 6. Show Sample Results ---
    logger.info("--- Sample Evaluation Results ---")
    for i, sample in enumerate(results['samples']):
        logger.info(f"Sample {i+1}:")
        logger.info(f"Prompt: {sample['prompt'][:100]}...")
        logger.success(f"Chosen: {sample['chosen'][:100]}...")
        logger.warning(f"Rejected: {sample['rejected'][:100]}...")
        logger.info(f"Chosen Reward: {sample['chosen_reward']:.4f}")
        logger.info(f"Rejected Reward: {sample['rejected_reward']:.4f}")
        logger.info(f"Difference: {sample['reward_difference']:.4f}")
        logger.info(f"Correct: {sample['preference_correct']}")
        logger.info("---")
    
    # Log sample results as table
    if results['samples']:
        sample_table = wandb.Table(columns=[
            "prompt", "chosen", "rejected", "chosen_reward", 
            "rejected_reward", "reward_difference", "preference_correct"
        ])
        for sample in results['samples']:
            sample_table.add_data(
                sample['prompt'][:200], sample['chosen'][:200], 
                sample['rejected'][:200], sample['chosen_reward'],
                sample['rejected_reward'], sample['reward_difference'],
                sample['preference_correct']
            )
        wandb.log({"sample_results": sample_table})
    
    # --- 7. Create and Log Visualizations ---
    logger.info("Creating visualizations...")
    fig = evaluator.create_visualizations(results)
    wandb.log({"reward_evaluation_plots": wandb.Image(fig)})
    plt.close(fig)
    
    # --- 8. Save Results ---
    results_path = os.path.join(REWARD_MODEL_DIR, "evaluation_results.pt")
    torch.save({
        'metrics': metrics,
        'results': results,
        'config': {
            'model_path': REWARD_MODEL_DIR,
            'dataset_name': DATASET_NAME,
            'num_samples': NUM_TEST_SAMPLES
        }
    }, results_path)
    
    logger.success(f"Evaluation results saved to: {results_path}")
    logger.success("Reward model evaluation completed!")
    
    # Finish wandb run
    wandb.finish()
