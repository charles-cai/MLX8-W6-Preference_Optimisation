import random
from datasets import load_dataset
from logger_utils import setup_logger
from logger_utils import format_number

# Model and dataset identifiers
model_id = "Qwen/Qwen2-0.5B-Base"

dataset_name_tldr = "CarperAI/openai_summarize_tldr" # 
dataset_name_comparison = "CarperAI/openai_summarize_comparisons" 

Prompts = [
    "Summarize the following post", 
     "Sumarize the following post in a few sentences:",
    "Please provide a concise summary of the content below:",
    "Can you give me a brief overview of the text that follows?",
    "Write a short recap of the main points in this post:",
    "Condense the key ideas from the following passage into a few sentences:",
    "Create a quick synopsis of the material below:",
    "Highlight the essential takeaways from this post:",
    "Craft a succinct digest of the following content:",
    "Explain the core message of this post in summary form:",
    "Summarize the central themes of the text provided:"
    ]

# --- OpenAI dataset, not used, Hugging Face seems NOT supporting old dataset shape like this one ---
def prepare_dataset_openai(database_name = "openai/summarize_from_feedback", split='train'):
    # We use the 'comparisons' configuration of the dataset
    dataset = load_dataset(database_name, "comparisons", split=split)

    dataset = dataset.filter(lambda x: len(x['summaries']) == 2 and x['info']['post'] is not None)
    
    # The dataset has a 'choice' field: 0 means the first summary is better, 1 means the second one is.
    def format_data(example):
        # The post is the prompt
        prompt ="Summarize the following post: \n\n" + example['info']['post']
        
        # Assign chosen and rejected summaries based on the 'choice' field
        chosen_summary = example['summaries'][example['choice']]['text']
        rejected_summary = example['summaries'][1 - example['choice']]['text']
        
        return {
            "prompt": prompt,
            "chosen": chosen_summary,
            "rejected": rejected_summary,
        }
        
    return dataset.map(format_data, remove_columns=dataset.column_names)

def prepare_dataset_comparison(dataset_name, split='train'):
    # Load the dataset with the specified split
    dataset = load_dataset(dataset_name, split=split)

    # Format the dataset to have 'prompt', 'chosen', and 'rejected' fields
    def format_data(example):
        prompt = random.choice(Prompts) + "\n\n" + example['prompt']
        chosen_summary = example['chosen']
        rejected_summary = example['rejected']
        
        return {
            "prompt": prompt,
            "chosen": chosen_summary,
            "rejected": rejected_summary,
        }
        
    return dataset.map(format_data, remove_columns=dataset.column_names)

def prepare_dataset_tldr(dataset_name, split='train'):
    # Load the dataset with the specified split
    dataset = load_dataset(dataset_name, split=split)

    # Format the dataset to have 'prompt', 'chosen', and 'rejected' fields
    def format_data(example):
        prompt = random.choice(Prompts) + "\n\n" + example['prompt']
        label = example['label']
        
        return {
            "prompt": prompt,
            "label": label,
        }
        
    return dataset.map(format_data, remove_columns=dataset.column_names)

if __name__ == "__main__":
    logger = setup_logger("dataset_test")

    train_dataset_tldr = prepare_dataset_tldr(dataset_name_tldr, "train")
    valid_dataset_tldr = prepare_dataset_tldr(dataset_name_tldr, "valid")
    test_dataset_tldr = prepare_dataset_tldr(dataset_name_tldr, "test") 
    logger.success("openai summary tldr datasets loaded:\n" + 
                   f"Train dataset records: {format_number(len(train_dataset_tldr))}\n" + 
                   f"Validation dataset records: {format_number(len(valid_dataset_tldr))}\n" + 
                   f"Test dataset records: {format_number(len(test_dataset_tldr))}")

    train_dataset_comparison = prepare_dataset_comparison(dataset_name_comparison, "train")
    valid1_dataset_comparison = prepare_dataset_comparison(dataset_name_comparison, "valid1")
    valid2_dataset_comparison = prepare_dataset_comparison(dataset_name_comparison, "valid2")
    test_dataset_comparison = prepare_dataset_comparison(dataset_name_comparison, "test") 

    logger.success("openai summary comparisons datasets loaded:\n" +
                   f"Train dataset records: {format_number(len(train_dataset_comparison))}\n" +
                   f"Validation1 dataset records: {format_number(len(valid1_dataset_comparison))}\n" +
                   f"Validation2 dataset records: {format_number(len(valid2_dataset_comparison))}\n" +
                   f"Test dataset records: {format_number(len(test_dataset_comparison))}")

"""    
SUCCESS  | openai summary tldr datasets loaded:
Train dataset records: 116.7k
Validation dataset records: 6.4k
Test dataset records: 6.6k

SUCCESS  | openai summary comparisons datasets loaded:
Train dataset records: 92.5k
Validation1 dataset records: 33.1k
Validation2 dataset records: 50.7k
Test dataset records: 83.6k
"""