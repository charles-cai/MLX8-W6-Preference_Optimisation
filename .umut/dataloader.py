from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

class RedditSummarySFTDataset(Dataset):
    def __init__(self, split="train", model_name="Qwen/Qwen1.5-0.5B", max_length=512):
        
        dataset = load_dataset("CarperAI/openai_summarize_comparisons")[split],
        subset_size=None, 
        seed=42,             # Optional for reproducibility
        
        # If subset_size is given, slice the dataset
        if subset_size is not None:
            dataset = dataset.shuffle(seed=seed).select(range(subset_size))

        self.dataset = dataset

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Qwen often lacks pad_token
        self.max_length = max_length,

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        # Qwen prefers chat-style formatting
        full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{chosen}"

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": tokenized["input_ids"].squeeze(0),  # same as input for SFT
        }

