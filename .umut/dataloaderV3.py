from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

class RedditSummaryRewardDataset(Dataset):
    def __init__(
        self,
        split="train",
        model_name="Qwen/Qwen1.5-0.5B",
        max_length=512,
        start_idx=0,
        end_idx=20000,
        seed=42,
    ):
        dataset = load_dataset("CarperAI/openai_summarize_comparisons", split=split)
        dataset = dataset.select(range(start_idx, end_idx))
        self.dataset = dataset

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        self.tokenized_data = []
        for row in tqdm(self.dataset, desc="Pre-tokenizing reward dataset"):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]

            # Tokenize accepted (chosen)
            accepted_text = f"<|user|>\n{prompt}\n<|assistant|>\n{chosen}"
            accepted = self.tokenizer(
                accepted_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Tokenize rejected
            rejected_text = f"<|user|>\n{prompt}\n<|assistant|>\n{rejected}"
            rejected = self.tokenizer(
                rejected_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            self.tokenized_data.append({
                "accepted_input_ids": accepted["input_ids"].squeeze(0),
                "accepted_attention_mask": accepted["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

class RedditSummaryRewardDataset(Dataset):
    def __init__(
        self,
        split="train",
        model_name="Qwen/Qwen1.5-0.5B",
        max_length=512,
        start_idx=0,
        end_idx=20000,
        seed=42,
    ):
        dataset = load_dataset("CarperAI/openai_summarize_comparisons", split=split)
        dataset = dataset.select(range(start_idx, end_idx))
        self.dataset = dataset

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        self.tokenized_data = []
        for row in tqdm(self.dataset, desc="Pre-tokenizing reward dataset"):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]

            # Tokenize accepted (chosen)
            accepted_text = f"<|user|>\n{prompt}\n<|assistant|>\n{chosen}"
            accepted = self.tokenizer(
                accepted_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Tokenize rejected
            rejected_text = f"<|user|>\n{prompt}\n<|assistant|>\n{rejected}"
            rejected = self.tokenizer(
                rejected_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            self.tokenized_data.append({
                "accepted_input_ids": accepted["input_ids"].squeeze(0),
                "accepted_attention_mask": accepted["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

