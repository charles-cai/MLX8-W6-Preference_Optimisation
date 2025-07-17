from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm

class RedditSummarySFTDataset(Dataset):
    def __init__(self, split="train", model_name="Qwen/Qwen1.5-0.5B", max_length=512,
        start_idx = 0,
        end_idx = 20000,
        seed=42,):

        dataset = load_dataset("CarperAI/openai_summarize_comparisons", split=split)
        dataset = dataset.select(range(start_idx, end_idx))
        self.dataset = dataset

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        self.tokenized_data = []
        for row in tqdm(self.dataset, desc="Pre-tokenizing dataset"):
            full_text = f"<|user|>\n{row['prompt']}\n<|assistant|>\n{row['chosen']}"
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            self.tokenized_data.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "labels": tokenized["input_ids"].squeeze(0),
            })

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]
