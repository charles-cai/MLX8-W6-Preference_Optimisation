import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW  # ✅ correct
from tqdm import tqdm
from dataloader import RedditSummarySFTDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)


# Assuming you have your dataset class imported and ready
train_dataset = RedditSummarySFTDataset(split="train", model_name="Qwen/Qwen3-0.6B", max_length=512)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# Load tokenizer and model, move model to device
tokenizer.pad_token = tokenizer.eos_token  # make sure pad token is set

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True).to(device)
model.train()

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)


# Function to mask prompt tokens in labels to ignore loss on them
def mask_prompt_tokens(input_ids, labels, tokenizer):
    sep_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    labels = labels.clone()

    for i in range(labels.size(0)):
        row = input_ids[i]
        if not torch.is_tensor(row):
            row = torch.tensor(row, device=labels.device)

        # Find where <|assistant|> token starts
        print(f"Type of row: {type(row)}")
        print(f"Shape of input_ids: {input_ids.shape}")
        print(f"input_ids[i]: {input_ids[i]}")
        print(f"sep_token_id: {sep_token_id}")

        sep_positions = (row == sep_token_id).nonzero(as_tuple=True)

        if len(sep_positions[0]) > 0:
            sep_index = sep_positions[0][0]
            labels[i, :sep_index] = -100  # mask prompt tokens

    return labels


# Training loop
def train_sft(epochs=1):
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
        for batch_idx, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Mask prompt tokens in labels (only compute loss on assistant response)
            labels = mask_prompt_tokens(input_ids, labels, tokenizer)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
            # Optional detailed print:
            # print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_sft(epochs=1)
