import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataloaderV2 import RedditSummarySFTDataset  # ✅ Your dataset must return accepted/rejected pairs


class RewardModel(nn.Module):
    def __init__(self, base_model_name, lora_ckpt_path):
        super().__init__()
        print("[*] Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        print("[*] Applying LoRA...")
        peft_model = get_peft_model(base, lora_config)
        peft_model.load_state_dict(torch.load(lora_ckpt_path), strict=False)
        self.backbone = peft_model

        hidden_size = self.backbone.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]  # [B, T, H]
        rewards = self.v_head(last_hidden).squeeze(-1)  # [B, T]
        last_token_index = attention_mask.sum(dim=1) - 1
        final_rewards = rewards.gather(1, last_token_index.unsqueeze(1)).squeeze(1)  # [B]
        return final_rewards


# Setup
device = torch.device("cuda")
model = RewardModel("Qwen/Qwen3-0.6B-Base", "qwen-lora-sft-epoch1.pth").to(device)

# Freeze backbone
for name, param in model.backbone.named_parameters():
    param.requires_grad = False
model.v_head.requires_grad_(True)

# Optimizer & loss
optimizer = torch.optim.AdamW(model.v_head.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# Load data
print("[*] Loading reward dataset...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
dataset = RedditSummaryRewardDataset(split="train", model_name="Qwen/Qwen3-0.6B-Base", max_length=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# Training loop
epochs = 1
print("[*] Starting training...")
for epoch in range(epochs):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(loop):
        chosen_input_ids = batch["accepted_input_ids"].to(device)
        chosen_attention = batch["accepted_attention_mask"].to(device)
        rejected_input_ids = batch["rejected_input_ids"].to(device)
        rejected_attention = batch["rejected_attention_mask"].to(device)

        chosen_rewards = model(chosen_input_ids, chosen_attention)
        rejected_rewards = model(rejected_input_ids, rejected_attention)

        # Simple reward difference loss (Reward > for accepted)
        loss = loss_fn(chosen_rewards, rejected_rewards + 1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item(), reward_diff=(chosen_rewards - rejected_rewards).mean().item())

        # Optional debug
        if step % 20 == 0:
            print(f"[{step}] Loss: {loss.item():.4f} | Reward mean diff: {(chosen_rewards - rejected_rewards).mean().item():.4f}")

        del chosen_input_ids, rejected_input_ids, chosen_attention, rejected_attention
        torch.cuda.empty_cache()
        gc.collect()

    # Save reward model
    torch.save(model.state_dict(), f"reward_model_epoch{epoch+1}.pth")
    print(f"[✓] Reward model saved to 'reward_model_epoch{epoch+1}.pth'")

