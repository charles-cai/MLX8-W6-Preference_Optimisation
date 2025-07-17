import torch, gc
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        self.v_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # shape: [B, T, H]
        rewards = self.v_head(last_hidden).squeeze(-1)  # shape: [B, T]
        final_rewards = rewards.gather(1, attention_mask.sum(dim=1).unsqueeze(1) - 1).squeeze(1)
        return final_rewards  # shape: [B]


ranking_loss = torch.nn.MarginRankingLoss(margin=0.5)
loss = ranking_loss(chosen_reward, rejected_reward, torch.ones_like(chosen_reward))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = RewardModel("Qwen/Qwen3-0.6B-Base").to("cuda")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
loss_fn = torch.nn.MarginRankingLoss(margin=0.5)

for epoch in range(1):
    loop = tqdm(dataloader, desc="Training Reward Model")
    for batch in loop:
        prompt = batch["prompt"]
        chosen = batch["accepted"]
        rejected = batch["rejected"]

        chosen_inputs = tokenizer([p + c for p, c in zip(prompt, chosen)], return_tensors="pt", padding=True, truncation=True).to("cuda")
        rejected_inputs = tokenizer([p + r for p, r in zip(prompt, rejected)], return_tensors="pt", padding=True, truncation=True).to("cuda")

        chosen_rewards = model(**chosen_inputs)
        rejected_rewards = model(**rejected_inputs)

        loss = loss_fn(chosen_rewards, rejected_rewards, torch.ones_like(chosen_rewards))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

# Save model
torch.save(model.state_dict(), "reward_model.pth")
print("Reward model saved!")












