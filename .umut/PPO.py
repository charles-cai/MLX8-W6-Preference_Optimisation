import torch, os
from torch import nn
from torch.nn import functional as F
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"



base_model_name = "Qwen/Qwen3-0.6B-Base"
lora_ckpt_path = "qwen-lora-sft-epoch1.pth"  # your LoRA checkpoint path
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load base model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# ---- Load policy model: base + LoRA ----
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


policy_model = get_peft_model(base_model, lora_config)
policy_model.load_state_dict(torch.load(lora_ckpt_path, map_location="cpu"), strict=False)
policy_model = policy_model.to(device)
policy_model.eval()

# ---- Load reference model (frozen base model) ----
ref_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)
ref_model.eval()


reward_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # or others depending on your model
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",  # ✅ IMPORTANT: Classification, not CausalLM
)


reward_model = AutoModelForSequenceClassification.from_pretrained(base_model_name).to(device)
reward_lora_path = "reward_model_epoch1.pth"
# ---- Load reward model (returns scalar reward) ----
reward_model = get_peft_model(reward_model, reward_config)
reward_model.load_state_dict(torch.load(reward_lora_path, map_location="cpu"), strict=False)
reward_model.eval()





tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)
clip_epsilon = 0.2

train_dataloader = [
    "Summarize this Reddit thread: ...",
    "What is the main point of this discussion?",
    "Give a TL;DR for this conversation:",
    "Summarize the key takeaways from this post:"
]

from torch.utils.data import DataLoader
from dataloaderV2 import RedditSummarySFTDataset  # ✅ Your dataset must return accepted/rejected pairs
train_dataset = RedditSummarySFTDataset(split="train", model_name="Qwen/Qwen3-0.6B", max_length=512, start_idx = 0, end_idx = 20000)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=12, pin_memory=True)



from tqdm import tqdm

num_epochs = 1

for epoch in range(num_epochs):
    loop = tqdm(train_loader, desc=f"PPO Epoch {epoch+1}", leave=True)
    
    for batch_idx, batch in enumerate(loop):
        # Get prompt (assuming batch is a dict with 'prompt' field)
        prompt_texts = batch["prompt"] if isinstance(batch, dict) else batch

        # Tokenize prompts
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)


        with torch.no_grad():
            response_ids = policy_model.generate(
                input_ids,
                max_new_tokens=100,
                attention_mask=attention_mask,  # ✅ include this
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id  # ✅ here
            )
        
        # Combine prompt and response
        full_input = torch.cat([input_ids, response_ids[:, input_ids.shape[-1]:]], dim=1)
        attention_mask = torch.ones_like(full_input)

        # --- Get log probs (policy vs ref)
        with torch.no_grad():
            ref_logits = ref_model(full_input, attention_mask=attention_mask).logits
        policy_logits = policy_model(full_input, attention_mask=attention_mask).logits

        # Get log-probs for the response tokens
        log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        chosen_tokens = full_input[:, 1:]  # skip first token
        chosen_log_probs = log_probs[:, :-1].gather(2, chosen_tokens.unsqueeze(-1)).squeeze(-1)
        ref_chosen_log_probs = ref_log_probs[:, :-1].gather(2, chosen_tokens.unsqueeze(-1)).squeeze(-1)

        # --- Calculate rewards and advantage
        with torch.no_grad():
            reward = reward_model(full_input).logits.squeeze()
            advantage = reward - reward.mean()  # optionally normalize
            advantage = advantage.unsqueeze(1)  # shape: [batch_size, 1]

        # --- Compute PPO loss
        log_ratio = chosen_log_probs - ref_chosen_log_probs
        ratio = torch.exp(log_ratio)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
        loss = -torch.min(unclipped, clipped).mean()

        # --- Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f} | Reward: {reward.mean().item():.4f}")
        from peft import PeftModel, PeftConfig

        # Save only the LoRA adapter (recommended way with `peft`)
        policy_model.save_pretrained("ppo-qwen-lora-adapter")
        torch.save(policy_model.state_dict(), "ppo-qwen-full.pth")

