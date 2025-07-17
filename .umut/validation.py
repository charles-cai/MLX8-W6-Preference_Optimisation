import torch, os, gc 
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from peft import PeftModel, LoraConfig, get_peft_model
from dataloaderV2 import RedditSummarySFTDataset  # ✅ Your dataset must return accepted/rejected pairs
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# --- Config ---
base_model_name = "Qwen/Qwen3-0.6B-Base"
lora_ckpt_path = "qwen-lora-sft-epoch1.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load base model & tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)

# --- Attach LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.load_state_dict(torch.load(lora_ckpt_path, map_location=torch.device('cpu')), strict=False)
model = model.to(device)
model.eval()

# --- Dataloader setup (you must define this yourself) ---
# For example, assuming test_loader yields a dict with 'text' key
# from torch.utils.data import DataLoader
# test_loader = DataLoader(test_dataset, batch_size=1)
test_dataset = RedditSummarySFTDataset(split="test", max_length=512, start_idx = 0, end_idx = 2000)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=12, pin_memory=True)


# --- Inference on 10 prompts from test_loader ---
print("🔁 Running summarization on 10 prompts from test_loader...\n")
sum_instruct = "Summarize in 1 sentence. Capture the main point: \n"

num_processed = 0
for batch in test_loader:
    if num_processed >= 10:
        break

    # Extract the prompt from the batch
    if isinstance(batch, dict):
        prompt = batch['prompt'][0]  # assuming batch_size = 1
        #prompt = '\n'.join(prompt.split('\n')[1:])
    else:
        raise ValueError("Expected dict from test_loader, got:", type(batch))
    formatted = f"""<|im_start|>system
You are a helpful assistant that summarizes text.<|im_end|>
<|im_start|>user
Summarize the following text in exactly in 2 complete sentences, no more. Give general context in first sentence and summarize in the second sentence. Here is text: {prompt}"""
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=85,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
)
# THIS SETUP NAILED SHORT AND SWEET SUMMARIES

    summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"📝 Prompt {num_processed + 1}:\n{prompt}...\n")  # print only first 300 chars
    print(f"🤖 Summary {num_processed + 1}: {summary.strip()}\n{'-'*280}")

    num_processed += 1

print("\n✅ Done summarizing 10 samples.")
del inputs, outputs, summary, formatted, prompt
gc.collect()

