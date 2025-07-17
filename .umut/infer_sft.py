import torch, os, gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

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
#model = get_peft_model(model, lora_config)
#model.load_state_dict(torch.load(lora_ckpt_path, map_location=torch.device('cpu')), strict=False)
model = model.to("cpu")
model.eval()

print("BOS token:", tokenizer.bos_token)
print("EOS token:", tokenizer.eos_token)
print("PAD token:", tokenizer.pad_token)
print("Special tokens:", tokenizer.special_tokens_map)
print("All added special tokens:", tokenizer.additional_special_tokens)




# --- Inference loop ---
print("🔁 Ready. Type a prompt and press Enter (Ctrl+C to exit).")
prompt = """
SUBREDDIT: r/relationships
TITLE: One sided pleasure, okay or not? Overreacting?
POST: I'm a 23 year old male that have been dating a girl (also 23) for 4 months, we have "been official" for three months.

This girl was my first, she took my virginity. She lost her virginity at 16 and has had more than ten partners. I have no issues with that. For the first few weeks I was unable to orgasm due to nervousnes and unfamiliarity. This resolved itself. The last month or so she has been unable to reach orgasm, due to stress. She assures me that the sex is still good and that she has a psychological cap preventing her from coming. While I accept the possibility that she isn't happy with the sex, I do believe her.

Well now two days in a row we have been in bed with me fingering her followed by her feeling "satisfied" (She certainly enjoyed it but didn't climax) and interrupting the cuddling, not returning the favour. Both times she barely touched me. It hurt me a lot and the second time I expressed my feelings to her.

She claims that sometimes you give and sometimes you get, and that it is natural for one partner to please the other without expecting something in return. That it is okay sometimes to be selfish. This has worked well in her previous relationships and has felt natural. She said she wanted to focus on her to be able to relax completely to make it easier for her to come. I told her that it was the fact that she didn't explain this to me that bothered me, that my expectations were off and therefore I became hurt and disappointed. After thinking about it some more I think that is not the case though.

I have been trying to wrap my head around the idea but I can't really see myself not taking offence when someone doesn't offer to return the favor. To me it feels like someone saying "I can't be bothered wasting energy on you".

I have been very emotional lately due to stress over work and sickness (I have a middle ear inflammation and a cold). Am I overreacting? Is it normal in relationships to have one-sided sexual pleasure?...
"""


sum_instruct = "Summarize in 1 sentence. Capture the main point: \n"
formatted = f"<|im_start|>\n{sum_instruct}{prompt}\n<|endoftext|>\n"
formatted = f"""<|im_start|>system
You are a helpful assistant that summarizes text.<|im_end|>
<|im_start|>user
Summarize the following text in exactly one complete sentence, no more: {prompt}"""
inputs = tokenizer(formatted, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=555,
        do_sample=False,
        num_beams=4,
        early_stopping=False,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("🤖 Response:", response)

print("\n👋 Exiting.")
del inputs, outputs, response, formatted, prompt
gc.collect()
