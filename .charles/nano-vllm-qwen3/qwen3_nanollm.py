from nanovllm import LLM, SamplingParams

try:
    llm = LLM(".data/huggingface/Qwen3-1.7B/", enforce_eager=True, tensor_parallel_size=1)
except RuntimeError as e:
    print("RuntimeError during LLM initialization:", e)
    print("Hint: Make sure a C compiler is installed and the CC environment variable is set.")
    raise

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])

prompts = ["why the sky is blue?"]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])