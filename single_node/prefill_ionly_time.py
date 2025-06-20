import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

CACHE_DIR = "/media/ssd/huggingface_cache"
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=CACHE_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

prompt = "Ronaldo began his senior career with Sporting CP, before signing with Manchester United in 2003..."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    # start_event.record()
    prefill_start = time.time()
    outputs = model(**inputs)  # prefill only
    prefill_end = time.time()
    # end_event.record()

# torch.cuda.synchronize()

# Compute elapsed time in milliseconds
# elapsed_time_ms = start_event.elapsed_time(end_event)
elapsed_time_ms = (prefill_end - prefill_start) * 1000  # Convert to milliseconds
print(f"Prefill (forward) GPU time: {elapsed_time_ms:.2f} ms")
