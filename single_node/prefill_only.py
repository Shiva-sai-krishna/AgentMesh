import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_DIR = "/media/ssd/huggingface_cache"
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", 
                                          cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2", 
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float32,
    )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

prompt_64 = "Ronaldo began his senior career with Sporting CP, before signing with Manchester United in 2003, winning the FA Cup in his first season. He went on to become a star player at United, as they won three consecutive Premier League titles, the Champions League and the FIFA Club World Cup; his 2007-08 season earned"
inputs = tokenizer(prompt_64, return_tensors="pt").to(device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    start_event.record()
    outputs = model(**inputs)  # prefill only
    end_event.record()

torch.cuda.synchronize()

# Compute elapsed time in milliseconds
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"Prefill (forward) GPU time: {elapsed_time_ms:.2f} ms")
