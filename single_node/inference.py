import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np

CACHE_DIR = "/media/ssd/huggingface_cache"
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=CACHE_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
tokenizer.pad_token = tokenizer.eos_token

def deterministic():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

deterministic()

# Batched prompts
prompt = [
    "What is the meaning of life",
    "What is the capital of France",
    "Explain the theory of relativity",
    "How does photosynthesis work",
    "What are the benefits of meditation"
]

E2E_start = time.time()

# Tokenize with padding and attention mask
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

max_new_tokens = 64
generated = input_ids
past = None

for _ in range(max_new_tokens):
    # Only use the last token for each sequence
    next_input = generated[:, -1].unsqueeze(-1)
    outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
    logits, past = outputs.logits, outputs.past_key_values

    logits = logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat([generated, next_token], dim=-1)

E2E_end = time.time()

# Decode each generated sequence individually
final_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

print("Generated Texts:\n")
for i, txt in enumerate(final_texts):
    print(f"[Prompt {i+1}] {txt}\n")

print(f"End-to-end inference time: {E2E_end - E2E_start:.4f} seconds")
