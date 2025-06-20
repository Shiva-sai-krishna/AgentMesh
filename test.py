from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/media/ssd/huggingface_cache"

full =  AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=cache_dir)

print(full)