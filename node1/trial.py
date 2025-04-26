# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
cache_dir = "/media/ssd/huggingface_cache"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=cache_dir)