import os
from utils import setup, cleanup, partition_layers
from transformers import GPT2LMHeadModel
import torch

def build_local_model_gpt2xl(rank: int, world_size: int, cache_dir: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full = GPT2LMHeadModel.from_pretrained("gpt2-xl",cache_dir=cache_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    blocks = full.transformer.h
    start, end = partition_layers(len(blocks), world_size, rank)
    full.transformer.h = torch.nn.ModuleList(blocks[start:end])

    if rank != 0:
        del full.transformer.wte
        del full.transformer.wpe
        del full.transformer.drop
 
    if rank != world_size - 1:
        del full.transformer.ln_f
        del full.lm_head
 
    def _id(name):
        setattr(full, name, torch.nn.Identity())
 
    if rank != 0:
        _id("embed_forward")
 
    full.to(device)
    return full





if __name__ == "__main__":
    params = setup()
    print(f"[Rank {params[0]}] World size: {params[1]}")
    model = build_local_model_gpt2xl(params[0], params[1], params[2])
    print(f"[Rank {params[0]}] Model built successfully.")
    print(model)
    cleanup()
