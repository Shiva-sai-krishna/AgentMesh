import os
from utils import setup, cleanup, build_local_model_gpt2xl, send_tensor, recv_token, recv_tensor, deterministic
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.nn.functional import softmax
import torch.distributed as dist
import math
from utils import top_k_top_p_filtering, send_token


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deterministic()
    
    rank, world, hf, max_tokens = setup()
    print(f"[Rank {rank}] World size is {world}")
    
    src = rank - 1
    if src < 0 : 
        src = world - 1
    dst = (rank + 1)% world
    print(f"[Rank {rank}] Source: {src}, Destination: {dst}")

    model = build_local_model_gpt2xl(rank, world, hf)
    print(f"[Rank {rank}] Model built successfully.")
    print(model)

    hidden = recv_tensor(src=src, dtype=torch.float32).to(device)
    print(f"[Rank {rank}] Received prefill hidden")

    cache = []
    print(f"[Rank {rank}] cache initialized")
    for _ in range(model.config.n_layer // world):
        k = recv_tensor(src=src, dtype=torch.float32).to(device)
        v = recv_tensor(src=src, dtype=torch.float32).to(device)
        cache.append((k,v))
    print(f"[Rank {rank}] Received prefill cache")

    outputs = model(inputs_embeds=hidden.to("cuda:0"), past_key_values=cache, use_cache=True, return_dict=True, output_hidden_states=True)
    print(f"[Rank {rank}] Model forward pass completed")

    if rank == world - 1 : 
        logits = outputs.logits
        next_logits = logits[:, -1, :]
        filtered = top_k_top_p_filtering(next_logits, top_k=50, top_p=0.95)
        probs = softmax(filtered / 0.8, dim=-1)
        tok = torch.multinomial(probs, num_samples=1)
        print(f"[Rank {rank}] Prefill sampled token: {tok.item()}")
        send_token(tok, dst=dst)
    
    else : 
        hidden  = outputs.hidden_states[-1]
        cache  = outputs.past_key_values
        send_tensor(hidden, dst=dst)
        for layer_kvs in cache:
            for part in layer_kvs:
                send_tensor(part, dst=dst)
        print(f"[Rank {rank}] Prefill hidden & cache sent")


    max_new = max_tokens
    for step in range(max_new):
        print(f"[Rank {rank}] Waiting for step {step+1}")
        hidden = recv_tensor(src=src, dtype=torch.float32).to(device)
        cache = []
        for _ in range(model.config.n_layer // world):
            k = recv_tensor(src=src, dtype=torch.float32).to(device)
            v = recv_tensor(src=src, dtype=torch.float32).to(device)
            cache.append((k, v))
        print(f"[Rank {rank}] Received step {step+1} hidden & cache")

        outputs = model(inputs_embeds=hidden, past_key_values=cache, use_cache=True, return_dict=True, output_hidden_states=True)
        
    
        if rank == world - 1 : 
            logits = outputs.logits
            next_logits = logits[:, -1, :]
            filtered = top_k_top_p_filtering(next_logits, top_k=50, top_p=0.95)
            probs = filtered.softmax(dim=-1) / 0.8
            tok = torch.multinomial(probs, num_samples=1)
            print(f"[Rank 1] Step {step+1} token: {tok.item()}")
            send_token(tok, dst=dst)

        else : 
            
            hidden  = outputs.hidden_states[-1]
            cache  = outputs.past_key_values

            send_tensor(hidden, dst=dst)
            for layer_kvs in cache:
                for part in layer_kvs:
                    send_tensor(part, dst=dst)
            print(f"[Rank {rank}] Step {step+1} hidden & cache sent")
    cleanup()
