# node0.py
import os, torch
import torch.distributed as dist
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/media/ssd/huggingface_cache"

def setup():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=2,
        rank=0
    )
    rank = dist.get_rank()
    print(f"[Rank {rank}] Initialized.")
    dist.barrier()
    return rank

def load_half_llama(rank=0):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=cache_dir)
    n_layers = model.config.num_hidden_layers   # e.g., 32 
    half     = n_layers // 2                   # e.g., 16

    # On Node 0: keep layers [0..half-1], delete [half..n_layers-1]
    if rank == 0:
        for i in reversed(range(half, n_layers)):
            del model.model.layers[i]          # note .model.layers, not .transformer.h
    # On Node 1: keep layers [half..n_layers-1], delete [0..half-1]
    else:
        for _ in range(half):
            del model.model.layers[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device


def send_tensor(tensor, dst):
    # 1) Make contiguous
    tensor = tensor.contiguous()
    # 2) Send number of dims
    dims = torch.tensor([tensor.dim()], dtype=torch.long)
    dist.send(dims, dst=dst)
    # 3) Send shape vector
    shape = torch.tensor(tensor.shape, dtype=torch.long)
    dist.send(shape, dst=dst)
    # 4) Send the actual data on CPU
    dist.send(tensor.cpu(), dst=dst)


def recv_token(src):
    buf = torch.zeros((1,1), dtype=torch.long)
    dist.recv(buf, src=src)
    return buf

if __name__ == "__main__":
    rank = setup()
    model, device = load_half_llama(0)
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=cache_dir)
    print("Tokenizer loaded")

    # PREFILL
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model(inputs, use_cache=True, return_dict=True, output_hidden_states=True)
      # cache during forward 
    hidden  = outputs.hidden_states[-1]
    cache0  = outputs.past_key_values

    print("[Rank 0] Sending prefill hidden & cache")
    send_tensor(hidden, dst=1)
    for layer_kvs in cache0:
        for part in layer_kvs:
            send_tensor(part, dst=1)

    token     = recv_token(src=1).to(device)
    generated = torch.cat([inputs, token], dim=-1)
    print(f"[Rank 0] Received initial token: {token.item()}")

    # GENERATION LOOP on Node 0
    max_new = 10
    for step in range(max_new):
        print(f"[Rank 0] Generation step {step+1}/{max_new}")

        outputs = model(
            generated[:, -1:].to(device),
            past_key_values=cache0,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]       # last layer’s hidden
        cache0 = outputs.past_key_values         # updated KV cache

        # send_tensor(hidden, dst=1)
        # for layer_kvs in cache0:
        #     k, v = layer_kvs
        #     send_tensor(k, dst=1)
        #     send_tensor(v, dst=1)

        send_tensor(hidden, dst=1) 

        token = recv_token(src=1).to(device)
        generated = torch.cat([generated, token], dim=-1)
        print(f"[Rank 0] Step {step+1} token: {token.item()}")

    print("[Rank 0] Final:", tokenizer.decode(generated[0], skip_special_tokens=True))
    dist.destroy_process_group()
