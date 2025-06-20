# node0.py
import os, torch
import torch.distributed as dist
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

def load_half_model():
    print("[Rank 0] Loading GPT-2 xl...")
    # Load full GPT2 with LM head
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl", cache_dir="/media/ssd/huggingface_cache")  # no use_cache here 

    # Keep only layers 0–5
    num_layers = model.config.n_layer  # should be 12 
    for i in reversed(range(num_layers // 2, num_layers)):
        del model.transformer.h[i]

    # Safely delete the LM head if it exists
    # if hasattr(model, "lm_head"):
    #     del model.lm_head
    #     print("[Rank 0] lm_head removed")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print("[Rank 0] Model loaded and truncated to layers 0–5.")
    return model, device

# def send_tensor(tensor, dst):
#     shape = torch.tensor(tensor.shape, dtype=torch.long)
#     dist.send(shape, dst=dst)
#     dist.send(tensor.cpu(), dst=dst)

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
    model, device = load_half_model()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("Tokenizer loaded")

    # PREFILL
    # TODO: log time at start
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model(inputs, use_cache=True, return_dict=True, output_hidden_states=True)
      # cache during forward 
    hidden  = outputs.hidden_states[-1]
    cache0  = outputs.past_key_values

    print("[Rank 0] Sending prefill hidden & cache")
    send_tensor(hidden, dst=1)
    # dist.send(torch.tensor([len(cache0)], dtype=torch.long), dst=1)
    for layer_kvs in cache0:
        for part in layer_kvs:
            send_tensor(part, dst=1)

    token     = recv_token(src=1).to(device)
    generated = torch.cat([inputs, token], dim=-1)
    print(f"[Rank 0] Received initial token: {token.item()}")

    # GENERATION LOOP on Node 0
    max_new = 50
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

        send_tensor(hidden, dst=1)
        for layer_kvs in cache0:
            k, v = layer_kvs
            send_tensor(k, dst=1)
            send_tensor(v, dst=1)

        token = recv_token(src=1).to(device)
        generated = torch.cat([generated, token], dim=-1)
        print(f"[Rank 0] Step {step+1} token: {token.item()}")

    print("[Rank 0] Final:", tokenizer.decode(generated[0], skip_special_tokens=True))
    dist.destroy_process_group()
