# node1.py
import os, torch
import torch.distributed as dist
from transformers import GPT2LMHeadModel
from torch.nn.functional import softmax
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import DynamicCache


cache_dir = "/media/ssd/huggingface_cache"

def setup():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=2,
        rank=1
    )
    rank = dist.get_rank()
    print(f"[Rank {rank}] Initialized.")
    dist.barrier()
    return rank

def load_half_llama(rank=1):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=cache_dir)
    n_layers = model.config.num_hidden_layers   
    half     = n_layers // 2                   

    if rank == 0:
        for i in reversed(range(half, n_layers)):
            del model.model.layers[i]          
    else:
        for _ in range(half):
            del model.model.layers[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device



def recv_tensor(src, dtype=torch.float32):
    # 1) Recv number of dims
    dims_buf = torch.zeros(1, dtype=torch.long)
    dist.recv(dims_buf, src=src)
    dims = dims_buf.item()
    # 2) Recv shape vector
    shape_buf = torch.zeros(dims, dtype=torch.long)
    dist.recv(shape_buf, src=src)
    shape = tuple(shape_buf.tolist())
    # 3) Allocate contiguous buffer of the right dtype
    buf = torch.zeros(shape, dtype=dtype)
    # 4) Recv actual tensor data
    dist.recv(buf, src=src)
    return buf


def send_token(token, dst):
    dist.send(token.cpu(), dst=dst)

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95):
    # Top-K
    if top_k > 0:
        topk_vals, _ = torch.topk(logits, top_k)
        min_topk = topk_vals[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_topk, torch.full_like(logits, -math.inf), logits)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = -math.inf
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

if __name__ == "__main__":
    rank = setup()
    model, device = load_half_llama(1)
    stage2_cache = None

    # PREFILL receive
    print("[Rank 1] Receiving prefill data")
    hidden = recv_tensor(src=0, dtype=torch.float32).to(device)
    prefix_len = hidden.shape[1]
    legacy_cache = []
    for _ in range(model.config.num_hidden_layers // 2):
        k = recv_tensor(src=0, dtype=torch.float32).to(device)
        v = recv_tensor(src=0, dtype=torch.float32).to(device)
        # cache0.append((k,v))
        legacy_cache.append((k, v))

    stage2_cache = DynamicCache.from_legacy_cache(legacy_cache)
    attention_mask = torch.ones(1, prefix_len, dtype=torch.long, device=device)
    # Run stage 2 on full context
    outputs = model(inputs_embeds=hidden.to("cuda:0"),
                     past_key_values=stage2_cache, 
                     use_cache=True, 
                     return_dict=True,
                     attention_mask=attention_mask)
    logits = outputs.logits  # [1,T,V]
    stage2_cache = outputs.past_key_values

    # Sample first token
    next_logits = logits[:, -1, :]
    filtered = top_k_top_p_filtering(next_logits, top_k=50, top_p=0.95)
    probs = softmax(filtered / 0.8, dim=-1)
    tok = torch.multinomial(probs, num_samples=1)
    print(f"[Rank 1] Prefill sampled token: {tok.item()}")
    send_token(tok, dst=0)

    # GENERATION LOOP
    max_new = 10
    for step in range(max_new):
        print(f"[Rank 1] Waiting for step {step+1}")

        hidden = recv_tensor(src=0, dtype=torch.float32).to(device)

        # legacy_cache = []
        # for _ in range(model.config.num_hidden_layers // 2):
        #     k = recv_tensor(src=0, dtype=torch.float32).to(device)
        #     v = recv_tensor(src=0, dtype=torch.float32).to(device)
        #     legacy_cache.append((k, v))

        # stage2_cache = DynamicCache.from_legacy_cache(legacy_cache)

        attention_mask = torch.cat([
        attention_mask,
        torch.ones(1,1, dtype=torch.long, device=device)], dim=1)

        print(attention_mask.shape)

        outputs = model(
            inputs_embeds=hidden,
            past_key_values=stage2_cache,
            use_cache=True,
            return_dict=True,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        stage2_cache = outputs.past_key_values

        next_logits = logits[:, -1, :]
        filtered = top_k_top_p_filtering(next_logits, top_k=50, top_p=0.95)
        probs = filtered.softmax(dim=-1) / 0.8
        tok = torch.multinomial(probs, num_samples=1)
        print(f"[Rank 1] Step {step+1} token: {tok.item()}")
        
        send_token(tok, dst=0)

    dist.destroy_process_group()
