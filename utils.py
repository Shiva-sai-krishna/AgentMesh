import os
import torch.distributed as dist
from transformers import GPT2LMHeadModel, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax
import math
import random, numpy as np, torch
import csv
import time


def append_to_csv(writer, row_data, file):
    """
    Append a single row of data to a CSV file. If the file does not exist, it will be created.

    Args:
        file_path (str): Path to the target CSV file.
        row_data (list or tuple): An iterable of values to write as one row.
    """
    writer.writerow(row_data)
    file.flush()


def deterministic() : 
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"])
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    hf_cache = os.environ["HF_HOME"]
    max_tokens = int(os.environ["MAX_TOKENS"])

    dist.barrier()  
    params = (rank, world, hf_cache, max_tokens)
    return params 

def partition_layers(num_layers: int, world_size: int, rank: int):                  
    base = num_layers // world_size
    remainder = num_layers % world_size
    extra = 1 if rank < remainder else 0
    start = rank * base + min(rank, remainder)
    end   = start + base + extra
    return start, end   

from torch.nn import Identity

def build_local_model_gpt2xl(rank: int, world_size: int, cache_dir: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model options
    # openai-community/gpt2
    # openai-community/gpt2-medium
    # openai-community/gpt2-large  -- done
    # openai-community/gpt2-xl

    full =  AutoModelForCausalLM.from_pretrained("openai-community/gpt2", cache_dir=cache_dir)
    start, end = partition_layers(len(full.transformer.h), world_size, rank)
    full.transformer.h = torch.nn.ModuleList(full.transformer.h[start:end])
    print(f"[Rank {rank}] Layers {start+1}-{end}")

    if rank != 0:
        full.transformer.wte = Identity()
        full.transformer.wpe.weight.data.zero_()
        full.transformer.wpe.weight.requires_grad_(False)

    if rank != world_size - 1:
        full.transformer.ln_f = Identity()
        full.lm_head       = Identity()

    full.to(device)
    return full



def send_tensor(tensor, dst, writer=None, label=""):
    tensor = tensor.contiguous()
    dims = torch.tensor([tensor.dim()], dtype=torch.long)
    dist.send(dims, dst=dst)
    shape = torch.tensor(tensor.shape, dtype=torch.long)
    dist.send(shape, dst=dst)
    dist.send(tensor.cpu(), dst=dst)

    if writer is not None:
        size_in_bytes = tensor.element_size() * tensor.nelement()
        writer.writerow(["Tensor Send", dst, label, f"{list(tensor.shape)}", size_in_bytes])

def recv_token(src):
    buf = torch.zeros((1,1), dtype=torch.long)
    dist.recv(buf, src=src)
    return buf

def cleanup():
    dist.destroy_process_group()

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
    # Top-P (nucleus)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative > top_p
    # shift mask right to include at least one token
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = -math.inf
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

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
    print(f"{time.time()} tensor received")
    return buf
