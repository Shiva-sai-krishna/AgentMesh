import os
import torch.distributed as dist
from transformers import GPT2LMHeadModel
import torch

from torch.nn.functional import softmax
import math

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

    dist.barrier()  
    params = (rank, world, hf_cache)
    return params 

def partition_layers(num_layers: int, world_size: int, rank: int):                  
    base = num_layers // world_size
    remainder = num_layers % world_size
    extra = 1 if rank < remainder else 0
    start = rank * base + min(rank, remainder)
    end   = start + base + extra
    return start, end   

def build_local_model_gpt2xl(rank: int, world_size: int, cache_dir: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full = GPT2LMHeadModel.from_pretrained("gpt2-xl",cache_dir=cache_dir, torch_dtype=torch.float16)
    blocks = full.transformer.h
    start, end = partition_layers(len(blocks), world_size, rank)
    full.transformer.h = torch.nn.ModuleList(blocks[start:end])

    if rank != 0:
        del full.transformer.wte
        del full.transformer.wpe
        del full.transformer.drop
 
    def _id(name):
        setattr(full, name, torch.nn.Identity())
 
    if rank != 0:
        _id("embed_forward")
 
    full.to(device)
    return full

def send_tensor(tensor, dst):
    tensor = tensor.contiguous()
    dims = torch.tensor([tensor.dim()], dtype=torch.long)
    dist.send(dims, dst=dst)
    shape = torch.tensor(tensor.shape, dtype=torch.long)
    dist.send(shape, dst=dst)
    dist.send(tensor.cpu(), dst=dst)

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
    return buf
