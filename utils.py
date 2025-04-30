import os
import torch.distributed as dist

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

def cleanup():
    dist.destroy_process_group()
