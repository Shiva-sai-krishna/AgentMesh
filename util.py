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
    print(f"[Rank {rank}] Initialized.")
    dist.barrier()  
    return rank

def cleanup():
    dist.destroy_process_group()
