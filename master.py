import os
from utils import setup, cleanup 

if __name__ == "__main__":
    rank = setup()
    print(f"[INFO] Rank {rank} Master node setup complete")
    cleanup()
