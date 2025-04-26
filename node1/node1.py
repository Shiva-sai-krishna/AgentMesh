# node1.py
import os, torch
from transformers import GPT2LMHeadModel
import torch.distributed as dist

# --- Common setup ---
def setup():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"])
    )
    rank = dist.get_rank()
    print(f"[Rank {rank}] Process group initialized.")
    dist.barrier()
    print(f"[Rank {rank}] Setup complete.")
    return rank

# --- Load & split model half (layers 6–11 + head) ---
def load_half_model():
    print("[Rank 1] Loading GPT-2 small...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Keep only layers 6–11
    num_layers = model.config.n_layer
    for _ in range(num_layers//2):
        del model.transformer.h[0]
    # Keep lm_head; remove embeddings
    del model.transformer.wte
    del model.transformer.wpe
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print("[Rank 1] Model loaded and truncated to layers 6–11 + head.")
    return model, device

# --- Receive hidden + metadata, run stage 2, sample next token ---
def stage2_and_sample(model, device, do_sample=False, top_k=50, top_p=0.95, temp=1.0):
    # 1. Receive shape metadata
    shape_meta = torch.zeros(3, dtype=torch.long)
    dist.recv(shape_meta, src=0)
    shape = tuple(shape_meta.tolist())
    print(f"[Rank 1] Received hidden shape metadata: {shape}")

    # 2. Allocate buffer & recv hidden
    hidden_cpu = torch.zeros(shape, dtype=torch.float32)
    dist.recv(hidden_cpu, src=0)
    print(f"[Rank 1] Received hidden tensor ({hidden_cpu.numel()*4} bytes)")

    # 3. Move to GPU & run layers 6–11 + head
    hidden = hidden_cpu.to(device)
    with torch.no_grad():
        for idx, block in enumerate(model.transformer.h):
            hidden = block(hidden)[0]
            print(f"[Rank 1]  → Completed block {idx + (model.config.n_layer//2)}")
        logits = model.lm_head(hidden)
    logits_cpu = logits.cpu()

    # 4. Sample next token from last step’s logits
    step_logits = logits_cpu[:, -1, :]
    if do_sample:
        probs = torch.softmax(step_logits / temp, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
    else:
        next_id = torch.argmax(step_logits, dim=-1, keepdim=True)
    print(f"[Rank 1] Selected next token ID {next_id.item()}")

    # 5. Send next token back
    dist.send(next_id, dst=0)
    print(f"[Rank 1] Sent next token ID to Rank 0")

# --- Main execution ---
if __name__ == "__main__":
    rank = setup()
    model, device = load_half_model()
    # Perform continuous receive → compute → send until node0 shuts down
    # Here we expect 1 prefill + 10 generate steps = 11 calls
    total_steps = 1 + 10
    for step in range(total_steps):
        print(f"[Rank 1] Waiting for step {step+1}/{total_steps}")
        stage2_and_sample(model, device, do_sample=False)  # greedy decoding
    dist.destroy_process_group()
    print("[Rank 1] Done.")
