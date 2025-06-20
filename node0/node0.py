# node0.py
import os, torch, torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

# --- Load & split model half (layers 0–5) ---
def load_half_model():
    print("[Rank 0] Loading GPT-2 small...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Keep only layers 0–5
    num_layers = model.config.n_layer  # 12
    for i in reversed(range(num_layers//2, num_layers)):
        del model.transformer.h[i]
    # Remove head; Node1 will apply it
    del model.lm_head
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print("[Rank 0] Model loaded and truncated to layers 0–5.")
    return model, device

# --- Prefill on Node0 ---
def prefill_stage1(model, device, input_ids):
    # 1. Token embeddings
    token_embeds = model.transformer.wte(input_ids)  # input_ids: LongTensor
    # 2. Position embeddings
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    pos_embeds = model.transformer.wpe(position_ids)
    # 3. Combine
    hidden = token_embeds + pos_embeds
    # 4. Run through layers 0–5
    with torch.no_grad():
        for idx, block in enumerate(model.transformer.h):
            hidden = block(hidden)[0]
            print(f"[Rank 0] Completed block {idx}")
    return hidden

# --- Send hidden + metadata ---
def send_intermediate(hidden):
    print(f"[Rank 0] >>> send_intermediate called for new hidden")
    hidden_cpu = hidden.cpu()
    # Send shape metadata
    shape = torch.tensor(hidden_cpu.shape, dtype=torch.long)
    dist.send(shape, dst=1)
    print(f"[Rank 0] Sent hidden shape metadata {tuple(hidden_cpu.shape)} to Rank 1")
    # Send actual hidden
    dist.send(hidden_cpu, dst=1)
    print(f"[Rank 0] Sent hidden tensor to Rank 1 ({hidden_cpu.numel()*4} bytes)")

# --- Receive next-token IDs during generation ---
def recv_token():
    token_tensor = torch.zeros((1,1), dtype=torch.long)
    dist.recv(token_tensor, src=1)
    next_id = token_tensor.item()
    print(f"[Rank 0] Received next token ID {next_id} from Rank 1")
    return token_tensor

# --- Main execution ---
if __name__ == "__main__":
    rank = setup()
    model, device = load_half_model()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 1. Prefill pass
    prompt = "Hello, world!"
    print(f"[Rank 0] Tokenizing prompt: {prompt!r}")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    hidden = prefill_stage1(model, device, inputs)

    # --- Prefill send & immediate receive of first token ⮕ ---
    send_intermediate(hidden)
    first_token = recv_token().to(device)  # ⮕ ensure the prefill-response is consumed
    generated_ids = torch.cat([inputs, first_token], dim=-1)
    print(f"[Rank 0] Prefill complete; received first token ID {first_token.item()}")

    # 2. Generation loop for next (max_new_tokens - 1) tokens
    max_new_tokens = 10
    # We already consumed one token above, so loop max_new_tokens-1 times:
    remaining = max_new_tokens - 1
    for step in range(remaining):
        print(f"[Rank 0] Generation step {step+1}/{remaining}")

        # Embed last token only
        last_id = generated_ids[:, -1].unsqueeze(-1)
        token_embeds = model.transformer.wte(last_id)
        position_ids = torch.tensor([[generated_ids.size(1)-1]], dtype=torch.long, device=device)
        pos_embeds = model.transformer.wpe(position_ids)
        hidden = token_embeds + pos_embeds
        for idx, block in enumerate(model.transformer.h):
            hidden = block(hidden)[0]
            print(f"[Rank 0] Completed block {idx}")

        # Send & receive for each subsequent generation step
        send_intermediate(hidden)
        next_token = recv_token().to(device)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    # 3. Decode and print full sequence
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"[Rank 0] Final generated text:\n{result}")

    dist.destroy_process_group()
    print("[Rank 0] Done.")
