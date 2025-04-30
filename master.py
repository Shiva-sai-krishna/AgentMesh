import os
from utils import setup, cleanup, build_local_model_gpt2xl, send_tensor, recv_token
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank, world, hf = setup()
    print(f"[Rank {rank}] World size: {world}")

    src = rank - 1
    if src < 0 : 
        src = world - 1
    dst = (rank + 1)% world
    print(f"[Rank {rank}] Source: {src}, Destination: {dst}")

    if world == 1: 
        pass 
    else : 
        model = build_local_model_gpt2xl(rank, world, hf)
        print(f"[Rank {rank}] Model built successfully.")
        print(model)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"[Rank {rank}] Tokenizer loaded")  

        prompt = "The meaning of life is"
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        outputs = model(inputs, use_cache=True, return_dict=True, output_hidden_states=True)
        print(f"[Rank {rank}] Model forward pass completed")
        
        hidden  = outputs.hidden_states[-1]
        cache  = outputs.past_key_values

        send_tensor(hidden, dst=dst)
        for layer_kvs in cache:
            for part in layer_kvs:
                send_tensor(part, dst=dst)
        print(f"[Rank {rank}] Prefill hidden & cache sent")

        token = recv_token(src=src).to(device)
        generated = torch.cat([inputs, token], dim=-1)

        max_new = 50
        for step in range(max_new):
            print(f"[Rank {rank}] Generation step {step+1}/{max_new}")

            outputs = model(generated[:, -1:].to(device),past_key_values=cache,use_cache=True,return_dict=True,output_hidden_states=True)

            hidden = outputs.hidden_states[-1]
            cache = outputs.past_key_values     

            send_tensor(hidden, dst=dst)
            for layer_kvs in cache:
                k, v = layer_kvs
                send_tensor(k, dst=dst)
                send_tensor(v, dst=dst)
            print(f"[Rank {rank}] Prefill hidden & cache sent")

            token = recv_token(src=src).to(device)
            generated = torch.cat([generated, token], dim=-1)
            print(f"[Rank {rank}] Step {step+1} token: {token.item()}")

        print(f"[Rank {rank}] Final:", tokenizer.decode(generated[0], skip_special_tokens=True))
        cleanup()
