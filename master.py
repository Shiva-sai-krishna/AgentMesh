import os
from utils import setup, cleanup, build_local_model_gpt2xl, send_tensor, recv_token, top_k_top_p_filtering, deterministic
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deterministic()
    rank, world, hf, max_tokens = setup()
    inputs = os.environ["INPUT_STRINGS"]
    inputs = inputs[1:len(inputs)-2]
    
    print(f"[Rank {rank}] World size: {world}")

    src = rank - 1
    if src < 0 : 
        src = world - 1
    dst = (rank + 1)% world
    print(f"[Rank {rank}] Source: {src}, Destination: {dst}")

    if world == 1: 
        model = GPT2LMHeadModel.from_pretrained("gpt2-xl", cache_dir=hf)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model.to(device).eval()

        prompt = inputs
        print("prompt", prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # greedy / sampling loop
        generated = inputs.input_ids
        past = None
        temperature = 0.8
        max_new = max_tokens
        for _ in range(max_new):
            outputs = model(generated[:, -1:], past_key_values=past, use_cache=True)
            logits, past = outputs.logits, outputs.past_key_values

            # apply top-k/top-p & temperature
            next_logits = logits[:, -1, :] / temperature
            filtered = top_k_top_p_filtering(next_logits, top_k=50, top_p=0.95)
            probs = filtered.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)


        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        print("Final:", result)
        with open("output.txt", "w") as file:
            file.write(result)
    
        
    else : 
        model = build_local_model_gpt2xl(rank, world, hf)
        print(f"[Rank {rank}] Model built successfully.")
        print(model)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        print(f"[Rank {rank}] Tokenizer loaded")  

        prompt = inputs
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

        max_new = max_tokens
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
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        with open("output.txt", "w") as file:
            file.write(result)
        print(f"[Rank {rank}] Final:", tokenizer.decode(generated[0], skip_special_tokens=True))
        cleanup()
