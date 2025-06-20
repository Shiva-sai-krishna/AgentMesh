import os
from utils import setup, cleanup, build_local_model_gpt2xl, send_tensor, recv_token, top_k_top_p_filtering, deterministic, append_to_csv
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
cache_dir = "/media/ssd/huggingface"
import torch
import csv
import time
from datetime import datetime

COMP2COMM_LOGFILE = "/media/ssd/AgentMesh/logs/comp2comm.csv"

file = open(COMP2COMM_LOGFILE, "w", newline="")
writer = csv.writer(file)
writer.writerow(["State","Rank","Type","Remark","Time"])

DATA_TRANSFER_LOGFILE = "/media/ssd/AgentMesh/logs/data_transfer.csv"
data_transfer_file = open(DATA_TRANSFER_LOGFILE, "w", newline="")
data_transfer_writer = csv.writer(data_transfer_file)
data_transfer_writer.writerow(["Operation", "Destination", "Label", "Shape", "Size (bytes)"])


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

    # we will focus on multi-device so ignore this
    if world == 1: 
        model = GPT2LMHeadModel.from_pretrained("gpt2-large", cache_dir=hf)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", cache_dir=hf)
        model.to(device).eval()

        prompt = inputs
        print("prompt", prompt)
        start_time = time.time()
        formatted_start_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print("Start Time: ", formatted_start_time)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # greedy / sampling loop
        generated = inputs.input_ids
        past = None
        temperature = 0.8
        max_new = max_tokens
        for _ in range(max_new):
            outputs = model(generated[:, -1:], past_key_values=past,
                             use_cache=True
                             )
            logits, past = outputs.logits, outputs.past_key_values

            # apply top-k/top-p & temperature
            next_logits = logits[:, -1, :] / temperature
            filtered = top_k_top_p_filtering(next_logits, top_k=50, top_p=0.95)
            probs = filtered.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)


        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        print("Final:", result)
        end_time = time.time()
        formatted_end_time = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        print("End Time:", formatted_end_time)
        elapsed_time = end_time - start_time
        print("Time taken on single node: ", elapsed_time)
    
    else : 
        model = build_local_model_gpt2xl(rank, world, hf)
        model.to(device).eval()
        print(model.get_memory_footprint()/1e9, "GB")
        print(f"[Rank {rank}] Model built successfully.")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=cache_dir)
        print(f"[Rank {rank}] Tokenizer loaded")  
        prompt = inputs
        
        # 64 tokens
        prompt = "Ronaldo began his senior career with Sporting CP, before signing with Manchester United in 2003, winning the FA Cup in his first season. He went on to become a star player at United, as they won three consecutive Premier League titles, the Champions League and the FIFA Club World Cup; his 2007-08 season earned"

        # 128 tokens
        # prompt = "Zebras are primarily grazers and can subsist on lower-quality vegetation. They are preyed on mainly by lions, and typically flee when threatened but also bite and kick. Zebra species differ in social behaviour, with plains and mountain zebra living in stable harems consisting of an adult male or stallion, several adult females or mares, and their young or foals; while Grévy's zebra live alone or in loosely associated herds. In harem-holding species, adult females mate only with their harem stallion, while male Grévy's zebras establish territories which attract females" 

        # 256 tokens
        # prompt = "Ronaldo began his senior career with Sporting CP, before signing with Manchester United in 2003, winning the FA Cup in his first season. He went on to become a star player at United, as they won three consecutive Premier League titles, the Champions League and the FIFA Club World Cup; his 2007–08 season earned him his first Ballon d'Or, aged 23. Ronaldo was the subject of the then-most expensive association football transfer when he signed for Real Madrid in 2009 in a transfer worth €94 million (£80 million). He was integral to Madrid becoming a dominant force again, as they won four Champions Leagues from 2014 to 2018, including La Décima. He won back-to-back Ballon d'Or awards in 2013 and 2014, and again in 2016 and 2017, and was runner-up three times behind Lionel Messi, his perceived career rival. He also became the club's all-time top goalscorer and finished as the Champions League's top scorer for six consecutive seasons between 2012 and 2018. With Madrid, Ronaldo's additional titles include two La Liga titles, including a record-breaking title win in 2011-12, and two Copas del Rey. In 2018, following issues with the Madrid hierarchy, Ronaldo made a surprise"

        # 512 tokens
        # prompt = "Ronaldo began his senior career with Sporting CP, before signing with Manchester United in 2003, winning the FA Cup in his first season. He went on to become a star player at United, as they won three consecutive Premier League titles, the Champions League and the FIFA Club World Cup; his 2007–08 season earned him his first Ballon d'Or, aged 23. Ronaldo was the subject of the then-most expensive association football transfer when he signed for Real Madrid in 2009 in a transfer worth €94 million (£80 million). He was integral to Madrid becoming a dominant force again, as they won four Champions Leagues from 2014 to 2018, including La Décima. He won back-to-back Ballon d'Or awards in 2013 and 2014, and again in 2016 and 2017, and was runner-up three times behind Lionel Messi, his perceived career rival. He also became the club's all-time top goalscorer and finished as the Champions League's top scorer for six consecutive seasons between 2012 and 2018. With Madrid, Ronaldo's additional titles include two La Liga titles, including a record-breaking title win in 2011-12, and two Copas del Rey. In 2018, following issues with the Madrid hierarchy, Ronaldo made a surprise transfer to Juventus in a transfer worth an initial €100 million (£88 million). He won several trophies in Italy, including two Serie A titles and a Coppa Italia, and broke several records for Juventus. He returned to United in 2021; despite a collectively disappointing season, Ronaldo's individual performances earned him being included in the PFA Team of the Year at 37 years old. His contract was terminated in 2022 due to a fall out with the newly appointed manager, and in 2023 he signed for Al-Nassr, a move that has since been widely credited for revolutionising football in Saudi Arabia.Ronaldo made his international debut for Portugal in 2003 at the age of 18 and has earned more than 200 caps, making him history's most-capped male player.[8] Ronaldo has played in eleven major tournaments and scored in ten; he scored his first international goal at Euro 2004, where he helped Portugal reach the final and subsequently made the team of the tournament. In the 2006 FIFA World Cup, his first World Cup, he was a focal part to Portugal ultimately finishing in fourth place. He assumed captaincy of the national team in July 2008 ahead of Euro 2008;four years later, at Euro 2012, he was named to the team of the tournament. In "
        
        E2E_start_time = time.time()
        tokenizer_start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        tokenizer_end_time = time.time()
        tokenizer_duration = tokenizer_end_time - tokenizer_start_time
        append_to_csv(writer, ["Tokenization",f"{rank}","Compute","",f"{tokenizer_duration:.3f}"], file) 
        device1_prefill_start = time.time()
        outputs = model(inputs, 
                        use_cache=True, 
                        return_dict=True, 
                        output_hidden_states=True)
        device1_prefill_end = time.time()
        device1_prefill_duration = device1_prefill_end - device1_prefill_start
        append_to_csv(writer, ["Prefill",f"{rank}","Compute","",f"{device1_prefill_duration:.3f}"], file)    
        print(f"[Rank {rank}] Model forward pass completed")
        
        hidden  = outputs.hidden_states[-1]
        local_cache  = outputs.past_key_values

        #send hidden
        send_hidden_start = time.time()
        send_tensor(hidden, dst=dst, writer=data_transfer_writer, label="Prefill Hidden") 
        send_hidden_end = time.time()
        send_hidden_duration = send_hidden_end - send_hidden_start
        append_to_csv(writer, ["Prefill",f"{rank}","Communication","Send Hidden",f"{send_hidden_duration:.3f}"],file)
    
        # send cache
        # send_cache_start = time.time()
        # total_kv_bytes = 0
        # for i, layer_kvs in enumerate(cache):
        #     for j, part in enumerate(layer_kvs):
        #         label = f"Prefill KV Layer {i} {'K' if j==0 else 'V'}"
        #         send_tensor(part, dst=dst, writer=data_transfer_writer, label=label)
        # send_cache_end = time.time()
        # send_cache_duration = send_cache_end - send_cache_start
        # append_to_csv(writer, ["Prefill",f"{rank}","Communication","Send Cache",f"{send_cache_duration:.3f}"],file)
        
        print(f"[Rank {rank}] Prefill hidden sent")

        token = recv_token(src=src).to(device)
        generated = torch.cat([inputs, token], dim=-1)

        # prefill ends

        # decode starts
        max_new = max_tokens 
        for step in range(max_new): 
            print(f"[Rank {rank}] Generation step {step+1}/{max_new}")
            device1_decode_start = time.time()
            outputs = model(generated[:, -1:].to(device),past_key_values=local_cache,
                            use_cache=True,
                            return_dict=True,
                            output_hidden_states=True)
            device1_decode_end = time.time()
            device1_decode_duration = device1_decode_end - device1_decode_start
            append_to_csv(writer, [f"Decode {step + 1}",f"{rank}","Compute","",f"{device1_decode_duration:.3f}"],file)
            hidden = outputs.hidden_states[-1]
            local_cache = outputs.past_key_values

            
            send_hidden_start = time.time()
            send_tensor(hidden, dst=dst, writer=data_transfer_writer, label=f"Decode {step} Hidden")
            send_hidden_end = time.time()
            send_hidden_duration = send_hidden_end - send_hidden_start
            append_to_csv(writer, [f"Decode {step + 1}",f"{rank}","Communication","Send Hidden",f"{send_hidden_duration:.3f}"],file)

            # send_cache_start = time.time()
            # # for layer_kvs in cache:
            # #     k, v = layer_kvs
            # #     send_tensor(k, dst=dst)
            # #     send_tensor(v, dst=dst)
            # for i, layer_kvs in enumerate(cache):
            #     for j, part in enumerate(layer_kvs):
            #         label = f"Decode {step+1} KV Layer {i} {'K' if j==0 else 'V'}"
            #         send_tensor(part, dst=dst, writer=data_transfer_writer, label=label)
            # send_cache_end = time.time()
            # send_cache_duration = send_cache_end - send_cache_start
            # append_to_csv(writer, [f"Decode {step + 1}",f"{rank}","Communication","Send Cache",f"{send_cache_duration:.3f}"],file)
            print(f"[Rank {rank}] Decode {step+1} hidden sent")

            token = recv_token(src=src).to(device)
            generated = torch.cat([generated, token], dim=-1)
            print(f"[Rank {rank}] Step {step+1} token: {token.item()}")
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        E2E_end_time = time.time()
        print(f"[Rank {rank}] Final:", tokenizer.decode(generated[0], skip_special_tokens=True))
        
        E2E_elapsed_time = E2E_end_time - E2E_start_time
        print(f"[Rank {rank}] Time taken on distributed {world}: ", E2E_elapsed_time)
        cleanup()
