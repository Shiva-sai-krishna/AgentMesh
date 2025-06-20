import os
import socket
import yaml
import paramiko
import threading
import sys
import os
import time


def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ssh_run_command(ip, username, password, command):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=ip, username=username, password=password)

    stdin, stdout, stderr = client.exec_command(command)
    for line in stdout:
        print(f"[{ip}] {line.strip()}")
    for line in stderr:
        print(f"[{ip} ERROR] {line.strip()}")
    
    client.close()

def setup_worker(device_info, master_addr, master_port, world_size, rank, maxTokens):
    ip = device_info["ip"]
    username = device_info["username"]
    password = device_info["password"]
    work_dir = device_info["work_dir"]
    python_env = device_info["python_env"]
    hf_cache = device_info["hf_cache"]

    session_name = f"agentmesh"
    
    full_command = f"""
    cd {work_dir} && \
    export MASTER_ADDR={master_addr} && \
    export MASTER_PORT={master_port} && \
    export WORLD_SIZE={world_size} && \
    export RANK={rank} && \
    export HF_HOME={hf_cache} && \
    export MAX_TOKENS={maxTokens} && \
    source {python_env} && \
    python3 worker.py
    """

    command = f"""
    tmux has-session -t {session_name} || tmux new-session -d -s {session_name} && \
    tmux send-keys -t {session_name} "{full_command}" C-m"""



    ssh_run_command(ip, username, password, command)

def setup_master(device_info, master_addr, master_port, world_size, rank, max_tokens, input_strings):
    ip = device_info["ip"]
    username = device_info["username"]
    password = device_info["password"]
    work_dir = device_info["work_dir"]
    python_env = device_info["python_env"]
    hf_cache = device_info["hf_cache"]

    session_name = "agentmesh"

    # Get local IP
    local_ip = socket.gethostbyname(socket.gethostname())

    full_command = f"""
    cd {work_dir} && \
    export MASTER_ADDR={master_addr} && \
    export MASTER_PORT={master_port} && \
    export WORLD_SIZE={world_size} && \
    export RANK={rank} && \
    export HF_HOME={hf_cache} && \
    export MAX_TOKENS={max_tokens} && \
    export INPUT_STRINGS={input_strings} && \
    source {python_env} && \
    python3 master.py
    """

    command = f"""
    tmux has-session -t {session_name} || tmux new-session -d -s {session_name} && \
    tmux send-keys -t {session_name} "{full_command}" C-m"""

    if local_ip == master_addr:
        # Run locally
        print("[Client] Running master.py locally in tmux.")
        os.system(command)
    else:
        # SSH and run remotely
        print("[Client] SSH into master to run master.py in tmux.")
        ssh_run_command(ip, username, password, command)

def main():
    config = load_config()
    master_addr = config["master"]["ip"]
    master_port = config["master"]["port"]
    devices = config["workers"]
    max_tokens = config.get('maxTokens')
    input_strings = config.get('inputStrings')

    if not devices:
        devices = []

    world_size = len(devices) + 1  
    threads = []

    # Setup workers (rank != 0)
    for rank, device in enumerate(devices, start=1):
        thread = threading.Thread(target=setup_worker, args=(device, master_addr, master_port, world_size, rank, max_tokens))
        thread.start()
        threads.append(thread)

    # Setup master (rank 0)
    master_device_info = {
        "ip": master_addr,
        "username": config["master"]["username"],
        "password": config["master"]["password"],
        "work_dir": config["master"]["work_dir"],
        "python_env": config["master"]["python_env"],
        "hf_cache": config["master"]["hf_cache"]
    }

    setup_master(master_device_info, master_addr, master_port, world_size, rank=0, max_tokens=max_tokens, input_strings=input_strings)

if __name__ == "__main__":
    main()
    


