#!/bin/bash

set -euo pipefail  # Exit if any error, undefined variable, or pipeline fails

CONFIG_FILE=$1

# Read configuration
MASTER_ADDR=$(jq -r '.master_ip' "$CONFIG_FILE")
MASTER_PORT=$(jq -r '.master_port' "$CONFIG_FILE")
CODE_DIR=$(jq -r '.code_dir' "$CONFIG_FILE")
ENV_PATH=$(jq -r '.env_path' "$CONFIG_FILE")
MODEL_DIR=$(jq -r '.model_dir' "$CONFIG_FILE")

# Function to execute command on remote device using tmux
run_remote_tmux() {
    local rank=$1
    local ip=$2
    local user=$3
    local pass=$4
    
    session_name="node${rank}_session"

    sshpass -p "$pass" ssh -o StrictHostKeyChecking=no "$user@$ip" "
        tmux kill-session -t $session_name 2>/dev/null || true
        tmux new-session -d -s $session_name '
            export MASTER_ADDR=$MASTER_ADDR;
            export MASTER_PORT=$MASTER_PORT;
            export RANK=$rank;
            export WORLD_SIZE=3;  # You can adjust based on device count dynamically
            source $ENV_PATH;
            cd $CODE_DIR;
            python node${rank}.py --model_dir $MODEL_DIR > node${rank}.log 2>&1
        '
    "
}

# Start worker nodes first (rank >= 1)
echo "Starting worker nodes..."
jq -c '.devices[]' "$CONFIG_FILE" | while read -r device; do
    rank=$(jq -r '.rank' <<< "$device")
    ip=$(jq -r '.ip' <<< "$device")
    user=$(jq -r '.user' <<< "$device")
    pass=$(jq -r '.password' <<< "$device")

    if [ "$rank" -ne 0 ]; then
        echo "Starting rank $rank on $ip"
        run_remote_tmux "$rank" "$ip" "$user" "$pass"
    fi
done

# Start master node (rank 0)
echo "Starting master node..."
master_device=$(jq -c '.devices[] | select(.rank == 0)' "$CONFIG_FILE")
rank=$(jq -r '.rank' <<< "$master_device")
ip=$(jq -r '.ip' <<< "$master_device")
user=$(jq -r '.user' <<< "$master_device")
pass=$(jq -r '.password' <<< "$master_device")

run_remote_tmux "$rank" "$ip" "$user" "$pass"

echo "All nodes started successfully in tmux sessions."

echo ""
echo "To attach to a session on a node, use:"
echo "sshpass -p <password> ssh <user>@<ip> 'tmux attach -t node<rank>_session'"
echo ""
