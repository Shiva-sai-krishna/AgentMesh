#!/bin/bash
# CSV file to store the logs
LOG_FILE="/media/ssd/AgentMesh/jtop_log.csv"
 
# Write CSV header
echo "Timestamp, RAM_Usage, Swap_Usage, CPU_Util_Core_0, CPU_Util_Core_1, CPU_Util_Core_2, CPU_Util_Core_3, CPU_Util_Core_4, CPU_Util_Core_5, CPU_Util_Core_6, CPU_Util_Core_7, CPU_Util_Core_8, CPU_Util_Core_9, CPU_Util_Core_10, CPU_Util_Core_11, GPU_Utilization, CPU_Temperature, GPU_Temperature, CPU_Power_Inst, GPU_Power_Inst, System_Power_Inst, Extra Power" > $LOG_FILE
 
# Collect data every 1 second
while true; do
    # Get current timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
 
    # Run tegrastats and extract desired stats
    stats=$(sudo tegrastats --logfile /dev/stdout | head -n 1)
 
    # Extract RAM usage
    ram_usage=$(echo "$stats" | grep -oP "RAM \K[0-9]+/[0-9]+MB")
 
    # Extract Swap usage
    swap_usage=$(echo "$stats" | grep -oP "SWAP \K[0-9]+/[0-9]+MB")
 
    # Extract CPU utilization for all 12 cores
    cpu_utilizations=$(echo "$stats" | grep -oP "CPU \[\K([0-9]+%@\d+,?)+")
    cpu_utils=$(echo "$cpu_utilizations" | grep -oP "[0-9]+(?=%)")
 
    # Separate CPU utilizations into individual columns
    cpu_util_array=($(echo "$cpu_utils"))
    cpu_util_row=$(IFS=,; echo "${cpu_util_array[*]}")
 
    # Extract GPU utilization
    gpu_util=$(echo "$stats" | grep -oP "GR3D_FREQ \K[0-9]+%")
 
    # Extract CPU temperature
    cpu_temp=$(echo "$stats" | grep -oP "cpu@\K[0-9]+(\.[0-9]+)?(?=C)")
 
    # Extract GPU temperature
    gpu_temp=$(echo "$stats" | grep -oP "gpu@\K[0-9]+(\.[0-9]+)?(?=C)")
 
    # Extract instantaneous CPU and GPU power consumption
    cpu_power_inst=$(echo "$stats" | grep -oP "VDD_CPU_CV \K[0-9]+(?=mW)")
    gpu_power_inst=$(echo "$stats" | grep -oP "VDD_GPU_SOC \K[0-9]+(?=mW)")
 
    # Extract system power consumption
    sys_power_inst=$(echo "$stats" | grep -oP "VIN_SYS_5V0 \K[0-9]+(?=mW)")
 
    extra_power=$(echo "$stats" | grep -oP "VDDQ_VDD2_1V8AO \K[0-9]+(?=mW)")
 
    # Append the data to the CSV file
    echo "$timestamp, $ram_usage, $swap_usage, $cpu_util_row, $gpu_util, $cpu_temp, $gpu_temp, $cpu_power_inst, $gpu_power_inst, $sys_power_inst, $extra_power" >> $LOG_FILE
done
 