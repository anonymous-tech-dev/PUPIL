#!/bin/bash

# Function to ping all active terminal windows inside this container
send_ping() {
    local message="$1"
    
    # 1. Log it to the screen/tmux session where the script is actually running
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $message"
    
    # 2. Broadcast the bell (\a) and message to every open terminal connected to this pod
    for term in /dev/pts/*; do
        # Check if it's a valid, writable terminal
        if [ -w "$term" ]; then
            echo -e "\a\n🚨 $message" > "$term" 2>/dev/null
        fi
    done
}

# Count initial unique GPUs in use
PREV_USED=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | wc -l)

echo "Monitoring started inside container $(hostname)."
echo "Currently $PREV_USED GPUs are in use."
echo "Checking every 5 minutes (300 seconds)..."

JOB_RAN=0

while true; do
    sleep 300 

    CURRENT_USED=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u | wc -l)

    if [ "$CURRENT_USED" -lt "$PREV_USED" ]; then
        send_ping "GPU Status Update: In-use GPUs dropped from $PREV_USED down to $CURRENT_USED."

        if [ "$CURRENT_USED" -eq 0 ] && [ "$JOB_RAN" -eq 0 ]; then
            send_ping "All 4 GPUs are free! Starting the data curation pipelines..."
            
            # Execute your scripts
            python /workspace/Pupil/dataset_curation/sft_data_curation/main.py
            python /workspace/Pupil/dataset_curation/sft_data_curation/main1.py
            
            send_ping "Pipelines finished executing."
            JOB_RAN=1
        fi
    fi

    # Reset if jobs start up again
    if [ "$CURRENT_USED" -gt 0 ]; then
        JOB_RAN=0
    fi

    PREV_USED=$CURRENT_USED
done