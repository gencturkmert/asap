#!/bin/bash

# Script to run a Python file with environment variables and auto-restart on crash

while true; do
    echo "Starting the Python script..."
    
    CUDA_VISIBLE_DEVICES="5,6,7" RAY_gcs_rpc_server_reconnect_timeout_s="600" bash -c "python test_eye.py"
    
    if [ $? -ne 0 ]; then
        echo "Script crashed. Restarting in 5 minutes..."
    else
        echo "Script exited normally. Restarting in 5 minutes..."
    fi

    sleep 300  # Wait for 5 minutes before restarting (300 seconds)
done