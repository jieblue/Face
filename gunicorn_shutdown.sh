#!/bin/bash


#!/bin/bash
# execute example: ./shutdown_script.sh face_app
# Get the project path and project ID from command-line arguments
project_id=$1

# Find the process ID (PID) of the gunicorn process
# Find the process IDs (PIDs) of the gunicorn processes
pids=$(pgrep -f "gunicorn $project_id:app")

# If PIDs exist, terminate the gunicorn processes
if [ ! -z "$pids" ]; then
    for pid in $pids; do
        kill "$pid"
        echo "Gunicorn process (PID: $pid) terminated."
    done
else
    echo "Gunicorn processes not found."
fi

echo "Shutdown script completed."