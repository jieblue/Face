#!/bin/bash

# Get the IP and project path from command-line arguments
ip_address=$1
project_path=$2

# Run the gunicorn command with output redirection
nohup gunicorn face_app:app --threads 4 -b $ip_address --timeout 6000 --access-logfile $project_path/access.log >> /dev/null 2>&1 &