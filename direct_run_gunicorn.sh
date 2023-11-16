#!/bin/bash
# ./run_gunicorn.sh 192.168.100.19 /home/jack-studio/video-search/face-project/
# Get the IP and project path from command-line arguments
ip_address=$1
project_path=$2

# Run the gunicorn command with output redirection
gunicorn face_app:app -b "$ip_address" --timeout 6000 --access-logfile "$project_path/access.log"