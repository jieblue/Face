#!/bin/bash

# Specify the path to the text file to monitor
#file_to_monitor="/home/ubuntu/jack_test_workspace/shell_test/video_id_file.txt"
#
## Initialize the last known line count
#last_line_count=$(wc -l < "$file_to_monitor")
#
## Monitor the file for changes in real-time
#tail -n +$last_line_count -f "$file_to_monitor" | while read -r new_line
#do
#    echo "New line added: $new_line"
#    /home/jack-studio/video-search/face-project/upload_hdfs.sh $new_line /home/jack-studio/video-search/face-project/
#    echo "Upload to hdfs done"
#done
# nohup ./monitor_changes.sh > changenohup.out 2>&1 &
# 正式环境的监控脚本
# Specify the path to the text file to monitor
file_to_monitor="/home/gpu6/video-face-project/need_upload_hdfs/video_id_file.txt"

# Initialize the last known line count
last_line_count=$(wc -l < "$file_to_monitor")

# Monitor the file for changes in real-time
tail -n +$last_line_count -f "$file_to_monitor" | while read -r new_line
do
    echo "New line added: $new_line"
    /home/gpu6/video-face-project/upload_hdfs.sh $new_line /home/gpu6/video-face-project/
    echo "Upload to hdfs done"
    sleep 1
done