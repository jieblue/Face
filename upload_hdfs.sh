#!/bin/bash
video_id=$1
project_path=$2
# 海博正式环境的
#hdfs dfs -put $project_path/keyframes_faces/$video_id hdfs://192.168.101.151:9820/VIDEO_FACE_TEST/face/ &>/dev/null
#hdfs dfs -put $project_path/keyframes/$video_id hdfs://192.168.101.151:9820/VIDEO_FACE_TEST/video/ &>/dev/null


su - lczydw -c "hdfs dfs -put $project_path/keyframes_faces/$video_id hdfs://192.168.101.151:9820/VIDEO_FACE_TEST/face/ &>/dev/null"
su - lczydw -c "hdfs dfs -put $project_path/keyframes/$video_id hdfs://192.168.101.151:9820/VIDEO_FACE_TEST/video/ &>/dev/null"