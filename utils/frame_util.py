import av
import os
import shutil

import cv2
import numpy as np



# 定义提取视频关键帧的函数
def extract_video(video_path):
    # return a list of nparray(bgr image)
    # print(video_path)
    try:
        content = av.datasets.curated(video_path)
        container = av.open(content)
    except:
        return None

    result = []
    # Signal that we only want to look at keyframes.
    stream = container.streams.video[0]

    # NONKEY 跳过非关键帧
    # NONINTRA 跳过非帧内帧
    stream.codec_context.skip_frame = 'NONINTRA'
    frames = container.decode(stream)
    for frame in frames:
        # print(stream.time_base)
        timestamp = round(frame.pts * stream.time_base)
        np_frame = av_frame2np(frame)
        result.append({
            'frame': np_frame,
            'timestamp': timestamp
        })


    return result


def av_frame2np(frame):
    img_frame = frame.to_image()
    np_frame = np.array(img_frame)
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    return np_frame


