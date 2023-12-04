import os
from typing import List

import cv2
import numpy as np
import av

from entity.file_entity import VideoFile, ImageFile
from entity.frame_entity import KeyFrame
from utils import log_util

# Create a logger
logger = log_util.get_logger(__name__)


def extract_key_frame_list(video_file: VideoFile) -> List[KeyFrame]:
    container = av.open(video_file.file_path)

    result = []
    # Signal that we only want to look at keyframes.
    stream = container.streams.video[0]

    stream.codec_context.skip_frame = 'NONKEY'
    frames = container.decode(stream)
    frame_num = 1
    for frame in frames:
        timestamp = round(frame.pts * stream.time_base)
        np_frame = av_frame2np(frame)
        key_frame = KeyFrame(file_name=video_file.file_name, video_id=video_file.video_id, frame_num=frame_num,
                             timestamp=timestamp, frame=np_frame, tag=video_file.tag)
        result.append(key_frame)
        frame_num = frame_num + 1

    return result


def extract_image_key_frame_list(image_file: ImageFile) -> List[KeyFrame]:

    result = []
    frame_num = 10000
    for file in os.listdir(image_file.file_path):
        file_path = os.path.join(image_file.file_path, file)
        if not os.path.isfile(file_path):
            continue
        np_frame = cv2.imread(file_path)
        key_frame = KeyFrame(file_name=image_file.file_name, video_id=image_file.image_id, frame_num=1,
                             timestamp=0, frame=np_frame, tag=image_file.tag)
        result.append(key_frame)
        frame_num = frame_num + 1
    return result


def av_frame2np(frame):
    img_frame = frame.to_image()
    np_frame = np.array(img_frame)
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    return np_frame
