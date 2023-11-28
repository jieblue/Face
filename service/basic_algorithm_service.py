from typing import List

import cv2
import numpy as np
import av

from entity.file_entity import VideoFile
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
        key_frame = KeyFrame(np_frame, timestamp, frame_num, video_file.file_name)
        result.append(key_frame)
        frame_num = frame_num + 1

    return result


def av_frame2np(frame):
    img_frame = frame.to_image()
    np_frame = np.array(img_frame)
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    return np_frame