import gc
import math
import os
from typing import List

import cv2
import numpy as np
import av

from entity.file_entity import VideoFile, ImageFile
from entity.frame_entity import KeyFrame
from service.visual_algorithm_service import video_model
from utils import log_util

# Create a logger
logger = log_util.get_logger(__name__)


def extract_key_frame_list(video_file: VideoFile) -> List[KeyFrame]:
    container = av.open(video_file.file_path)

    result = []
    # Signal that we only want to look at keyframes.
    stream = container.streams.video[0]

    stream.codec_context.skip_frame = 'NONINTRA'
    frames = container.decode(stream)
    frame_num = 1
    # 关键帧提取
    for frame in frames:
        timestamp = round(frame.pts * stream.time_base)
        np_frame = av_frame2np(frame)
        key_frame = KeyFrame(file_name=video_file.file_name, video_id=video_file.video_id, frame_num=frame_num,
                             timestamp=timestamp, frame=np_frame, frame_stream=frame, tag=video_file.tag)
        result.append(key_frame)
        frame_num = frame_num + 1

    # logger.info(f"Extracted {len(frames)} frames from {video_file.file_name}")
    # start_pos = 0
    # end_pos = 0
    # length = len(frames)
    # while end_pos < length:
    #     if start_pos == end_pos:
    #         end_pos = end_pos + 1
    #     else:
    #         start_frame = frames[start_pos]
    #         end_frame = frames[end_pos]
    #         if keyframe_similarity(start_frame, end_frame) > 0.9:
    #             end_pos = end_pos + 1
    #         else:
    #             np_frame = av_frame2np(start_frame)
    #             timestamp = round(start_frame.pts * stream.time_base)
    #             key_frame = KeyFrame(file_name=video_file.file_name, video_id=video_file.video_id,
    #                                  frame_num=frame_num,
    #                                  timestamp=timestamp, frame=np_frame, frame_stream=start_frame,
    #                                  tag=video_file.tag)
    #             result.append(key_frame)
    #             frame_num = frame_num + 1
    #             start_pos = end_pos
    #             end_pos = end_pos + 1
    # logger.info(f"Extracted {len(result)} key frames from {video_file.file_name}")

    #gc.collect()释放frames占用的内存
    gc.collect()
    return result


def keyframe_similarity(frame_x, frame_y):
    distance = np.linalg.norm(np.array(frame_x) - np.array(frame_y))
    return normalized_euclidean_distance(distance)


def normalized_euclidean_distance(L2, dim=512):
    dim_sqrt = math.sqrt(dim)
    return 1 / (1 + L2 / dim_sqrt)


def extract_image_key_frame_list(image_file: ImageFile) -> List[KeyFrame]:
    result = []
    frame_num = 10000
    for file in os.listdir(image_file.file_path):
        file_path = os.path.join(image_file.file_path, file)
        if not os.path.isfile(file_path) or file.startswith("."):
            continue
        np_frame = cv2.imread(file_path)
        key_frame = KeyFrame(file_name=image_file.file_name, video_id=image_file.image_id, frame_num=frame_num,
                             timestamp=0, frame=np_frame, frame_stream=file_path, tag=image_file.tag)
        result.append(key_frame)
        frame_num = frame_num + 1
    return result


def av_frame2np(frame):
    img_frame = frame.to_image()
    np_frame = np.array(img_frame)
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    return np_frame
