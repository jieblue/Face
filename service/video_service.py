from model.model_video import Video_Model
import av
import os
import shutil

import cv2
import numpy as np



def extract_video(video_path, unique_filename):
    """
    Extracts faces from a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        List[ndarray]: A list of NumPy arrays, each representing a face image.
    """
    # return a list of nparray(bgr image)
    # print(video_path)
    try:
        container = av.open(video_path)
    except:
        return None

    result = []
    # Signal that we only want to look at keyframes.
    stream = container.streams.video[0]

    stream.codec_context.skip_frame = 'NONKEY'
    frames = container.decode(stream)
    frame_num = 1
    for frame in frames:
        # print(stream.time_base)
        timestamp = round(frame.pts * stream.time_base)
        np_frame = av_frame2np(frame)
        result.append({
            'frame': np_frame,
            'timestamp': timestamp,
            'frame_num': frame_num,
            'unique_filename': unique_filename
        })
        frame_num = frame_num + 1


    return result


def av_frame2np(frame):
    img_frame = frame.to_image()
    np_frame = np.array(img_frame)
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    return np_frame

def get_frames_feature(model: Video_Model, key_frames_info_list):
    frame_feature_list = []
    for key_frames_info in key_frames_info_list:
        frame_file_path = key_frames_info['frame_file_path']
        key_frames_info['frame_feature'] = model.get_frame_embedding(frame_file_path)
        frame_feature_list.append(key_frames_info)

    return frame_feature_list

def save_frame_to_disk(key_frames, key_frames_path, unique_filename):
    """
    Save key frames to disk.

    Args:
        key_frames (list): A list of key frames.
        key_frames_path (str): The path to the directory where the key frames will be saved.
        unique_filename (str): A unique filename for the key frames.

    Returns:
        list: A list of file paths for the saved key frames.
    """
    # Write key frames to disk
    for i, keyframe in enumerate(key_frames):
        frame = keyframe['frame']
        timestamp = keyframe['timestamp']
        dir_path = os.path.join(key_frames_path, unique_filename)
        os.makedirs(dir_path, exist_ok=True)
        frame_embedding_id = keyframe['unique_filename'] + "_" +str(keyframe['frame_num']) + "_" + str(keyframe["timestamp"])
        keyframe['frame_embedding_id'] = frame_embedding_id
        file_name = f"{frame_embedding_id}.jpg"
        file_path = os.path.join(dir_path, file_name)
        keyframe['frame_file_name'] = file_name
        keyframe['frame_file_path'] = file_path
        cv2.imwrite(file_path, frame)