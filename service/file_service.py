import os
from typing import List

import cv2

from entity.frame_entity import FaceKeyFrame
from utils import log_util

key_faces_path = './keyframes_faces/'

logger = log_util.get_logger(__name__)


def save_face_to_disk(face_frame_list: List[FaceKeyFrame]):
    for face_frame in face_frame_list:
        dir_path = os.path.join(key_faces_path, face_frame.tag + "/face/" + face_frame.file_name + "/")
        os.makedirs(dir_path, exist_ok=True)
        file_path = key_faces_path + face_frame.path_suffix
        cv2.imwrite(file_path, face_frame.face_frame)
        logger.info(f"save_face_to_disk success {file_path}")
