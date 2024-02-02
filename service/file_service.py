import os
from typing import List

import cv2
import requests

from entity.frame_entity import FaceKeyFrame, KeyFrame
from utils import log_util

key_faces_path = './keyframes_faces/'
key_frame_path = './keyframes/'

logger = log_util.get_logger(__name__)


def save_face_to_disk(face_frame_list: List[FaceKeyFrame]):
    for face_frame in face_frame_list:
        dir_path = os.path.join(key_faces_path, face_frame.tag + "/face/" + face_frame.file_name + "/")
        os.makedirs(dir_path, exist_ok=True)
        file_path = key_faces_path + face_frame.path_suffix
        cv2.imwrite(file_path, face_frame.face_frame)
        logger.info(f"save_face_to_disk success {file_path}")


def save_frame_to_disk(key_frame_list: List[KeyFrame]):
    for face_frame in key_frame_list:
        dir_path = os.path.join(key_frame_path, face_frame.tag + "/key_frame/" + face_frame.file_name + "/")
        os.makedirs(dir_path, exist_ok=True)
        file_path = key_frame_path + face_frame.path_suffix
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
        # 保存压缩后的图像
        cv2.imwrite(file_path, face_frame.frame, compression_params)

        logger.info(f"save_face_to_disk success {file_path}")


# 写入压缩图片
def write_compression_img(save_path, img, bit_rate=50, size=None):
    # save_path 存储路径  img输入图片
    # bit_rate压缩比特率 默认50，降低约一半空间占用
    # size 默认为None，表示是否修改图片的分辨率，如果修改分辨率, eg. size=(1920, 1080)
    if size is not None:
        img = cv2.resize(img, size)
    # 设置压缩参数（比特率）
    compression_params = [cv2.IMWRITE_JPEG_QUALITY, bit_rate]
    # 保存压缩后的图像
    cv2.imwrite(save_path, img, compression_params)


def download_image_file(image_path, file_image_url, image_id):
    if not os.path.exists(image_path):
        logger.info(f"Create image file path {image_path}")
        os.makedirs(image_path)

    for url in file_image_url:
        response = requests.get(url)
        file_name = url.split("/")[-1]
        file_path = os.path.join(image_path, image_id + "_" + file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Download image file {file_path} success.")

    logger.info("Download image file success.")
