import os
import time
from typing import List, Any

import requests

import config.config
from entity.file_entity import VideoFile, ImageFile
from entity.frame_entity import KeyFrame, FaceKeyFrame
from service import video_validator, basic_algorithm_service, visual_algorithm_service, milvus_service, file_service, \
    elasticsearch_service
from utils import log_util

# Create a logger
logger = log_util.get_logger(__name__)

url_prefix = 'http://vod-face.fjtv.net/'

video_root_path = config.config.get_config()['video_root_path']


def download_video(video_file: VideoFile):
    file_path = video_file.file_path
    url = video_file.file_path.replace(video_root_path, url_prefix)

    logger.info(f"Download video from {url} to {video_file.file_name}")

    file_path_parent = os.path.dirname(file_path)

    # 创建目录
    os.makedirs(str(file_path_parent), exist_ok=True)

    if os.path.exists(file_path):
        logger.info(f"{file_path} exists in local disk")
    else:
        logger.info(f"{file_path} does not exist in HDFS, url: {url}, filename: {file_path}")
        # Download video file
        start_time = time.time()

        response = requests.get(url)
        # Save the file to HDFS
        # Write file to HDFS
        with open(file_path, "wb") as f:
            f.write(response.content)
        end_time = time.time()

        file_target_size = os.path.getsize(file_path)
        logger.info(f"file target size {file_target_size} ")
        if file_target_size > 0:
            logger.info(
                f"{file_path} and file target size {int(file_target_size / 1024 / 1024)} MB Time taken to write file to local: {end_time - start_time} seconds")
        else:
            print(f"{file_path} Time taken to write file to local: {end_time - start_time} seconds")


def delete_video_file(video_file):
    file_path = video_file.file_path
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Video file {file_path} deleted.")
    else:
        logger.info(f"Video file {file_path} does not exist.")


def process_video_file(video_file: VideoFile):
    if not video_file.local_disk:
        download_video(video_file)

    # 验证数据格式
    video_validator.video_validate(video_file)
    logger.info(f"Video id {video_file.video_id} and video path {video_file.file_path} file validated.")

    # 提取视频关键帧对象集合
    key_frame_list = basic_algorithm_service.extract_key_frame_list(video_file)
    logger.info(f"Key frame list extracted. number: {len(key_frame_list)}")
    # 转换成ImageFile对象集合

    # 提出视频关键帧中的人脸关键帧集合
    face_frame_list = visual_algorithm_service.extract_face_list(key_frame_list)
    logger.info(f"Face frame list extracted. number: {len(face_frame_list)}")
    if not video_file.local_disk:
        delete_video_file(video_file)

    return process_key_frame(video_file, key_frame_list, face_frame_list)


def process_image_file(image_file: ImageFile):
    # 验证数据格式
    video_validator.image_validate(image_file)
    logger.info(f"Image id {image_file.image_id} and image path {image_file.file_path} file validated.")

    # 提取图片关键帧对象集合
    key_frame_list = basic_algorithm_service.extract_image_key_frame_list(image_file)
    logger.info(f"Image file Key frame list extracted. number: {len(key_frame_list)}")

    # 提出图片关键帧中的人脸关键帧集合
    face_frame_list = visual_algorithm_service.extract_face_list(key_frame_list)
    logger.info(f"Image file Face frame list extracted. number: {len(face_frame_list)}")

    return process_key_frame(image_file, key_frame_list, face_frame_list)


def process_key_frame(file_data: Any, key_frame_list: List[KeyFrame], face_frame_list: List[FaceKeyFrame]):
    # TODO 将视频关键帧进行转换， 提取视频关键帧特征集合
    key_frame_embedding_list = visual_algorithm_service.translate_frame_embedding(key_frame_list)
    logger.info(f"Key frame embedding list extracted. {len(key_frame_embedding_list)}")
    # 将视频关键帧特征集合存储到Milvus中
    frame_insert_result = elasticsearch_service.insert_frame_embedding(file_data, key_frame_embedding_list)
    logger.info(f"Key frame embedding list inserted elasticsearch. {frame_insert_result}")
    # 将人脸关键帧进行转换， 提取人脸关键帧特征集合
    face_frame_embedding_list = visual_algorithm_service.translate_face_embedding(face_frame_list)
    logger.info(f"Face frame embedding list extracted. {len(face_frame_embedding_list)}")

    # 将人脸关键帧特征集合存储到Milvus中
    face_insert_result = elasticsearch_service.insert_face_embedding(file_data, face_frame_embedding_list)
    logger.info(f"Face frame embedding list inserted elasticsearch. {face_insert_result}")

    # 将人脸关键帧特征集合存储到本地路径
    file_service.save_face_to_disk(face_frame_list)
    logger.info(f"Face frame list saved to disk.")

    # 将视频关键帧特征集合存储到本地路径
    file_service.save_frame_to_disk(key_frame_list)

    key_frame_embedding_list_result = []
    # 删除多余的字段
    for key_frame in key_frame_list:
        key_frame_embedding_list_result.append(key_frame.to_dict())

    face_embedding_list_result = []
    for face_embedding in face_frame_embedding_list:
        if face_embedding.object_id != "notgoodface":
            face_embedding_list_result.append(face_embedding.to_dict())

    return key_frame_embedding_list_result, face_embedding_list_result
