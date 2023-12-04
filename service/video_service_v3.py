from typing import List, Any

from entity.file_entity import VideoFile, ImageFile
from entity.frame_entity import KeyFrame, FaceKeyFrame
from service import video_validator, basic_algorithm_service, visual_algorithm_service, milvus_service, file_service
from utils import log_util

# Create a logger
logger = log_util.get_logger(__name__)


def process_video_file(video_file: VideoFile):
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
    # key_frame_embedding_list = visual_algorithm_service.translate_frame_embedding(key_frame_list)
    # 将视频关键帧特征集合存储到Milvus中
    # frame_insert_result = milvus_service.insert_frame_embedding(key_frame_embedding_list)

    # 将人脸关键帧进行转换， 提取人脸关键帧特征集合
    face_frame_embedding_list = visual_algorithm_service.translate_face_embedding(face_frame_list)
    logger.info(f"Face frame embedding list extracted. {len(face_frame_embedding_list)}")

    # 将人脸关键帧特征集合存储到Milvus中
    face_insert_result = milvus_service.insert_face_embedding(file_data, face_frame_embedding_list)
    logger.info(f"Face frame embedding list inserted. {face_insert_result}")

    # 将人脸关键帧特征集合存储到本地路径
    file_service.save_face_to_disk(face_frame_list)
    logger.info(f"Face frame list saved to disk.")

    key_frame_embedding_list_result = []
    # 删除多余的字段
    for key_frame in key_frame_list:
        key_frame_embedding_list_result.append(key_frame.to_dict())

    face_embedding_list_result = []
    for face_embedding in face_frame_embedding_list:
        face_embedding_list_result.append(face_embedding.to_dict())

    return key_frame_embedding_list_result, face_embedding_list_result

