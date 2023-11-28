from entity.file_entity import VideoFile
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
    logger.info(f"Key frame list extracted. {len(key_frame_list)}")

    # 提出视频关键帧中的人脸关键帧集合
    face_frame_list = visual_algorithm_service.extract_face_list(key_frame_list)
    logger.info(f"Face frame list extracted. {len(face_frame_list)}")

    # TODO 将视频关键帧进行转换， 提取视频关键帧特征集合
    # key_frame_embedding_list = visual_algorithm_service.translate_frame_embedding(key_frame_list)
    # 将视频关键帧特征集合存储到Milvus中
    # frame_insert_result = milvus_service.insert_frame_embedding(key_frame_embedding_list)

    # 将人脸关键帧进行转换， 提取人脸关键帧特征集合
    face_frame_embedding_list = visual_algorithm_service.translate_face_embedding(face_frame_list)
    logger.info(f"Face frame embedding list extracted. {len(face_frame_embedding_list)}")

    # 将人脸关键帧特征集合存储到Milvus中
    face_insert_result = milvus_service.insert_face_embedding(face_frame_embedding_list)
    logger.info(f"Face frame embedding list inserted. {len(face_insert_result)}")

    # 将视频关键帧特征集合存储到本地路径
    # frame_save_result = file_service.save_frame_to_disk(key_frame_list)

    # 将人脸关键帧特征集合存储到本地路径
    face_save_result = file_service.save_face_to_disk(face_frame_list)
    logger.info(f"Face frame list saved. {len(face_save_result)}")

    return key_frame_list, face_frame_list
