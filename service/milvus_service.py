from typing import List

from entity.milvus_entity import MainFaceKeyFrameEmbedding, FaceKeyFrameEmbedding
from utils import log_util

logger = log_util.get_logger(__name__)
def search_main_face(face_frame_embedding) -> MainFaceKeyFrameEmbedding:
    # TODO search main face
    return MainFaceKeyFrameEmbedding(None, None, None, None, None, None)


def insert_main_face(main_face_info: MainFaceKeyFrameEmbedding):
    return None


def update_main_face(main_face_info: MainFaceKeyFrameEmbedding):
    return None


def insert_face_embedding(face_frame_embedding_list: List[FaceKeyFrameEmbedding]):

    for face_frame_embedding_info in face_frame_embedding_list:
        # TODO 查找这次人员是否存在主头像， 不存在直接插入， 存在进行得分比较进行更新
        main_face_info = search_main_face(face_frame_embedding_info.embedding)
        if main_face_info is None:
            main_face_info = MainFaceKeyFrameEmbedding(face_frame_embedding_info.key_id,
                                                       face_frame_embedding_info.key_id,
                                                       face_frame_embedding_info.video_id,
                                                       face_frame_embedding_info.frame_num,
                                                       face_frame_embedding_info.timestamp,
                                                       face_frame_embedding_info.face_num,
                                                       face_frame_embedding_info.embedding)
            insert_result = insert_main_face(main_face_info)
            logger.info(f"Insert main face {main_face_info.key_id} to Milvus. {insert_result}")
        else:
            if main_face_info.quantity_score < face_frame_embedding_info.quantity_score:
                main_face_info.quantity_score = face_frame_embedding_info.quantity_score
                main_face_info.embedding = face_frame_embedding_info.embedding
                main_face_info.hdfs_path = face_frame_embedding_info.hdfs_path
                main_face_info.object_id = face_frame_embedding_info.key_id
                update_result = update_main_face(main_face_info)
                logger.info(f"Update main face {main_face_info.key_id} to Milvus. {update_result}")
    return None