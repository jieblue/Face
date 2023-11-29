from typing import List

from entity.milvus_entity import MainFaceKeyFrameEmbedding, FaceKeyFrameEmbedding
# from face_app import image_faces_v1
from utils import log_util

logger = log_util.get_logger(__name__)


def search_main_face(face_frame_embedding) -> MainFaceKeyFrameEmbedding:
    # TODO search main face
    return MainFaceKeyFrameEmbedding(None, None, None, None, None, None)


def insert_main_face(main_face_info: MainFaceKeyFrameEmbedding):
    return None


def update_main_face(main_face_info: MainFaceKeyFrameEmbedding):
    return None


def main_face_election(face_frame_embedding_info):
    # TODO 查找这次人员是否存在主头像， 不存在直接插入， 存在进行得分比较进行更新
    # main_face_info = search_main_face(face_frame_embedding_info.embedding)
    # if main_face_info is None:
    #     main_face_info = MainFaceKeyFrameEmbedding(face_frame_embedding_info.key_id,
    #                                                face_frame_embedding_info.key_id,
    #                                                face_frame_embedding_info.video_id,
    #                                                face_frame_embedding_info.frame_num,
    #                                                face_frame_embedding_info.timestamp,
    #                                                face_frame_embedding_info.face_num,
    #                                                face_frame_embedding_info.embedding)
    #     insert_result = insert_main_face(main_face_info)
    #     logger.info(f"Insert main face {main_face_info.key_id} to Milvus. {insert_result}")
    # else:
    #     if main_face_info.quantity_score < face_frame_embedding_info.quantity_score:
    #         main_face_info.quantity_score = face_frame_embedding_info.quantity_score
    #         main_face_info.embedding = face_frame_embedding_info.embedding
    #         main_face_info.hdfs_path = face_frame_embedding_info.hdfs_path
    #         main_face_info.object_id = face_frame_embedding_info.key_id
    #         update_result = update_main_face(main_face_info)
    #         logger.info(f"Update main face {main_face_info.key_id} to Milvus. {update_result}")
    pass


def insert_face_embedding(face_frame_embedding_list: List[FaceKeyFrameEmbedding]):
    entities = [[], [], [], [], [], [], [], []]
    for face_frame_embedding_info in face_frame_embedding_list:

        # 插入到人脸向量库里
        # 0 id, 1 object_id, 2 embedding, 3 hdfs_path 4 quantity_score, 5 video_id_arr, 6 earliest_video_id, 7 file_name
        entities[0].append(face_frame_embedding_info.key_id)
        entities[1].append(face_frame_embedding_info.object_id)
        entities[2].append(face_frame_embedding_info.embedding)
        entities[3].append(face_frame_embedding_info.hdfs_path)
        entities[4].append(face_frame_embedding_info.quantity_score)
        entities[5].append(face_frame_embedding_info.video_id_arr)
        entities[6].append(face_frame_embedding_info.earliest_video_id)
        entities[7].append(face_frame_embedding_info.file_name)

        # 主人像选举
        # main_face_election(face_frame_embedding_info)

    # res = image_faces_v1.insert(entities)
    # if len(entities[0]) > 0:
    #     res = image_faces_v1.insert(entities)
    #     logger.info(f"Insert face frame embedding to Milvus. {res}")

    # 执行插入向量
    return None
