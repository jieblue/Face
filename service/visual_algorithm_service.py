import time
from typing import List

import numpy as np
import torch

from config.config import get_config
from entity.frame_entity import KeyFrame, FaceKeyFrame
from entity.milvus_entity import FaceKeyFrameEmbedding, MainFaceKeyFrameEmbedding, KeyFrameEmbedding
from model.model_onnx import Face_Onnx
from model.model_video import VideoModel
from service import milvus_service
from utils import log_util
from utils.img_util import cv_imread

logger = log_util.get_logger(__name__)

# 获取config信息
conf = get_config()
# 加载人脸模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)
logger.info("Face model loaded successfully")

video_model = VideoModel('./config/weights/Pvt.onnx', gpu_id=0)
logger.info("Video model loaded successfully")


def extract_face_list(key_frame_list: List[KeyFrame]) -> List[FaceKeyFrame]:
    face_key_frame_list = []
    for key_frame in key_frame_list:
        face_list = face_model.extract_face(key_frame.frame, enhance=False,
                                            confidence=0.99)
        face_num = 1
        for face_frame in face_list:
            face_key_frame = FaceKeyFrame(file_name=key_frame.file_name, video_id=key_frame.video_id,
                                          frame_num=key_frame.frame_num,
                                          timestamp=key_frame.timestamp, face_num=face_num, face_frame=face_frame,
                                          tag=key_frame.tag)
            face_num = face_num + 1
            face_key_frame_list.append(face_key_frame)

    return face_key_frame_list


def translate_frame_embedding(key_frame_list: List[KeyFrame]) -> List[KeyFrameEmbedding]:
    frame_embedding_list = []
    for frame_info in key_frame_list:
        # 转换向量
        frame_embedding = video_model.get_frame_embedding(frame_info.frame_stream)

        # 人员ID在主人像选举的时候进行添加， HDFS path， 关联的视频组， 在向量插入的时候进行关联
        key_frame_embedding = KeyFrameEmbedding(key_id=frame_info.key_id,
                                                embedding=frame_embedding,
                                                hdfs_path=frame_info.hdfs_path,
                                                earliest_video_id=frame_info.video_id,
                                                file_name=frame_info.file_name,
                                                frame_num=frame_info.frame_num,
                                                timestamp=frame_info.timestamp)

        frame_embedding_list.append(key_frame_embedding)

    return frame_embedding_list


def translate_face_embedding(face_frame_list: List[FaceKeyFrame]) -> List[FaceKeyFrameEmbedding]:
    face_frame_embedding_list = []
    for face_frame_info in face_frame_list:
        # 人脸质量得分
        score = face_model.tface.forward(face_frame_info.face_frame)
        # TODO 人脸质量得分低于多少分的可以启动是否增强人脸
        # 转换向量
        original_embedding = face_model.turn2embeddings(face_frame_info.face_frame,
                                                        enhance=False,
                                                        aligned=True,
                                                        confidence=0.99)
        # 压缩向量
        face_frame_embedding = squeeze_faces(original_embedding)[0]

        # 人员ID在主人像选举的时候进行添加， HDFS path， 关联的视频组， 在向量插入的时候进行关联
        face_key_frame_embedding = FaceKeyFrameEmbedding(tag=face_frame_info.tag,
                                                         key_id=face_frame_info.key_id,
                                                         object_id=face_frame_info.key_id,
                                                         quantity_score=score,
                                                         embedding=face_frame_embedding,
                                                         hdfs_path=face_frame_info.hdfs_path,
                                                         video_id_arr=face_frame_info.video_id,
                                                         earliest_video_id=face_frame_info.video_id,
                                                         file_name=face_frame_info.file_name,
                                                         frame_num=face_frame_info.frame_num,
                                                         timestamp=face_frame_info.timestamp,
                                                         face_num=face_frame_info.face_num)
        logger.info(f"FaceKeyFrameEmbedding is {face_key_frame_embedding.key_id} is complete")
        face_frame_embedding_list.append(face_key_frame_embedding)

    return grouping_face(face_frame_embedding_list)


def squeeze_faces(faces_list, threshold=0.48):
    """
    Squeezes the faces in the given list based on the given threshold.

    Args:
        faces_list (list): A list of faces to be squeezed.
        threshold (float): The threshold value for squeezing the faces. Default is 0.48.

    Returns:
        list: A list of squeezed faces.
    """
    faces = np.array(faces_list)
    _len = len(faces_list)

    # numpy to tensor
    faces_tensor = torch.from_numpy(faces).float()
    unique_vectors = []
    # ids = []
    for i, vector in enumerate(faces_tensor):

        # 检查是否与之前的向量重复
        is_duplicate = False
        vector_tensor = vector.unsqueeze(0)
        # print(vector_tensor.size())
        for x in unique_vectors:
            x_tensor = x.unsqueeze(0)
            # print(x_tensor.size())
            # 计算余弦相似度
            if torch.nn.functional.cosine_similarity(vector_tensor, x_tensor) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_vectors.append(vector)
            # ids.append(i)

    numpy_list = [t.unsqueeze(0).numpy().tolist()[0] for t in unique_vectors]
    # 从有范数的向量列表中提取没有范数的向量列表.astype(np.float32)
    return numpy_list


# 批获取人脸图片的质量分数
# 返回result 和 err, err记录读取出错的图片路径
# 批获取人脸图片的质量分数
# imgs为list，list中的每个img是经过人脸检测得到的人脸图片
# 返回result 和 err, err记录错误信息，暂时为空list
def get_face_quality_single_img(model: Face_Onnx, image_path):
    img = cv_imread(image_path)
    score = model.tface.forward(img)
    return score


def grouping_face(face_embedding_list: List[FaceKeyFrameEmbedding], threshold=0.6):
    start_time = time.time()
    res = []
    start_point = 0
    last_point = 0
    group_human = []
    list_len = len(face_embedding_list)
    # 第一重聚类
    while last_point < list_len:
        first_face = face_embedding_list[start_point]
        if start_point == last_point:
            group_human.append(first_face)
            last_point += 1
        else:
            second_face = face_embedding_list[last_point]
            if cul_similarity(first_face.embedding, second_face.embedding) > threshold:
                group_human.append(second_face)
                last_point += 1
            else:
                res.append(group_human)
                group_human = []
                start_point = last_point
    if len(group_human) != 0:
        res.append(group_human)
    logger.info(f"Grouping face res len {len(res)} time taken: {time.time() - start_time} seconds")

    # # 第二重聚类
    # twice_start_point = 0
    # twice_last_point = 0
    # twice_group_human = []
    # twice_list_len = len(res)
    # twice_res = []
    # while twice_last_point < twice_list_len:
    #     first_face = res[twice_start_point]
    #     if twice_start_point == twice_last_point:
    #         twice_group_human.extend(first_face)
    #         twice_last_point += 1
    #     else:
    #         second_face = res[twice_last_point]
    #         if cul_similarity(first_face[0].embedding, second_face[0].embedding) > threshold:
    #             twice_group_human.extend(second_face)
    #             twice_last_point += 1
    #         else:
    #             twice_res.append(twice_group_human)
    #             twice_group_human = []
    #             twice_start_point = twice_last_point
    #
    # if len(twice_group_human) != 0:
    #     twice_res.append(twice_group_human)
    # logger.info(f"Grouping twice face res len {len(twice_res)} time taken: {time.time() - start_time} seconds")
    result_list = [max(single, key=lambda face_embedding: face_embedding.quantity_score) for single in res]
    for i, face_embedding_info in enumerate(result_list):
        logger.info(f"Grouping face {i} max score is {face_embedding_info.quantity_score}, face_embedding_info "
                    f"is {face_embedding_info.key_id}")
    return result_list


# def grouping_face(face_embedding_list: List[FaceKeyFrameEmbedding], threshold=0.6):
#     start_time = time.time()
#     res = []
#     for face_embedding_info in face_embedding_list:
#         max_score, need_insert_index = -1, -1
#         for i, single in enumerate(res):
#             for current_face in single:
#                 similarity_score = cul_similarity(current_face.embedding, face_embedding_info.embedding)
#                 if (current_face.key_id != face_embedding_info.key_id and similarity_score > threshold
#                         and similarity_score > max_score):
#                     need_insert_index, max_score = i, similarity_score
#                     logger.info(f"Grouping face {current_face.key_id} and "
#                                 f"face {face_embedding_info.key_id} similarity is {similarity_score}")
#         res[need_insert_index].append(
#             face_embedding_info) if need_insert_index != -1 and max_score > threshold else res.append(
#             [face_embedding_info])
#
#     logger.info(f"Grouping face time taken: {time.time() - start_time} seconds")
#
#     result_list = [max(single, key=lambda face_embedding_info: face_embedding_info.quantity_score) for single in res]
#     # for i, face_embedding_info in enumerate(result_list):
#     #     logger.info(f"Grouping face {i} max score is {face_embedding_info.quantity_score}, face_embedding_info "
#     #                 f"is {face_embedding_info.to_dict()}")
#     return result_list


def cul_similarity(face_x, face_y):
    # list 2 numpy
    np_fx = np.array(face_x)
    np_fy = np.array(face_y)
    return np.dot(np_fx, np_fy)
