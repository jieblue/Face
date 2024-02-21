import math
import time
from typing import List

import numpy as np
import torch

from config.config import get_config
from entity.frame_entity import KeyFrame, FaceKeyFrame
from entity.milvus_entity import FaceKeyFrameEmbedding, KeyFrameEmbedding
from model.model_onnx import Face_Onnx
from model.model_video import VideoModel
from utils import log_util
from utils.img_util import cv_imread

logger = log_util.get_logger(__name__)

logger.info("Loading config")
# 获取config信息
conf = get_config()
# 加载人脸模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)
logger.info("Face model loaded successfully")

video_model = VideoModel('./config/weights/ResNet2048_v224.onnx', gpu_id=0)
logger.info("Video model loaded successfully")

embedding_dim = conf["face_app"]["embedding_dim"]
logger.info(f"embedding_dim is {embedding_dim}")


def grouping_key_frame(key_frame_list: List[KeyFrame]):
    for key_frame in key_frame_list:
        frame_embedding = video_model.get_frame_embedding(key_frame.frame_stream)
        logger.info(f"Extracted frame num {key_frame.frame_num} frames ")
        key_frame.embedding = frame_embedding

    logger.info(f"Extracted {len(key_frame_list)} frames ")
    start_pos = 0
    end_pos = 0
    length = len(key_frame_list)
    result = []
    while end_pos < length:
        if start_pos == end_pos:
            end_pos = end_pos + 1
        else:
            start_frame = key_frame_list[start_pos]
            end_frame = key_frame_list[end_pos]

            rate = keyframe_similarity(start_frame.embedding, end_frame.embedding)
            logger.info(f"key frame {start_frame.key_id} and {end_frame.key_id} similarity is {rate}")
            if rate > 0.9:
                end_pos = end_pos + 1
            else:
                result.append(start_frame)
                start_pos = end_pos
                end_pos = end_pos + 1
    logger.info(f"Extracted {len(result)} key frames ")
    return result


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
            logger.info(f"Extracted face num {face_key_frame.face_num} in {key_frame.frame_num} frames ")
            face_key_frame_list.append(face_key_frame)

    return face_key_frame_list


def translate_frame_embedding(key_frame_list: List[KeyFrame]) -> List[KeyFrameEmbedding]:
    frame_embedding_list = []
    for frame_info in key_frame_list:
        frame_embedding = frame_info.embedding
        if frame_embedding is None:
            frame_embedding = video_model.get_frame_embedding(frame_info.frame_stream)
            logger.info(f"Extracted {frame_info.frame_num} frames ")
        else:
            logger.info(f"Extracted {frame_info.frame_num} frames from cache")
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


def translate_face_embedding_256(face_frame_list: List[FaceKeyFrame]) -> List[FaceKeyFrameEmbedding]:
    face_frame_embedding_list = []
    for face_frame_info in face_frame_list:
        # 人脸质量得分
        score = face_model.tface.forward(face_frame_info.face_frame)
        # TODO 人脸质量得分低于多少分的可以启动是否增强人脸
        # 转换向量
        original_embedding = face_model.turn2embeddings_256(face_frame_info.face_frame,
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
    result_list = [max(single, key=lambda face_embedding: face_embedding.quantity_score) for single in res]
    for i, face_embedding_info in enumerate(result_list):
        logger.info(f"Grouping face {i} max score is {face_embedding_info.quantity_score}, face_embedding_info "
                    f"is {face_embedding_info.key_id}")
    return result_list


def cul_similarity(face_x, face_y):
    # list 2 numpy
    np_fx = np.array(face_x)
    np_fy = np.array(face_y)
    return np.dot(np_fx, np_fy)


def keyframe_similarity(frame_x, frame_y):
    distance = np.linalg.norm(np.array(frame_x) - np.array(frame_y))
    return normalized_euclidean_distance(distance)


# 人脸图片转为特征向量
def turn_to_face_embedding(self, img, enhance=False, aligned=False,
                           confidence=0.99):
    if embedding_dim == 256:
        logger.info("Turn to 256 embeddings")
        return face_model.turn2embeddings_256(img, enhance, aligned, confidence)
    else:
        logger.info("Turn to 512 embeddings")
        return face_model.turn2embeddings(img, enhance, aligned, confidence)


def get_frame_embedding(self, frame_image):
    if embedding_dim == 256:
        logger.info("Get 256 frame embedding")
        return video_model.get_frame_embedding_256(frame_image)
    else:
        logger.info("Get 512 frame embedding")
        return video_model.get_frame_embedding(frame_image)


def normalized_euclidean_distance(L2, dim=512):
    if embedding_dim == 256:
        dim = 256
    dim_sqrt = math.sqrt(dim)
    return 1 / (1 + L2 / dim_sqrt)
