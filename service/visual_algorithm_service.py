from typing import List

import numpy as np
import torch

from config.config import get_config
from entity.frame_entity import KeyFrame, FaceKeyFrame
from entity.milvus_entity import FaceKeyFrameEmbedding, MainFaceKeyFrameEmbedding
from model.model_onnx import Face_Onnx
from service import milvus_service
from utils import log_util
from utils.img_util import cv_imread

logger = log_util.get_logger(__name__)

# 获取config信息
conf = get_config()
# 加载人脸模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)


def extract_face_list(key_frame_list: List[KeyFrame]) -> List[FaceKeyFrame]:
    face_key_frame_list = []
    for key_frame in key_frame_list:
        face_list = face_model.extract_face(key_frame.frame, enhance=False,
                                            confidence=0.99)
        face_num = 1
        for face_frame in face_list:
            face_key_frame = FaceKeyFrame(key_frame.file_name, key_frame.video_id,
                                          key_frame.frame_num,
                                          key_frame.timestamp, face_num, face_frame)
            face_num = face_num + 1
            face_key_frame_list.append(face_key_frame)

    return face_key_frame_list


def translate_frame_embedding(key_frame_list: List[KeyFrame]):
    raise NotImplementedError("translate_frame_embedding not implemented")


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

        # 人员ID， HDFS path， 关联的视频组， 在向量插入的时候进行关联
        face_key_frame_embedding = FaceKeyFrameEmbedding(face_frame_info.key_id,
                                                         None,
                                                         score,
                                                         face_frame_embedding,
                                                         None,
                                                         None,
                                                         face_frame_info.video_id,
                                                         file_name=face_frame_info.file_name)

        face_frame_embedding_list.append(face_key_frame_embedding)
    return face_frame_embedding_list


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
