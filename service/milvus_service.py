from typing import List

from config.config import get_config
from entity.milvus_entity import MainFaceKeyFrameEmbedding, FaceKeyFrameEmbedding
# from face_app import image_faces_v1
from utils import log_util
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,

)
# 获取config信息
conf = get_config()
logger = log_util.get_logger(__name__)

# 获取Milvus的配置
milvus_conf = conf['milvus']
logger.info("Milvus 配置信息： " + str(conf['milvus']))

connections.connect("default", host=milvus_conf["host"], port=milvus_conf["port"], user=milvus_conf["user"],
                    password=milvus_conf["password"])
# 获取face_app的配置
face_app_conf = conf['face_app']
image_faces_v1_name = face_app_conf["image_face_collection"]

has = utility.has_collection(image_faces_v1_name)
logger.info(f"Milvus collection {image_faces_v1_name} exist in Milvus: {has}")

# 人像库索引
image_faces_fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
    FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="hdfs_path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="quality_score", dtype=DataType.FLOAT, max_length=256),
    FieldSchema(name="video_id_arr", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="earliest_video_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=128),
]
image_faces_schema = CollectionSchema(image_faces_fields, "image_faces_v1 is the simplest demo to introduce the APIs")
image_faces_v1 = Collection(image_faces_v1_name, image_faces_schema)

# 主头像二级人像库
# 人像库索引
main_avatar_fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
    FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="hdfs_path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="quality_score", dtype=DataType.FLOAT, max_length=256),
    FieldSchema(name="recognition_state", dtype=DataType.VARCHAR, max_length=64),

]
main_avatar_schema = CollectionSchema(main_avatar_fields, "image_faces_v1 is the simplest demo to introduce the APIs")
main_avatar_v1 = Collection(face_app_conf["main_avatar_collection"], main_avatar_schema)

index = {
    "index_type": 'IVF_SQ8',
    "metric_type": "IP",
    "params": {"nlist": 100},
}

image_faces_v1.create_index("embedding", index)
main_avatar_v1.create_index("embedding", index)
logger.info(f"Index {index} created successfully")

image_faces_v1.load()
main_avatar_v1.load()

logger.info(f"Collection {image_faces_v1_name} loaded successfully")

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
        entities[1].append(str(face_frame_embedding_info.object_id))
        entities[2].append(face_frame_embedding_info.embedding)
        entities[3].append(str(face_frame_embedding_info.hdfs_path))
        entities[4].append(face_frame_embedding_info.quantity_score)
        entities[5].append(face_frame_embedding_info.video_id_arr)
        entities[6].append(face_frame_embedding_info.earliest_video_id)
        entities[7].append(face_frame_embedding_info.file_name)

        # TODO 主人像选举
        # main_face_election(face_frame_embedding_info)

    res = image_faces_v1.insert(entities)
    if len(entities[0]) > 0:
        res = image_faces_v1.insert(entities)
        logger.info(f"Insert face frame embedding to Milvus. {res}")

    # 执行插入向量
    return res
