from typing import List, Any

from config.config import get_config
from entity.milvus_entity import MainFaceKeyFrameEmbedding, FaceKeyFrameEmbedding, KeyFrameEmbedding
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
content_faces_v1_name = face_app_conf["content_face_collection"]

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
content_faces_v1 = Collection(content_faces_v1_name, image_faces_schema)
logger.info(f"Collection {content_faces_v1} created successfully")

# 内容人像库索引


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
content_faces_v1.create_index("embedding", index)

logger.info(f"Index {index} created successfully")

image_faces_v1.load()
main_avatar_v1.load()
content_faces_v1.load()

logger.info(f"Collection {image_faces_v1_name} loaded successfully")
logger.info(f"Collection {main_avatar_v1} loaded successfully")
logger.info(f"Collection {content_faces_v1} loaded successfully")

# 视频关键帧向量集合
video_frame_v1_name = face_app_conf["video_frame_collection"]
content_frame_v1_name = face_app_conf["content_frame_collection"]

has = utility.has_collection(video_frame_v1_name)
logger.info(f"Does collection {video_frame_v1_name} exist in Milvus: {has}")

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),
    FieldSchema(name="hdfs_path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="earliest_video_id", dtype=DataType.VARCHAR, max_length=256),
]
schema = CollectionSchema(fields, "video_frame_v1_name is the simplest demo to introduce the APIs")
video_frame_v1 = Collection(video_frame_v1_name, schema)
logger.info(f"Collection {video_frame_v1_name} created successfully")

index = {
    "index_type": 'IVF_FLAT',
    "metric_type": "L2",
    "params": {"nlist": 128},
}

video_frame_v1.create_index("embedding", index)
logger.info(f"Index {index} created successfully")

video_frame_v1.load()

logger.info(f"Collection {video_frame_v1_name} loaded successfully")

has = utility.has_collection(content_frame_v1_name)
logger.info(f"Does collection {content_frame_v1_name} exist in Milvus: {has}")
content_frame_v1 = Collection(content_frame_v1_name, schema)
logger.info(f"Collection {content_frame_v1_name} created successfully")

index = {
    "index_type": 'IVF_FLAT',
    "metric_type": "L2",
    "params": {"nlist": 128},
}

content_frame_v1.create_index("embedding", index)
logger.info(f"Index {index} created successfully")

content_frame_v1.load()


def search_main_face(face_frame_embedding: FaceKeyFrameEmbedding) -> MainFaceKeyFrameEmbedding:
    search_params = {
        "metric_type": "IP",
        "ignore_growing": False,
        "params": {"nprobe": 50}
    }
    limit = 1
    limit = 16383 if limit > 16383 else limit
    search_res = main_avatar_v1.search([face_frame_embedding.embedding], 'embedding', search_params,
                                       limit=limit, output_fields=['object_id', 'hdfs_path', 'quality_score',
                                                                   'recognition_state'], round_decimal=4)
    if len(search_res) == 0:
        return None
    main_face_list = []
    for one in search_res:
        for single in one:
            if single.score < 0.5:
                continue
            logger.info(f"Search single result: {single} and score is {single.score}")
            main_face_info = MainFaceKeyFrameEmbedding(key_id=single.entity.id, object_id=single.entity.object_id,
                                                       quantity_score=single.entity.quality_score,
                                                       hdfs_path=single.entity.hdfs_path,
                                                       recognition_state=single.entity.recognition_state)
            main_face_list.append(main_face_info)
    return main_face_list[0] if len(main_face_list) > 0 else None


def insert_main_face(main_face_info: MainFaceKeyFrameEmbedding):
    entities = [[], [], [], [], [], []]
    entities[0].append(main_face_info.key_id)
    entities[1].append(str(main_face_info.object_id))
    entities[2].append(main_face_info.embedding)
    entities[3].append(str(main_face_info.hdfs_path))
    entities[4].append(main_face_info.quantity_score)
    entities[5].append(main_face_info.recognition_state)
    res = main_avatar_v1.insert(entities)
    logger.info(f"Insert main face {main_face_info.key_id} to Milvus. {res}")

    return res


def update_main_face(main_face_info: MainFaceKeyFrameEmbedding):
    # 2.2.9 Collection暂无更新语句， 使用insert_main_face替代
    logger.info(f"update_main_face main face {main_face_info.key_id} to Milvus. ")
    return insert_main_face(main_face_info)


def main_face_election(face_frame_embedding_info):
    logger.info(f"Main face election for {face_frame_embedding_info.key_id}")
    main_face_info = search_main_face(face_frame_embedding_info)
    # 无法找到主人像的信息，并且质量得分大约0.4分才可选举为主人像
    if main_face_info is None and float(face_frame_embedding_info.quantity_score) > 40:
        main_face_info = MainFaceKeyFrameEmbedding(key_id=face_frame_embedding_info.key_id,
                                                   object_id=face_frame_embedding_info.key_id,
                                                   quantity_score=face_frame_embedding_info.quantity_score,
                                                   embedding=face_frame_embedding_info.embedding,
                                                   hdfs_path=face_frame_embedding_info.hdfs_path,
                                                   recognition_state="unidentification")
        insert_result = insert_main_face(main_face_info)

        face_frame_embedding_info.object_id = main_face_info.key_id
        logger.info(f"Insert main face {main_face_info.key_id} to Milvus. {insert_result}")
    elif main_face_info is not None and float(face_frame_embedding_info.quantity_score) > 40:
        logger.info(f"Main face {main_face_info.key_id} found in Milvus. {main_face_info}")
        if main_face_info.quantity_score < face_frame_embedding_info.quantity_score:
            logger.info(f"Main face {main_face_info.key_id} quality score is lower than "
                        f"{face_frame_embedding_info.key_id}")
            main_face_info.quantity_score = face_frame_embedding_info.quantity_score
            main_face_info.embedding = face_frame_embedding_info.embedding
            main_face_info.hdfs_path = face_frame_embedding_info.hdfs_path
            main_face_info.object_id = face_frame_embedding_info.key_id

            face_frame_embedding_info.object_id = main_face_info.key_id
            update_result = update_main_face(main_face_info)
            logger.info(f"Update main face {main_face_info.key_id} to Milvus. {update_result}")
    else:
        logger.info(f"Main face {face_frame_embedding_info.key_id} quality score is lower than 40")
        face_frame_embedding_info.object_id = "notgoodface"


def insert_face_embedding(file_data: Any, face_frame_embedding_list: List[FaceKeyFrameEmbedding]):
    entities = [[], [], [], [], [], [], [], []]
    for face_frame_embedding_info in face_frame_embedding_list:

        # 来自视频库才进行， 主人像选举
        if file_data.tag == 'video':
            main_face_election(face_frame_embedding_info)
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
    res = None
    if file_data.tag == 'video':
        if len(entities[0]) > 0:
            res = image_faces_v1.insert(entities)
            logger.info(f"Insert face frame embedding to Milvus. {res}")
    elif file_data.tag == 'content':
        if len(entities[0]) > 0:
            res = content_faces_v1.insert(entities)
            logger.info(f"Insert content face frame embedding to Milvus. {res}")

    # 执行插入向量
    return res


def insert_frame_embedding(file_data, key_frame_embedding_list: List[KeyFrameEmbedding]):
    entities = [[], [], [], []]
    for key_frame_embedding_info in key_frame_embedding_list:
        # 来自视频库才进行， 主人像选举
        # 插入到人脸向量库里
        # 0 id, 1 embedding, 2 hdfs_path 3 earliest_video_id
        entities[0].append(key_frame_embedding_info.key_id)
        entities[1].append(key_frame_embedding_info.embedding)
        entities[2].append(str(key_frame_embedding_info.hdfs_path))
        entities[3].append(key_frame_embedding_info.earliest_video_id)
    res = None
    if file_data.tag == 'video':
        if len(entities[0]) > 0:
            res = video_frame_v1.insert(entities)
            logger.info(f"Insert face frame embedding to Milvus. {res}")
    elif file_data.tag == 'content':
        if len(entities[0]) > 0:
            res = content_frame_v1.insert(entities)
            logger.info(f"Insert content face frame embedding to Milvus. {res}")

    # 执行插入向量
    return res
