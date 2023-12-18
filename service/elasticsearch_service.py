from typing import List, Any

import numpy as np
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

from entity.milvus_entity import FaceKeyFrameEmbedding
from model.model_onnx import Face_Onnx
from utils import log_util

logger = log_util.get_logger(__name__)

elasticsearch_host = '10.10.38.72'
elasticsearch_port = 9200
es_client = Elasticsearch(hosts=[{'host': elasticsearch_host, 'port': elasticsearch_port, 'scheme': "http"}])
logger.info(f"Elasticsearch client created. host: {elasticsearch_host}, port: {elasticsearch_port}")

main_avatar_v1_index = 'main_avatars_v1'

image_faces_v1_index = 'image_faces_v1'

content_faces_v1_index = 'content_faces_v1'


def insert_face_embedding(file_data: Any, face_frame_embedding_list: List[FaceKeyFrameEmbedding]):
    actions = []
    index_name = None
    if file_data.tag == 'video':
        index_name = image_faces_v1_index
    elif file_data.tag == 'content':
        index_name = content_faces_v1_index

    if index_name is None:
        raise ValueError(f"Index is None, Invalid tag: {file_data.tag}")

    for face_frame_embedding in face_frame_embedding_list:

        # if file_data.tag == 'video':
        #     # 主头像选举
        if float(face_frame_embedding.quantity_score) > 40:
            action = {
                "_index": image_faces_v1_index,
                "_id": face_frame_embedding.key_id,
                "_source": {
                    'object_id': face_frame_embedding.object_id,
                    'embedding': face_frame_embedding.embedding,
                    'hdfs_path': face_frame_embedding.hdfs_path,
                    'quantity_score': face_frame_embedding.quantity_score,
                    'video_id_arr': face_frame_embedding.video_id_arr,
                    'earliest_video_id': face_frame_embedding.earliest_video_id,
                    'file_name': face_frame_embedding.file_name
                }
            }
            actions.append(action)
    res = bulk(es_client, actions)
    logger.info(f"FaceKeyFrameEmbedding bulk insert result is {res}")
    return res


# def main_face_election(face_frame_embedding_info):
#     logger.info(f"Main face election for {face_frame_embedding_info.key_id}")
#     main_face_info = search_main_face(face_frame_embedding_info)
#     # 无法找到主人像的信息，并且质量得分大约0.4分才可选举为主人像
#     if main_face_info is None and float(face_frame_embedding_info.quantity_score) > 40:
#         main_face_info = MainFaceKeyFrameEmbedding(key_id=face_frame_embedding_info.key_id,
#                                                    object_id=face_frame_embedding_info.key_id,
#                                                    quantity_score=face_frame_embedding_info.quantity_score,
#                                                    embedding=face_frame_embedding_info.embedding,
#                                                    hdfs_path=face_frame_embedding_info.hdfs_path,
#                                                    recognition_state="unidentification")
#         insert_result = insert_main_face(main_face_info)
#
#         face_frame_embedding_info.object_id = main_face_info.key_id
#         logger.info(f"Insert main face {main_face_info.key_id} to Milvus. {insert_result}")
#     elif main_face_info is not None and float(face_frame_embedding_info.quantity_score) > 40:
#         logger.info(f"Main face {main_face_info.key_id} found in Milvus. {main_face_info}")
#         if main_face_info.quantity_score < face_frame_embedding_info.quantity_score:
#             logger.info(f"Main face {main_face_info.key_id} quality score is lower than "
#                         f"{face_frame_embedding_info.key_id}")
#             main_face_info.quantity_score = face_frame_embedding_info.quantity_score
#             main_face_info.embedding = face_frame_embedding_info.embedding
#             main_face_info.hdfs_path = face_frame_embedding_info.hdfs_path
#             main_face_info.object_id = face_frame_embedding_info.key_id
#
#             face_frame_embedding_info.object_id = main_face_info.key_id
#             update_result = update_main_face(main_face_info)
#             logger.info(f"Update main face {main_face_info.key_id} to Milvus. {update_result}")
#     else:
#         logger.info(f"Main face {face_frame_embedding_info.key_id} quality score is lower than 40")
#         face_frame_embedding_info.object_id = "notgoodface"
#
#
# def search_main_face(face_frame_embedding: FaceKeyFrameEmbedding) -> MainFaceKeyFrameEmbedding:
#     search_params = {
#         "metric_type": "IP",
#         "ignore_growing": False,
#         "params": {"nprobe": 50}
#     }
#     limit = 1
#     limit = 16383 if limit > 16383 else limit
#     search_res = main_avatar_v1_index.search([face_frame_embedding.embedding], 'embedding', search_params,
#                                              limit=limit, output_fields=['object_id', 'hdfs_path', 'quality_score',
#                                                                          'recognition_state'], round_decimal=4)
#     if len(search_res) == 0:
#         return None
#     main_face_list = []
#     for one in search_res:
#         for single in one:
#             if single.score < 0.5:
#                 continue
#             logger.info(f"Search single result: {single} and score is {single.score}")
#             main_face_info = MainFaceKeyFrameEmbedding(key_id=single.entity.id, object_id=single.entity.object_id,
#                                                        quantity_score=single.entity.quality_score,
#                                                        hdfs_path=single.entity.hdfs_path,
#                                                        recognition_state=single.entity.recognition_state)
#             main_face_list.append(main_face_info)
#     return main_face_list[0] if len(main_face_list) > 0 else None


def search_face_image(model: Face_Onnx, index_name: str, image, enhance=False, score=0.5, start=0, size=10):
    embedding = model.turn2embeddings(image, enhance=enhance)

    body = {
        "from": start,
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1000",
                    "params": {
                        "query_vector": embedding[0]
                    }
                }
            }
        }
    }
    result = []
    search_res = es_client.search(index=index_name, body=body)
    search_res = search_res['hits']['hits']
    for one in search_res:
        if one['_score'] < score:
            continue
        result.append(one['_source'])

    return result


def check_index_exists(index_name):
    """Check if an index exists."""
    return es_client.indices.exists(index=index_name)


def create_index(index_name):
    """Create an index."""
    return es_client.indices.create(index=index_name, ignore=400)


def delete_index(index_name):
    """Delete an index."""
    return es_client.indices.delete(index=index_name, ignore=[400, 404])


def index_document(index_name, doc_type, doc_id, body):
    """Index a document."""
    return es_client.index(index=index_name, doc_type=doc_type, id=doc_id, body=body)


def get_document(index_name, doc_type, doc_id):
    """Get a document."""
    try:
        return es_client.get(index=index_name, doc_type=doc_type, id=doc_id)
    except NotFoundError:
        return None


def delete_document(index_name, doc_type, doc_id):
    """Delete a document."""
    return es_client.delete(index=index_name, doc_type=doc_type, id=doc_id, ignore=[400, 404])


def search(index_name, body):
    """Search documents."""
    return es_client.search(index=index_name, body=body)
