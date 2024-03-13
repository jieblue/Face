from typing import List, Any

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

from config.config import get_config
from entity.milvus_entity import FaceKeyFrameEmbedding, MainFaceKeyFrameEmbedding, KeyFrameEmbedding
from model.model_onnx import Face_Onnx
from service import visual_algorithm_service
from utils import log_util

logger = log_util.get_logger(__name__)
conf = get_config()

es_config = conf['elasticsearch']
elasticsearch_host = es_config['host']
elasticsearch_port = es_config['port']
elasticsearch_user = es_config['username']
elasticsearch_password = es_config['password']
es_client = Elasticsearch(
    hosts=[{'host': elasticsearch_host, 'port': elasticsearch_port, 'scheme': "http"}],
    http_auth=(elasticsearch_user, elasticsearch_password),
    timeout=300,
)
logger.info(f"Elasticsearch client host: {elasticsearch_host}, port: {elasticsearch_port}")
logger.info(f"Elasticsearch client user: {elasticsearch_user}, password: {elasticsearch_password}")
face_app_conf = conf['face_app']

main_avatar_v1_index = face_app_conf['main_avatar_collection']
logger.info(f"Face app main_avatar_collection: {main_avatar_v1_index}")

image_faces_v1_index = face_app_conf['image_face_collection']
logger.info(f"Face app image_face_collection: {image_faces_v1_index}")

content_faces_v1_index = face_app_conf['content_face_collection']
logger.info(f"Face app content_face_collection: {content_faces_v1_index}")

video_frames_v1_index = face_app_conf['video_frame_collection']
logger.info(f"Face app video_frame_collection: {video_frames_v1_index}")

content_frames_v1_index = face_app_conf['content_frame_collection']
logger.info(f"Face app content_frame_collection: {content_frames_v1_index}")

embedding_dim = face_app_conf['embedding_dim']
logger.info(f"Face app embedding_dim: {embedding_dim}")

if embedding_dim == 256:
    main_avatar_v1_index = main_avatar_v1_index + "_" + str(embedding_dim)
    image_faces_v1_index = image_faces_v1_index + "_" + str(embedding_dim)
    content_faces_v1_index = content_faces_v1_index + "_" + str(embedding_dim)
    video_frames_v1_index = video_frames_v1_index + "_" + str(embedding_dim)
    content_frames_v1_index = content_frames_v1_index + "_" + str(embedding_dim)
    logger.info(f"Face app dim_embedding is 256, so index name is {main_avatar_v1_index}, {image_faces_v1_index}")

similarity_search = "cosineSimilarity(params.query_vector, 'embedding') + 1000"


def insert_face_embedding(file_data: Any, face_frame_embedding_list: List[FaceKeyFrameEmbedding]):
    actions = []
    index_name = get_image_face_index(file_data.saas_flag)

    for face_frame_embedding in face_frame_embedding_list:

        if file_data.tag == 'video' or file_data.tag == 'content':
            main_face_election(face_frame_embedding, file_data.saas_flag)
        if float(face_frame_embedding.quantity_score) > 40:
            earliest_video_id = face_frame_embedding.earliest_video_id
            if file_data.tag == 'content' and earliest_video_id is not None:
                earliest_video_id = str(earliest_video_id).split("_")[0]
            if file_data.library_type is not None:
                face_frame_embedding.key_id = f"{face_frame_embedding.key_id}_{file_data.library_type}"
            action = {
                "_index": index_name,
                "_id": face_frame_embedding.key_id,
                "_source": {
                    'key_id': face_frame_embedding.key_id,
                    'object_id': face_frame_embedding.object_id,
                    'embedding': face_frame_embedding.embedding,
                    'hdfs_path': face_frame_embedding.hdfs_path,
                    'quality_score': face_frame_embedding.quantity_score,
                    'video_id_arr': face_frame_embedding.video_id_arr,
                    'earliest_video_id': earliest_video_id,
                    'file_name': face_frame_embedding.file_name,
                    'tag': file_data.tag
                }
            }
            if file_data.library_type is not None:
                action['_source']['from_source'] = file_data.library_type
            actions.append(action)
    res = bulk(es_client, actions)
    return res


def main_face_election(face_frame_embedding_info, saas_flag=None):
    logger.info(f"Main face election for {face_frame_embedding_info.key_id}")
    main_face_info = search_main_face(face_frame_embedding_info)

    # 无法找到主人像的信息，并且质量得分大约0.4分才可选举为主人像
    if main_face_info is None and float(face_frame_embedding_info.quantity_score) > 40:
        if face_frame_embedding_info.tag == 'video':
            main_face_info = MainFaceKeyFrameEmbedding(key_id=face_frame_embedding_info.key_id,
                                                       object_id=face_frame_embedding_info.key_id,
                                                       quantity_score=face_frame_embedding_info.quantity_score,
                                                       embedding=face_frame_embedding_info.embedding,
                                                       hdfs_path=face_frame_embedding_info.hdfs_path,
                                                       recognition_state="unidentification")
            insert_result = insert_main_face(main_face_info, saas_flag)
            logger.info(f"Insert main face {main_face_info.key_id} to Milvus. {insert_result}")

        face_frame_embedding_info.object_id = face_frame_embedding_info.key_id
    elif main_face_info is not None and float(face_frame_embedding_info.quantity_score) > 40:
        logger.info(f"Main face {main_face_info.key_id} found in Milvus. {main_face_info}")
        face_frame_embedding_info.object_id = main_face_info.key_id
        if main_face_info.recognition_state == "identification":
            face_frame_embedding_info.recognition_state = "identification"
            logger.info(f"Human Face face {face_frame_embedding_info.key_id} is identification")
        if main_face_info.quantity_score < face_frame_embedding_info.quantity_score:
            logger.info(f"Main face {main_face_info.key_id} quality score is lower than "
                        f"{face_frame_embedding_info.key_id}")
            main_face_info.quantity_score = face_frame_embedding_info.quantity_score
            main_face_info.embedding = face_frame_embedding_info.embedding
            main_face_info.hdfs_path = face_frame_embedding_info.hdfs_path

            if face_frame_embedding_info.tag == 'video':
                update_result = update_main_face(main_face_info, saas_flag)
                logger.info(f"Update main face {main_face_info.key_id} to Milvus. {update_result}")
    else:
        logger.info(f"Main face {face_frame_embedding_info.key_id} quality score is lower than 40")
        face_frame_embedding_info.object_id = "notgoodface"
        logger.info(f"Main face {face_frame_embedding_info.key_id} is not good face")


def insert_main_face(main_face_info: MainFaceKeyFrameEmbedding, saas_flag=None):
    body = {
        'object_id': main_face_info.object_id,
        'embedding': main_face_info.embedding,
        'hdfs_path': main_face_info.hdfs_path,
        'quality_score': main_face_info.quantity_score,
        'recognition_state': main_face_info.recognition_state
    }
    index_name = get_main_avatar_index(saas_flag)
    res = es_client.index(index=index_name, id=main_face_info.key_id, body=body)
    logger.info(f"Insert main face {main_face_info.key_id} to Elasticsearch. {res}")
    return res


def update_main_face(main_face_info: MainFaceKeyFrameEmbedding, saas_flag=None):
    body = {
        'doc': {
            'object_id': main_face_info.object_id,
            'embedding': main_face_info.embedding,
            'hdfs_path': main_face_info.hdfs_path,
            'quality_score': main_face_info.quantity_score,
            'recognition_state': main_face_info.recognition_state
        }
    }
    index_name = get_main_avatar_index(saas_flag)

    res = es_client.update(index=index_name, id=main_face_info.key_id, body=body)
    logger.info(f"Update main face {main_face_info.key_id} in Elasticsearch. {res}")
    return res


def search_main_face(face_frame_embedding: FaceKeyFrameEmbedding) -> MainFaceKeyFrameEmbedding:
    body = {
        'min_score': 1000.6,
        "size": 1,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": similarity_search,
                    "params": {
                        "query_vector": face_frame_embedding.embedding
                    }
                }
            }
        },
        "_source": ["object_id", "hdfs_path", "quality_score", "recognition_state"]
    }
    search_res = es_client.search(index=main_avatar_v1_index, body=body)

    search_res = search_res['hits']['hits']
    if len(search_res) == 0:
        return None

    main_face_list = []
    for one in search_res:
        current_score = one['_score'] - 1000
        logger.info(f"Search main avatar result: {one} and score is {current_score}")
        main_face_info = MainFaceKeyFrameEmbedding(key_id=one['_id'], object_id=one['_source']['object_id'],
                                                   quantity_score=one['_source']['quality_score'],
                                                   hdfs_path=one['_source']['hdfs_path'],
                                                   recognition_state=one['_source']['recognition_state'])

        main_face_list.append(main_face_info)

    return main_face_list[0] if len(main_face_list) > 0 else None


def search_main_face_image(model: Face_Onnx, index_name: str, image, enhance=False, score=0.5, start=0, size=10):
    embedding = visual_algorithm_service.turn_to_face_embedding(image, enhance=enhance)
    min_score = score + 1000
    body = {
        "min_score": min_score,
        "from": start,
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": similarity_search,
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
        current_score = one['_score'] - 1000
        logger.info(f"Search single result: {one} and score is {current_score}")
        tmp = {
            'id': one['_id'],
            'object_id': one['_source']['object_id'],
            'hdfs_path': one['_source']['hdfs_path'],
            'score': current_score,
            'quality_score': one['_source']['quality_score'],
            'recognition_state': one['_source']['recognition_state']
        }

        result.append(tmp)

    return result


def search_content_face_image(model: Face_Onnx, index_name: str, image, enhance=False, score=0.5, start=0, size=10):
    embedding = visual_algorithm_service.turn_to_face_embedding(image, enhance=enhance)

    min_score = score + 1000
    body = {
        "min_score": min_score,
        "from": start,
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match_all": {}
                            }

                        ],
                        "must_not": [
                            {
                                "match_phrase": {
                                    "tag": "content"
                                }
                            }
                        ]
                    }
                },
                "script": {
                    "source": similarity_search,
                    "params": {
                        "query_vector": embedding[0]
                    }
                }
            }
        },
        "collapse": {
            "field": "earliest_video_id.raw"
        }

    }
    return search_face_similarity(index_name, body)


def search_face_image(model: Face_Onnx, index_name: str, image, enhance=False, score=0.5, start=0, size=10,
                      embedding_arr=[]):
    embedding = visual_algorithm_service.turn_to_face_embedding(image, enhance=enhance)[0] if len(
        embedding_arr) == 0 else embedding_arr
    body = {
        "min_score": score + 1000,
        "from": start,
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [{"match_all": {}}],
                        "must_not": [{"match_phrase": {"tag": "content"}}]
                    }
                },
                "script": {
                    "source": similarity_search,
                    "params": {"query_vector": embedding}
                }
            }
        },
        "collapse": {"field": "earliest_video_id.raw"}
    }
    return search_face_similarity(index_name, body)


def search_face_image_with_date(index_name: str, image, enhance=False, score=0.5, start=0, size=10,
                                embedding_arr=[],
                                begin_time="2023-11-01 00:00:00", end_time="2024-11-01 00:00:00"):
    embedding = visual_algorithm_service.turn_to_face_embedding(image, enhance=enhance)[0] if len(
        embedding_arr) == 0 else embedding_arr
    must_condition = []
    must_condition.append({
        "range": {
            "created_at": {
                "gte": begin_time,
                "lte": end_time
            }
        }
    })
    body = {
        "min_score": score + 1000,
        "from": start,
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": must_condition,
                        "must_not": [{"match_phrase": {"tag": "content"}}]
                    }
                },
                "script": {
                    "source": similarity_search,
                    "params": {"query_vector": embedding}
                }
            }
        },
        "collapse": {"field": "earliest_video_id.raw"}
    }

    logger.info(f"Search face image with date query: {body}")

    return search_face_similarity(index_name, body)


def search_face_similarity(index_name: str, body):
    result = []
    search_res = es_client.search(index=index_name, body=body)
    total = search_res['hits']['total']['value']
    search_res = search_res['hits']['hits']
    for one in search_res:
        current_score = one['_score'] - 1000
        logger.info(f"Search single result: {one} and score is {current_score}")
        tmp = {
            'id': one['_id'],
            'object_id': one['_source']['object_id'],
            'hdfs_path': one['_source']['hdfs_path'],
            'score': current_score,
            'quality_score': one['_source']['quality_score'],
            'video_id_arr': one['_source']['video_id_arr'],
            'earliest_video_id': one['_source']['earliest_video_id'],
            'file_name': one['_source']['file_name']
        }

        result.append(tmp)
    return [result], total


def search_main_face_image(model: Face_Onnx, index_name: str, image, enhance=False, score=0.5, start=0, size=10,
                           embedding_arr=[]):
    if len(embedding_arr) <= 0:
        embedding = visual_algorithm_service.turn_to_face_embedding(image, enhance=enhance)[0]
    else:
        embedding = embedding_arr

    min_score = score + 1000

    body = {
        "min_score": min_score,
        "from": start,
        "size": size,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": similarity_search,
                    "params": {
                        "query_vector": embedding
                    }
                }
            }
        }
    }
    result = []
    search_res = es_client.search(index=index_name, body=body)
    search_res = search_res['hits']['hits']
    for one in search_res:
        current_score = one['_score'] - 1000

        logger.info(f"Search single result: {one} and score is {current_score}")
        tmp = {
            'id': one['_id'],
            'object_id': one['_source']['object_id'],
            'hdfs_path': one['_source']['hdfs_path'],
            'score': current_score,
            'quality_score': one['_source']['quality_score'],
            'recognition_state': one['_source']['recognition_state'],
            "embedding": one['_source']['embedding']
        }

        result.append(tmp)

    return result


def insert_frame_embedding(file_data: Any, key_frame_embedding_list: List[KeyFrameEmbedding]):
    actions = []

    index_name = None
    if file_data.tag == 'video':
        logger.info(f"Insert frame embedding for video {file_data.video_id}")
        index_name = get_video_frame_index(file_data.saas_flag)
    elif file_data.tag == 'content' or file_data.tag == 'video-index':
        logger.info(f"Insert frame embedding for content {file_data.video_id}")
        index_name = get_video_frame_index(file_data.saas_flag)

    if index_name is None:
        raise ValueError(f"Index is None, Invalid tag: {file_data.tag}")

    for key_frame_embedding in key_frame_embedding_list:
        earliest_video_id = key_frame_embedding.earliest_video_id
        if file_data.tag == 'content' and earliest_video_id is not None:
            earliest_video_id = str(earliest_video_id).split("_")[0]

        if file_data.library_type is not None:
            key_frame_embedding.key_id = f"{key_frame_embedding.key_id}_{file_data.library_type}"
            logger.info(f"Key frame embedding key id is {key_frame_embedding.key_id}")
        action = {
            "_index": index_name,
            "_id": key_frame_embedding.key_id,
            "_source": {
                'key_id': key_frame_embedding.key_id,
                'embedding': key_frame_embedding.embedding,
                'hdfs_path': key_frame_embedding.hdfs_path,
                'earliest_video_id': earliest_video_id,
                'tag': file_data.tag
            }
        }

        if file_data.library_type is not None:
            action['_source']['from_source'] = file_data.library_type

        actions.append(action)
    res = bulk(es_client, actions)

    logger.info(f"KeyFrameEmbedding bulk insert result is {res}")
    return res


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


def get_main_avatar_index(saas_flag):
    if saas_flag is not None and saas_flag != "":
        logger.info(f"Get main_avatar_v1_index: {saas_flag + main_avatar_v1_index}")
        return saas_flag + "_" + main_avatar_v1_index
    else:
        logger.info(f"No saas, Get main_avatar_v1_index: {main_avatar_v1_index}")
        return main_avatar_v1_index


def get_video_frame_index(saas_flag):
    if saas_flag is not None and saas_flag != "":
        logger.info(f"Get video_frames_v1_index: {saas_flag + video_frames_v1_index}")
        return saas_flag + "_" + video_frames_v1_index
    else:
        logger.info(f"No saas, Get video_frames_v1_index: {video_frames_v1_index}")
        return video_frames_v1_index


def get_image_face_index(saas_flag):
    if saas_flag is not None and saas_flag != "":
        logger.info(f"Get image_faces_v1_index: {saas_flag + image_faces_v1_index}")
        return saas_flag + "_" + image_faces_v1_index
    else:
        logger.info(f"No saas, Get image_faces_v1_index: {image_faces_v1_index}")
        return image_faces_v1_index


def main_avatar_search(saas_flag, query):
    """
    This function is used to search for the main avatar in the Elasticsearch index.

    Parameters:
    saas_flag (str): A flag indicating the Software as a Service (SaaS) context. This is used to determine the index name.
    query (dict): The Elasticsearch query to be executed.

    Returns:
    dict: The search results from Elasticsearch.

    Raises:
    ValueError: If the Elasticsearch index does not exist.
    """
    index_name = get_main_avatar_index(saas_flag)
    logger.info(f"Main avatar search query: {query}")
    if not es_client.indices.exists(index=index_name):
        raise ValueError(f"Index {index_name} does not exist")
    return es_client.search(index=index_name, body=query)


def main_avatar_insert(saas_flag, id, body):
    """
    This function is used to insert a main avatar into the Elasticsearch index.

    Parameters:
    saas_flag (str): A flag indicating the Software as a Service (SaaS) context. This is used to determine the index name.
    body (dict): The document to be inserted into the Elasticsearch index.

    Returns:
    dict: The response from Elasticsearch.

    Raises:
    ValueError: If the Elasticsearch index does not exist.
    """
    index_name = get_main_avatar_index(saas_flag)
    if not es_client.indices.exists(index=index_name):
        raise ValueError(f"Index {index_name} does not exist")
    return es_client.index(index=index_name, id=id, body=body)


def image_faces_search(saas_flag, query):
    index_name = get_image_face_index(saas_flag)
    if not es_client.indices.exists(index=index_name):
        raise ValueError(f"Index {index_name} does not exist")
    return es_client.search(index=index_name, body=query)


def video_frame_search(saas_flag, query):
    index_name = get_video_frame_index(saas_flag)
    if not es_client.indices.exists(index=index_name):
        raise ValueError(f"Index {index_name} does not exist")
    return es_client.search(index=index_name, body=query)
