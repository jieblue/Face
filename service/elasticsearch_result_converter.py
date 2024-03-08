from utils import log_util

logger = log_util.get_logger(__name__)


def main_avatar_result_converter(original_result):
    logger.info(f"main_avatar_result_converter search spend time {original_result['took']} ms")
    total = original_result['hits']['total']['value']
    logger.info(f"main_avatar_result_converter search total {total}")
    search_result = []
    for hit in original_result['hits']['hits']:
        current_score = hit['_score'] - 1000
        tmp = {
            'id': hit['_id'],
            'object_id': hit['_source']['object_id'],
            'hdfs_path': hit['_source']['hdfs_path'],
            'score': current_score,
            'quality_score': str(hit['_source']['quality_score']),
            'recognition_state': hit['_source']['recognition_state'],
            "embedding": hit['_source']['embedding']
        }
        search_result.append(tmp)

    return total, search_result


def face_predict_result_converter(original_result):
    logger.info(f"face_predict_result_converter search spend time {original_result['took']} ms")
    total = original_result['hits']['total']['value']
    logger.info(f"face_predict_result_converter search total {total}")
    search_res = original_result['hits']['hits']
    result = []
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
    return total, [result]
