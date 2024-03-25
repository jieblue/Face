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
            'recognition_state': hit['_source']['recognition_state'],
            "embedding": hit['_source']['embedding']
        }

        if "quality_score" in hit['_source']:
            tmp['quality_score'] = hit['_source']['quality_score']
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

        if "from_source" in one['_source']:
            tmp['from_source'] = one['_source']['from_source']

        result.append(tmp)
    return total, [result]


def total_result_converter(original_result):
    logger.info(f"total_result_converter search spend time {original_result['took']} ms")
    return original_result['aggregations']['total_num']['value']


def content_video_predict_result_converter(original_result):
    logger.info(f"content_video_predict_result_converter search spend time {original_result['took']} ms")
    total = original_result['hits']['total']['value']
    logger.info(f"content_video_predict_result_converter search total {total}")
    search_res = original_result['hits']['hits']
    result = []
    for one in search_res:
        earliest_video_id = ""
        if one['_source']['earliest_video_id'] is not None:
            earliest_video_id = str(one['_source']['earliest_video_id']).split("_")[0]
        current_score = one['_score'] - 1000
        tmp = {
            'id': one['_id'],
            'hdfs_path': one['_source']['hdfs_path'],
            'score': current_score,
            'earliest_video_id': earliest_video_id,
            'tag': one['_source']['tag'],
        }

        if "from_source" in one['_source']:
            tmp['from_source'] = one['_source']['from_source']

        if "public_topic_arr" in one['_source']:
            tmp['public_topic_arr'] = one['_source']['public_topic_arr']

        result.append(tmp)
    return total, [result]


def video_predict_result_converter(original_result):
    logger.info(f"content_video_predict_result_converter search spend time {original_result['took']} ms")
    total = original_result['hits']['total']['value']
    logger.info(f"content_video_predict_result_converter search total {total}")
    search_res = original_result['hits']['hits']
    result = []
    for one in search_res:
        earliest_video_id = ""
        if one['_source']['earliest_video_id'] is not None:
            earliest_video_id = str(one['_source']['earliest_video_id']).split("_")[0]
        current_score = one['_score'] - 1000
        if float(current_score) < 0.9:
            logger.info(f"{one['_id']} document current_score: {current_score} < score: {0.9}")
            continue
        tmp = {
            'id': one['_id'],
            'hdfs_path': one['_source']['hdfs_path'],
            'score': current_score,
            'earliest_video_id': earliest_video_id,
            'tag': one['_source']['tag'],
        }

        if "from_source" in one['_source']:
            tmp['from_source'] = one['_source']['from_source']

        if "public_topic_arr" in one['_source']:
            tmp['public_topic_arr'] = one['_source']['public_topic_arr']

        result.append(tmp)
    return total, [result]
