from utils import log_util

logger = log_util.get_logger(__name__)


def main_avatar_result_converter(original_result):
    logger.info(f"main_avatar_result_converter search spend time {original_result['took']} ms")
    total = original_result['hits']['total']['value']
    logger.info(f"main_avatar_result_converter search total {total}")
    search_result = []
    for hit in original_result['hits']['hits']:
        tmp = {
            'id': hit['_id'],
            'object_id': hit['_source']['object_id'],
            'hdfs_path': hit['_source']['hdfs_path'],
            'quality_score': str(hit['_source']['quality_score']),
            'recognition_state': hit['_source']['recognition_state'],
        }
        search_result.append(tmp)

    return total, search_result
