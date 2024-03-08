# Create a logger
from utils import log_util

logger = log_util.get_logger(__name__)


class MainFaceRequestEntity:
    def __init__(self, request):
        self.score = request.form.get("score", 0.6)
        self.page_num = request.form.get("page_num", 1)
        self.page_size = request.form.get("page_size", 10)
        self.recognition_state = request.form.get('recognitionState', 'unidentification')
        self.object_id = request.form.get('objectId')
        self.saas_flag = request.form.get('office_code')
        self.offset = (int(self.page_num) - 1) * int(self.page_size)

    def validate(self):
        logger.info(f"validate request {self.to_dict()}")
        if self.recognition_state not in ['unidentification', 'identification']:
            raise ValueError(
                f"recognitionState {self.recognition_state} is not valid， valid value is ['unidentification', 'identification']")

    def to_dict(self):
        return {
            "score": self.score,
            "page_num": self.page_num,
            "page_size": self.page_size,
            "recognition_state": self.recognition_state,
            "object_id": self.object_id,
            "saas_flag": self.saas_flag,
            "offset": self.offset
        }

    def to_esl_query(self):
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "recognition_state.keyword": self.recognition_state
                            }
                        }
                    ]
                }
            },
            "from": self.offset,
            "size": self.page_size
        }
        if self.object_id:
            query["query"]["bool"]["must"].append({
                "term": {
                    "object_id.keyword": self.object_id
                }
            })
        return query


class MainFaceInsertEntity:
    def __init__(self, request):
        self.score = request.form.get("score", 0.6)
        self.object_id = request.form.get('objectId')
        self.hdfs_path = request.form.get('hdfsPath')
        self.file = request.files.get('file')
        self.saas_flag = request.form.get('office_code')

    def validate(self):
        logger.info(f"validate request {self.to_dict()}")
        if not self.object_id:
            raise ValueError("objectId is required")
        if not self.hdfs_path:
            raise ValueError("hdfsPath is required")
        if not self.file:
            raise ValueError("file is required")

    def to_dict(self):
        return {
            "score": self.score,
            "object_id": self.object_id,
            "hdfs_path": self.hdfs_path,
            "file": self.file,
            "saas_flag": self.saas_flag
        }

    def determine_face_exist_query(self, embedding):
        similarity_search = "cosineSimilarity(params.query_vector, 'embedding') + 1000"
        query = {
            "min_score": 1000 + float(self.score),
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "params": {
                            "query_vector": embedding
                        },
                        "source": similarity_search
                    }
                }
            }
        }
        return query

    def insert_query(self, embedding):
        query = {
            "object_id": self.object_id,
            "hdfs_path": self.hdfs_path,
            "embedding": embedding,
            "recognition_state": "identification",
        }
        return query
