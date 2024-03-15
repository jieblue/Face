# Create a logger
import json
from datetime import datetime

from dateutil.relativedelta import relativedelta

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
                    "object_id": self.object_id
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


class FacePredictEntity:

    def __init__(self, request):
        self.file = request.files.get('file')
        self.score = request.form.get("score", 0.6)
        self.page_num = request.form.get("pageNum", 1)
        self.page_size = request.form.get("pageSize", 10)
        self.begin_time = request.form.get("beginTime")
        self.end_time = request.form.get("endTime")
        self.saas_flag = request.form.get('office_code')
        self.embedding_arr = json.loads(request.form.get('embedding', '[]'))
        self.offset = (int(self.page_num) - 1) * int(self.page_size)
        self.total_query = ""

    def to_dict(self):
        return {
            "file": self.file,
            "score": self.score,
            "page_num": self.page_num,
            "page_size": self.page_size,
            "begin_time": self.begin_time,
            "end_time": self.end_time,
            "saas_flag": self.saas_flag,
            "embedding_arr": self.embedding_arr
        }

    def validate(self):
        logger.info(f"validate request {self.to_dict()}")
        if not self.file and len(self.embedding_arr) == 0:
            raise ValueError("file is required or embedding is required")

        if self.begin_time is None or self.begin_time == "":
            logger.info("beginTime is None, start use default value, past three months ago")
            now = datetime.now()
            # Get the time three months ago
            three_months_ago = now - relativedelta(months=3)
            # Format the time strings
            self.end_time = now.strftime('%Y-%m-%d %H:%M:%S')
            self.begin_time = three_months_ago.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Current time: {self.begin_time}")
            logger.info(f"Three months ago: {self.end_time}")

        if self.saas_flag is None:
            logger.warn("office_code is None, saas is not started")

    def to_esl_query(self):
        similarity_search = "cosineSimilarity(params.query_vector, 'embedding') + 1000"
        must_condition = [{
            "range": {
                "created_at": {
                    "gte": self.begin_time,
                    "lte": self.end_time
                }
            }
        }]
        query = {
            "min_score": float(self.score) + 1000,
            "size": 0,
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
                        "params": {"query_vector": self.embedding_arr}
                    }
                }
            }
        }

        self.total_query = query.copy()
        query["size"] = self.page_size
        query["from"] = self.offset
        query["collapse"] = {"field": "earliest_video_id.raw"}

        logger.info(f"Search face image with date query: {query}")
        return query

    def to_total_query(self):
        self.total_query["aggs"] = {
            "total_num": {
                "cardinality": {
                    "field": "earliest_video_id.raw"}

            }
        }
        return self.total_query


class MainFacePredictEntity:

    def __init__(self, request):
        self.file = request.files.get('file')
        self.score = request.form.get("score", 0.6)
        self.page_num = request.form.get("pageNum", 1)
        self.page_size = request.form.get("pageSize", 10)
        self.saas_flag = request.form.get('office_code')
        self.offset = (int(self.page_num) - 1) * int(self.page_size)

    def to_dict(self):
        return {
            "file": self.file,
            "score": self.score,
            "page_num": self.page_num,
            "page_size": self.page_size,
            "saas_flag": self.saas_flag
        }

    def validate(self):
        logger.info(f"validate request {self.to_dict()}")
        if not self.file:
            raise ValueError("file is required")

        if self.saas_flag is None:
            logger.warn("office_code is None, saas is not started")

    def to_esl_query(self, embedding):
        similarity_search = "cosineSimilarity(params.query_vector, 'embedding') + 1000"
        query = {
            "min_score": float(self.score) + 1000,
            "from": self.offset,
            "size": self.page_size,
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

        query["collapse"] = {"field": "object_id.raw"}
        return query


class ContentFacePredictEntity:
    def __init__(self, request):
        self.file = request.files.get('file')
        self.score = request.form.get("score", 0.6)
        self.page_num = request.form.get("pageNum", 1)
        self.page_size = request.form.get("pageSize", 10)
        self.saas_flag = request.form.get('office_code')
        self.public_type = request.form.get('public_type')
        self.offset = (int(self.page_num) - 1) * int(self.page_size)
        self.begin_time = request.form.get("beginTime")
        self.end_time = request.form.get("endTime")
        self.library_type = request.form.get("libraryType")
        self.category_id = request.form.get("category_id")
        self.column_id = request.form.get('column_id')
        self.create_user_id = request.form.get('create_user_id')
        self.site_id = request.form.get('site_id')
        self.total_query = ""

    def to_dict(self):
        return {
            "file": self.file,
            "score": self.score,
            "page_num": self.page_num,
            "page_size": self.page_size,
            "saas_flag": self.saas_flag,
            "public_type": self.public_type,
            "begin_time": self.begin_time,
            "end_time": self.end_time,
            "library_type": self.library_type,
            "category_id": self.category_id,
            "column_id": self.column_id,
            "create_user_id": self.create_user_id,
            "site_id": self.site_id
        }

    def validate(self):
        logger.info(f"validate request {self.to_dict()}")
        if not self.file:
            raise ValueError("file is required")

        if self.saas_flag is None:
            logger.warn("office_code is None, saas is not started")

    def to_esl_query(self, embedding):
        query = {
            "bool": {
                "must": [
                    {
                        "match_all": {}
                    }
                ],
                "must_not": [
                    {
                        "match_phrase": {
                            "del_flag": "1"
                        }
                    }
                ],
            }
        }

        must_condition_list = []
        # TODO beginTime and endTime 等待对接上
        # begin_time = request.form.get('beginTime')
        # end_time = request.form.get('endTime')
        # if begin_time is None or begin_time == "":
        #     now = datetime.now()
        #     # Get the time three months ago
        #     three_months_ago = now - relativedelta(months=3)
        #     # Format the time strings
        #     end_time = now.strftime('%Y-%m-%d %H:%M:%S')
        #     begin_time = three_months_ago.strftime('%Y-%m-%d %H:%M:%S')
        #     logger.info(f"Current time: {begin_time}")
        #     logger.info(f"Three months ago: {end_time}")
        #
        # must_condition_list.append({
        #     "range": {
        #         "created_at": {
        #             "gte": begin_time,
        #             "lte": end_time
        #         }
        #     }
        # })

        if self.library_type is not None and self.library_type != "":

            if self.library_type == "video":
                must_condition_list.append({
                    "match_phrase": {
                        "tag": "video"
                    }
                })

            if self.library_type == "public":
                must_condition_list.append({
                    "match_phrase": {
                        "from_source": "public"
                    }
                })

                if self.public_type is not None and self.public_type != "":
                    must_condition_list.append({
                        "match_phrase": {
                            "public_type": self.public_type
                        }
                    })
            elif self.library_type == "article":
                must_condition_list.append({
                    "match_phrase": {
                        "from_source": "article"
                    }
                })

            if self.category_id is not None and self.category_id != "":
                category_id_arr = self.category_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_category_id": category_id_arr
                    }
                })

            if self.column_id is not None and self.column_id != "":
                column_id_arr = self.column_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_column_id": column_id_arr
                    }
                })

            if self.create_user_id is not None and self.create_user_id != "":
                create_user_id_arr = self.create_user_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_create_user_id": create_user_id_arr
                    }
                })

            if self.site_id is not None and self.site_id != "":
                site_id_arr = self.site_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_site_id": site_id_arr
                    }
                })

        if len(must_condition_list) > 0:
            query['bool']['must'] = must_condition_list

        min_score = 1000 + float(self.score)

        body = {
            "min_score": min_score,
            "size": 0,
            "query": {
                "script_score": {
                    "query": query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1000",
                        "params": {
                            "query_vector": embedding
                        }
                    }
                }
            }
        }
        self.total_query = body.copy()
        body["size"] = self.page_size
        body["from"] = self.offset
        body["collapse"] = {"field": "earliest_video_id.raw"}
        return body

    def to_total_query(self):
        self.total_query["aggs"] = {
            "total_num": {
                "cardinality": {
                    "field": "earliest_video_id.raw"}

            }
        }
        return self.total_query


class ContentVideoPredictEntity:
    def __init__(self, request):
        self.file = request.files.get('file')
        self.score = request.form.get("score", 0.9)
        self.page_num = request.form.get("pageNum", 1)
        self.page_size = request.form.get("pageSize", 10)
        self.saas_flag = request.form.get('office_code')
        self.public_type = request.form.get('public_type')
        self.offset = (int(self.page_num) - 1) * int(self.page_size)
        self.begin_time = request.form.get("beginTime")
        self.end_time = request.form.get("endTime")
        self.library_type = request.form.get("libraryType")
        self.category_id = request.form.get("category_id")
        self.column_id = request.form.get('column_id')
        self.create_user_id = request.form.get('create_user_id')
        self.site_id = request.form.get('site_id')
        self.filter_site = request.form.get('filter_site')
        self.topic_arr = request.form.get('topic_arr')
        self.total_query = ""

    def to_dict(self):
        return {
            "file": self.file,
            "score": self.score,
            "page_num": self.page_num,
            "page_size": self.page_size,
            "saas_flag": self.saas_flag,
            "public_type": self.public_type,
            "begin_time": self.begin_time,
            "end_time": self.end_time,
            "library_type": self.library_type,
            "category_id": self.category_id,
            "column_id": self.column_id,
            "create_user_id": self.create_user_id,
            "site_id": self.site_id,
            "filter_site": self.filter_site,
            "topic_arr": self.topic_arr
        }

    def validate(self):
        logger.info(f"validate request {self.to_dict()}")
        if not self.file:
            raise ValueError("file is required")

        if self.saas_flag is None:
            logger.warn("office_code is None, saas is not started")

    def to_esl_query(self, embedding):
        query = {
            "bool": {
                "must": [
                    {
                        "match_all": {}
                    }
                ],
                "must_not": [
                    {
                        "match_phrase": {
                            "del_flag": "1"
                        }
                    }
                ],
            }
        }
        must_not_arr = [{
            "match_phrase": {
                "del_flag": "1"
            }
        }]
        if self.filter_site is not None and self.filter_site != "":
            site_id_arr = self.filter_site.split(",")
            must_not_arr.append({
                "terms": {
                    "public_site_id": site_id_arr
                }
            })

        query['bool']['must_not'] = must_not_arr

        must_condition_list = []
        # TODO beginTime and endTime 等待对接上
        # begin_time = request.form.get('beginTime')
        # end_time = request.form.get('endTime')
        # if begin_time is None or begin_time == "":
        #     now = datetime.now()
        #     # Get the time three months ago
        #     three_months_ago = now - relativedelta(months=3)
        #     # Format the time strings
        #     end_time = now.strftime('%Y-%m-%d %H:%M:%S')
        #     begin_time = three_months_ago.strftime('%Y-%m-%d %H:%M:%S')
        #     logger.info(f"Current time: {begin_time}")
        #     logger.info(f"Three months ago: {end_time}")
        #
        # must_condition_list.append({
        #     "range": {
        #         "created_at": {
        #             "gte": begin_time,
        #             "lte": end_time
        #         }
        #     }
        # })

        if self.library_type is not None and self.library_type != "":

            if self.library_type == "video":
                must_condition_list.append({
                    "match_phrase": {
                        "tag": "video"
                    }
                })

            if self.library_type == "public":
                must_condition_list.append({
                    "match_phrase": {
                        "from_source": "public"
                    }
                })

                if self.public_type is not None and self.public_type != "":
                    must_condition_list.append({
                        "match_phrase": {
                            "public_type": self.public_type
                        }
                    })

                if self.topic_arr is not None and self.topic_arr != "":
                    must_condition_list.append({
                        "exists": {
                            "field": "public_topic_arr"
                        }
                    })
            else:
                must_condition_list.append({
                    "match_phrase": {
                        "from_source": self.library_type
                    }
                })

            if self.category_id is not None and self.category_id != "":
                category_id_arr = self.category_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_category_id": category_id_arr
                    }
                })

            if self.column_id is not None and self.column_id != "":
                column_id_arr = self.column_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_column_id": column_id_arr
                    }
                })

            if self.create_user_id is not None and self.create_user_id != "":
                create_user_id_arr = self.create_user_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_create_user_id": create_user_id_arr
                    }
                })

            if self.site_id is not None and self.site_id != "":
                site_id_arr = self.site_id.split(",")
                must_condition_list.append({
                    "terms": {
                        self.library_type + "_site_id": site_id_arr
                    }
                })

        if len(must_condition_list) > 0:
            query['bool']['must'] = must_condition_list

        min_score = 1000 + float(self.score)
        body = {
            "min_score": min_score,
            "size": 0,
            "query": {
                "script_score": {
                    "query": query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1000",
                        "params": {
                            "query_vector": embedding
                        }
                    }
                }
            },
            "_source": ["key_id", "hdfs_path", "earliest_video_id", "tag", "from_source", "public_topic_arr"],
        }
        self.total_query = body.copy()
        body["size"] = self.page_size
        body["from"] = self.offset
        body["collapse"] = {"field": "earliest_video_id.raw"}
        return body

    def to_total_query(self):
        self.total_query["aggs"] = {
            "total_num": {
                "cardinality": {
                    "field": "earliest_video_id.raw"}

            }
        }
        return self.total_query


class VideoPredictEntity:

    def __init__(self, request):
        self.file = request.files.get('file')
        self.score = request.form.get("score", 0.9)
        self.page_num = request.form.get("pageNum", 1)
        self.page_size = request.form.get("pageSize", 10)
        self.begin_time = request.form.get("beginTime")
        self.end_time = request.form.get("endTime")
        self.saas_flag = request.form.get('office_code')
        self.offset = (int(self.page_num) - 1) * int(self.page_size)
        self.total_query = ""

    def to_dict(self):
        return {
            "file": self.file,
            "score": self.score,
            "page_num": self.page_num,
            "page_size": self.page_size,
            "begin_time": self.begin_time,
            "end_time": self.end_time,
            "saas_flag": self.saas_flag
        }

    def validate(self):
        logger.info(f"validate request {self.to_dict()}")
        if not self.file:
            raise ValueError("file is required")

        if self.begin_time is None or self.begin_time == "":
            logger.info("beginTime is None, start use default value, past three months ago")
            now = datetime.now()
            # Get the time three months ago
            three_months_ago = now - relativedelta(months=3)
            # Format the time strings
            self.end_time = now.strftime('%Y-%m-%d %H:%M:%S')
            self.begin_time = three_months_ago.strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Current time: {self.begin_time}")
            logger.info(f"Three months ago: {self.end_time}")

        if self.saas_flag is None:
            logger.warn("office_code is None, saas is not started")

    def to_esl_query(self, embedding):
        query = {
            "bool": {
                "must": [
                    {
                        "match_phrase": {
                            "tag": "video"
                        }
                    }
                ],
                "must_not": [
                    {
                        "match_phrase": {
                            "del_flag": "1"
                        }
                    }
                ],
            }
        }

        must_condition_list = [{
            "terms": {
                "tag": ["video", "video-index"]
            }
        }]

        if len(must_condition_list) > 0:
            query['bool']['must'] = must_condition_list

        body = {
            "min_score": float(self.score) + 1000,
            "size": 0,
            "query": {
                "script_score": {
                    "query": query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1000",
                        "params": {
                            "query_vector": embedding
                        }
                    }
                }
            }
        }

        self.total_query = body.copy()
        body["size"] = self.page_size
        body["from"] = self.offset
        body["collapse"] = {"field": "earliest_video_id.raw"}
        return body

    def to_total_query(self):
        self.total_query["aggs"] = {
            "total_num": {
                "cardinality": {
                    "field": "earliest_video_id.raw"}

            }
        }
        return self.total_query
