class FaceKeyFrameEmbedding:

    def __init__(self, key_id, object_id, quantity_score, embedding, hdfs_path, video_id_arr, earliest_video_id, file_name):
        """
        初始化
        :param key_id: 唯一主键ID
        :param object_id: 人员ID
        :param quantity_score: 质量得分
        :param embedding: 人脸向量
        :param hdfs_path: HDFS上的绝对路径
        :param video_id_arr: 视频ID拼接的字符串， 逗号作为分隔符
        :param earliest_video_id: 最早插入到这个人脸的视频ID
        """
        self.key_id = key_id
        self.object_id = object_id
        self.quantity_score = quantity_score
        self.embedding = embedding
        self.hdfs_path = hdfs_path
        self.video_id_arr = video_id_arr
        self.earliest_video_id = earliest_video_id
        self.file_name = file_name

    def to_dict(self):
        return {
            "key_id": self.key_id,
            "object_id": self.object_id,
            "quantity_score": str(float(self.quantity_score)),
            "hdfs_path": self.hdfs_path,
            "video_id_arr": self.video_id_arr,
            "earliest_video_id": self.earliest_video_id,
            "file_name": self.file_name
        }


class MainFaceKeyFrameEmbedding:

    def __init__(self, key_id, object_id, quantity_score, embedding, hdfs_path, recognition_state):
        """
        初始化
        :param key_id: 唯一主键ID
        :param object_id: 人员ID
        :param quantity_score: 质量得分
        :param embedding: 人脸向量
        :param hdfs_path: HDFS上的绝对路径
        """
        self.key_id = key_id
        self.object_id = object_id
        self.quantity_score = quantity_score
        self.embedding = embedding
        self.hdfs_path = hdfs_path
        self.recognition_state = recognition_state
