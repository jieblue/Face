class FaceKeyFrameEmbedding:

    def __init__(self, *, key_id, object_id, quantity_score, embedding, hdfs_path, video_id_arr, earliest_video_id,
                 file_name, frame_num, timestamp, face_num):
        """
        初始化
        :param key_id: 唯一主键ID
        :param object_id: 人员ID
        :param quantity_score: 质量得分
        :param embedding: 人脸向量
        :param hdfs_path: HDFS上的绝对路径
        :param video_id_arr: 视频ID拼接的字符串， 逗号作为分隔符
        :param earliest_video_id: 最早插入到这个人脸的视频ID
        :param file_name: 文件名
        :param frame_num: 视频关键帧的序号
        :param timestamp: 时间戳
        :param face_num: 人脸序号
        """
        self.key_id = key_id
        self.object_id = object_id
        self.quantity_score = quantity_score
        self.embedding = embedding
        self.hdfs_path = hdfs_path
        self.video_id_arr = video_id_arr
        self.earliest_video_id = earliest_video_id
        self.file_name = file_name
        self.frame_num = frame_num
        self.timestamp = timestamp
        self.face_num = face_num

    def to_dict(self):
        return {
            "key_id": self.key_id,
            "object_id": self.object_id,
            "quantity_score": str(float(self.quantity_score)),
            "hdfs_path": self.hdfs_path,
            "video_id_arr": self.video_id_arr,
            "earliest_video_id": self.earliest_video_id,
            "file_name": self.file_name,
            "frame_num": self.frame_num,
            "timestamp": self.timestamp,
            "face_num": self.face_num
        }

class KeyFrameEmbedding:
    def __init__(self, *, key_id, embedding, hdfs_path, earliest_video_id,
                 file_name, frame_num, timestamp):
        """
        初始化
        :param key_id: 唯一主键ID
        :param object_id: 人员ID
        :param quantity_score: 质量得分
        :param embedding: 人脸向量
        :param hdfs_path: HDFS上的绝对路径
        :param video_id_arr: 视频ID拼接的字符串， 逗号作为分隔符
        :param earliest_video_id: 最早插入到这个人脸的视频ID
        :param file_name: 文件名
        :param frame_num: 视频关键帧的序号
        :param timestamp: 时间戳
        :param face_num: 人脸序号
        """
        self.key_id = key_id
        self.embedding = embedding
        self.hdfs_path = hdfs_path
        self.earliest_video_id = earliest_video_id
        self.file_name = file_name
        self.frame_num = frame_num
        self.timestamp = timestamp

    def to_dict(self):
        return {
            "key_id": self.key_id,
            "hdfs_path": self.hdfs_path,
            "earliest_video_id": self.earliest_video_id,
            "file_name": self.file_name,
            "frame_num": self.frame_num,
            "timestamp": self.timestamp,
        }


class MainFaceKeyFrameEmbedding:

    def __init__(self, *, key_id, object_id, quantity_score, embedding=None, hdfs_path, recognition_state):
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
