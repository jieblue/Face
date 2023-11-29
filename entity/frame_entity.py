# 获取config信息
from config.config import get_config

conf = get_config()
# 获取face_app的配置
face_app_conf = conf['face_app']
hdfs_prefix = face_app_conf['hdfs_prefix']


class KeyFrame:

    def __init__(self, *, file_name, video_id, frame_num, timestamp, frame):
        """
        :param file_name: 文件名
        :param frame_num: 视频关键帧的序号
        :param timestamp: 时间戳
        :param frame: 关键帧的源图片
        :param video_id: 视频id
        """
        self.file_name = file_name
        self.frame_num = frame_num
        self.timestamp = timestamp
        self.frame = frame
        self.video_id = video_id
        # TODO 增加关键帧在HDFS中所在位置
        self.hdfs_path = None
        # 关键帧的唯一ID
        self.key_id = self.generate_key_id()

    def generate_key_id(self):
        return self.file_name + "_" + str(self.frame_num) + "_" + str(self.timestamp)


class FaceKeyFrame:
    def __init__(self, file_name, video_id, frame_num, timestamp, face_num, face_frame):
        self.file_name = file_name
        self.frame_num = frame_num
        self.timestamp = timestamp
        self.video_id = video_id
        self.face_num = face_num
        self.face_frame = face_frame
        # TODO 增加人脸在HDFS中所在位置
        self.hdfs_path = None

        # TODO 增加人脸在此关键帧图片中所在位置
        # self.face_location = face_location
        self.key_id = self.generate_key_id()

    def generate_key_id(self):
        return self.file_name + "_" + str(self.frame_num) + "_" + str(self.timestamp) + "_" + str(self.face_num)

    def generate_hdfs_path(self):
        return hdfs_prefix + "/" + self.file_name + "/" + str(self.key_id) + ".jpg"
