from flask import Flask, request, jsonify
import os.path
import time

from model.model_video import *
from service import video_service
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db,

)
import traceback

import uuid
import logging
from config.config import *



class UniqueGenerator:
    def __init__(self):
        self.generated_values = set()

    def generate_unique_value(self):
        unique_value = str(uuid.uuid4())
        while unique_value in self.generated_values:
            unique_value = str(uuid.uuid4())
        self.generated_values.add(unique_value)
        return unique_value


app = Flask(__name__)
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the log file and format
log_file = 'video_app.log'
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a file handler and set the log format
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_format)

# Create a stream handler to print log messages to the console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

conf = get_config()
milvus_conf = conf['milvus']


video_model = VideoModel('./config/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', gpu_id=0)
logger.info("Video model loaded successfully")



connections.connect("default", host=milvus_conf["host"], port=milvus_conf["port"])
logger.info("Milvus connection established successfully")


video_frame_v1_name = "video_faces_v1"
hdfs_prefix = "/VIDEO_FACE_TEST/video/"

has = utility.has_collection(video_frame_v1_name)
logger.info(f"Does collection {video_frame_v1_name} exist in Milvus: {has}")

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),
    FieldSchema(name="hdfs_path", dtype=DataType.VARCHAR, max_length=256),
]
schema = CollectionSchema(fields, "video_frame_v1_name is the simplest demo to introduce the APIs")
video_frame_v1 = Collection(video_frame_v1_name, schema)
logger.info(f"Collection {video_frame_v1_name} created successfully")

index = {
    "index_type": 'IVF_FLAT',
    "metric_type": "L2",
    "params": {"nlist": 128},
}

video_frame_v1.create_index("embedding", index)
logger.info(f"Index {index} created successfully")

video_frame_v1.load()

logger.info(f"Collection {video_frame_v1_name} loaded successfully")

key_frames_path = './keyframes'
key_faces_path = './keyframes_faces'


@app.route('/api/ability/video_vectorization', methods=['POST'])
def video_vectorization():
    result = {
        "code": 0,
        "msg": "success",
    }
    try:
        start_time = time.time()
        json_data = request.get_json()
        # 必须是本地磁盘路径
        video_path = json_data["videoPath"]
        video_id = json_data["videoId"]
        logger.info(f"video_path: {video_path}")
        logger.info(f"video_id: {video_id}")
        if not os.path.exists(video_path):
            result["code"] = -1
            result["msg"] = "视频路径不存在"
            return jsonify(result)
        # 获取文件名唯一标识
        unique_filename = video_id
        logger.info(f"Processing video file: {unique_filename}")

        # Extract key frames from video
        key_frames_info_list = video_service.extract_video(video_path, unique_filename)
        print(f"Extracted {len(key_frames_info_list)} key frames from video in {time.time() - start_time:.2f} seconds")
        logger.info(
            f"Extracted {len(key_frames_info_list)} key frames from video in {time.time() - start_time:.2f} seconds")

        # Write key frames to disk
        video_service.save_frame_to_disk(key_frames_info_list, key_frames_path, unique_filename)
        logger.info(f"Wrote {len(key_frames_info_list)} key frames to disk in {time.time() - start_time:.2f} seconds")

        # 获取关键帧向量
        frame_feature_list = video_service.get_frames_feature(video_model, key_frames_info_list)
        logger.info( f"Extracted {len(frame_feature_list)} key frames features from video in {time.time() - start_time:.2f} seconds")

        entities = [[], [], []]

        frame_result_list = []
        for i, frame_info in enumerate(frame_feature_list):
            length = len(frame_info['frame_feature'])
            if length > 0:
                entities[0].append(frame_info['frame_embedding_id'])
                entities[1].append(frame_info['frame_feature'])
                hdfs_file = hdfs_prefix + frame_info['unique_filename'] + "/" + frame_info[
                    'frame_embedding_id'] + ".jpg"
                frame_info['hdfs_path'] = hdfs_file
                entities[2].append(hdfs_file)
                frame_result_list.append(frame_info)

        # Insert embeddings into Milvus
        res = video_frame_v1.insert(entities)
        print(res)
        print(f"Inserted vectors into Milvus in {time.time() - start_time:.2f} seconds")

        result["msg"] = f"Processed video file {unique_filename} in {time.time() - start_time:.2f} seconds"

        # 删除一些不必要的元素
        for i, frame_info in enumerate(frame_result_list):
            del frame_info['frame']
            del frame_info['frame_feature']

        # result['milvs_msg'] = res
        result['face_info_list'] = frame_result_list
    except Exception as e:
        traceback.print_exc()
        # handle the exception
        result["code"] = -100
        result['unique_filename'] = unique_filename

    return jsonify(result)


video_predict_dir = '/tmp/video_predict_tmp'
generator = UniqueGenerator()


@app.route('/api/ability/video_predict', methods=['POST'])
def face_predict():
    result = {
        "code": 0,
        "msg": "success",
    }
    file = request.files['file']  # Assuming the file input field is named 'file'
    score = request.form.get('score')
    if score is None:
        score = 0.4
    # limit = request.form.get('limit')
    page_num = request.form.get('pageNum')
    if page_num is None:
        page_num = 1
    page_size = request.form.get('pageSize')
    if page_size is None:
        page_size = 10

    offset = (int(page_num) - 1) * int(page_size)
    search_params = {
        "metric_type": "L2",
        "offset": offset,
        "params": {"nprobe": 20},
    }

    if file:
        uuid_filename = generator.generate_unique_value()
        print("uuid_filename")
        print(uuid_filename)

        dir_path = video_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        # img1 = cv_imread(dir_path)

        # 批量检索人脸图片， 每张人脸图片只能有一张人脸
        # imgs = [img1]
        start = time.time()
        search_vectors = video_model.get_frame_embedding(dir_path)
        # res = video_service.search_face_image(face_model, image_faces_v1, imgs,
        #                                             enhance=False, score=float(score), limit=int(page_size), 
        #                                             search_params=search_params)

        res = video_frame_v1.search([search_vectors], 'embedding', search_params, limit=int(page_size),
                                    output_fields=['hdfs_path'])
        frame_result = []
        for one in res:
            _result = []
            for single in one:
                # print(single)
                tmp = {
                    # 'primary_key': single.id,
                    'id': single.entity.id,
                    'score': single.distance,
                    'hdfs_path': single.entity.hdfs_path
                }
                # get_search_result(single.id, single.entity.user_id, single.score)
                _result.append(tmp)
            frame_result.append(_result)
        print('搜索耗时: ' + str(time.time() - start))
        print("搜索结果: ")
        print(res)
        print("搜索结果 res[0]: ")
        print(res[0])
        result['res'] = frame_result
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"

    print('Result')
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='192.168.100.19', port=5010)
