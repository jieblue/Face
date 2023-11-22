from flask import Flask, request, jsonify
from utils.frame_util import *
import os.path
import time

from model.model_onnx import *
from service.core_service import *
from service import core_service
from utils.img_util import *
from config.config import *
from milvus_tool import local_milvus
from milvus_tool.local_milvus import *
from config.config import *
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
import hashlib
import logging
import subprocess


class UniqueGenerator:
    def __init__(self):
        self.generated_values = set()

    def generate_unique_value(self):
        unique_value = str(uuid.uuid4())
        while unique_value in self.generated_values:
            unique_value = str(uuid.uuid4())
        self.generated_values.add(unique_value)
        return unique_value


# 获取config信息
conf = get_config()

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 获取face_app的配置
face_app_conf = conf['face_app']
# 获取Milvus的配置
milvus_conf = conf['milvus']
log_file = face_app_conf['log_file']

# Define the log file and format
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

# 加载人脸模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)

logger.info("Milvus 配置信息： " + str(conf['milvus']))

connections.connect("default", host=milvus_conf["host"], port=milvus_conf["port"])

image_faces_v1_name = face_app_conf["image_face_collection"]

has = utility.has_collection(image_faces_v1_name)
logger.info(f"Milvus collection {image_faces_v1_name} exist in Milvus: {has}")

# 人像库索引
image_faces_fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
    FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="hdfs_path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="quality_score", dtype=DataType.FLOAT, max_length=256)
]
image_faces_schema = CollectionSchema(image_faces_fields, "image_faces_v1 is the simplest demo to introduce the APIs")
image_faces_v1 = Collection(image_faces_v1_name, image_faces_schema)



# 主头像二级人像库
# 人像库索引
main_avatar_fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
    FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="hdfs_path", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="quality_score", dtype=DataType.FLOAT, max_length=256)

]
main_avatar_schema = CollectionSchema(main_avatar_fields, "image_faces_v1 is the simplest demo to introduce the APIs")
main_avatar_v1 = Collection(face_app_conf["main_avatar_collection"], main_avatar_schema)

index = {
    "index_type": 'IVF_SQ8',
    "metric_type": "IP",
    "params": {"nlist": 100},
}

image_faces_v1.create_index("embedding", index)
main_avatar_v1.create_index("embedding", index)
logger.info(f"Index {index} created successfully")

image_faces_v1.load()
main_avatar_v1.load()

logger.info(f"Collection {image_faces_v1_name} loaded successfully")

key_frames_path = './keyframes'
key_faces_path = './keyframes_faces'

hdfs_prefix = face_app_conf['hdfs_prefix']
face_predict_dir = face_app_conf['face_predict_dir']
generator = UniqueGenerator()

app = Flask(__name__)
"""
{
    'frame': 视频帧的数据,
    'timestamp': 视频帧的时间戳,
    'frame_num': 第几个视频关键帧,
    'unique_filename': 文件唯一标识名,
}
"""


@app.route('/api/ability/face_vectorization', methods=['POST'])
def face_vectorization():
    result = {
        "code": 0,
        "msg": "success",
    }
    try:
        start_time = time.time()
        json_data = request.get_json()
        logger.info(f"face_vectorization json_data: {json_data}")
        # 必须是本地磁盘路径
        video_path = json_data["videoPath"]
        video_id = json_data["videoId"]
        if not os.path.exists(video_path):
            result["code"] = -1
            result["msg"] = "视频路径不存在"
            logger.warn(f"视频路径不存在 {video_id}")

            return jsonify(result)
        # 获取文件名唯一标识
        unique_filename = video_id
        result['unique_filename'] = unique_filename

        logger.info(f"Processing video file: {unique_filename} path {video_path}")

        # Extract key frames from video
        key_frames_info_list = core_service.extract_video(video_path, unique_filename)
        logger.info(
            f"Extracted {len(key_frames_info_list)} key frames from video in {time.time() - start_time:.2f} seconds")

        # Write key frames to disk
        core_service.save_frame_to_disk(key_frames_info_list, key_frames_path, unique_filename)
        logger.info(f"Wrote {len(key_frames_info_list)} key frames to disk in {time.time() - start_time:.2f} seconds")

        # Detect faces in key frames
        core_service.get_align_faces_batch(face_model, key_frames_info_list, True, 0.99)
        face_num = 0
        for key_frame_info in key_frames_info_list:
            face_num += len(key_frame_info['align_face'])
        logger.info(f"Detected {str(face_num)} faces in key frames in {time.time() - start_time:.2f} seconds")

        # Write faces to disk
        faces_paths = core_service.save_face_to_disk(key_frames_info_list, key_faces_path, unique_filename)
        logger.info(f"Wrote {face_num} faces to disk in {time.time() - start_time:.2f} seconds")

        logger.info("filepahts:" + str(faces_paths))
        # Extract facial embeddings from faces
        core_service.get_face_embeddings(face_model, faces_paths, aligned=False, enhance=False, confidence=0.99,
                                         merge=False)

        # 0 face_embedding_id
        # 1 object_id
        # 2 embedding
        # 3 hdfs_path
        entities = [[], [], [], [], []]

        face_result_list = []
        for i, face_info in enumerate(faces_paths):
            if (len(face_info['embedding'])) > 0:
                # 获取人脸质量得分
                face_score = core_service.get_face_quality_single_img(face_model, face_info['face_file_path'])
                logger.info("face_score:" + str(face_score))
                entities[0].append(face_info['face_embedding_id'])
                entities[2].append(core_service.squeeze_faces(face_info['embedding'])[0])
                hdfs_file = hdfs_prefix + face_info['unique_filename'] + "/" + face_info['face_embedding_id'] + ".jpg"
                face_info['hdfs_path'] = hdfs_file
                entities[3].append(hdfs_file)
                entities[4].append(face_score)
                face_info['quality_score'] = str(face_score)
                # search_params = {
                #     "metric_type": "IP",
                #     "ignore_growing": False,
                #     "params": {"nprobe": 50}
                # }
                # # 查找看是否存在主头像索引中
                # res = core_service.search_face_image(face_model, main_avatar_v1, [face_info['face_file_path']],
                #                                      enhance=False, score=float(0.5), limit=int(10),
                #                                      search_params=search_params)
                # if len(res[0]) > 0:
                #     main_avatar_id = res[0][0]['id']
                #     # 主头像存在， 讲该头像关联到该次人员ID中
                #     logger.info(f"主头像库中存在 ID {main_avatar_id}")
                #     entities[1].append(res[0][0]['id'])
                # else:
                #     logger.info(f"主头像库中不存在 ID {face_info['face_embedding_id']}")
                #     # 判断是否该主头像的得分是否高于现有主头像的得分
                entities[1].append('unidentification')
                face_result_list.append(face_info)

        res = "No face found in this video"
        # Insert embeddings into Milvus
        if len(entities[0]) > 0:
            res = image_faces_v1.insert(entities)
        logger.info(f"image_faces_v1.insert res: {res}")
        logger.info(f"Inserted vectors into Milvus in {time.time() - start_time:.2f} seconds")

        result["msg"] = f"Processed video file {unique_filename} in {time.time() - start_time:.2f} seconds"
        # 删除一些不必要的元素
        for i, face_info in enumerate(face_result_list):
            del face_info['frame']
            del face_info['embedding']
            del face_info['single_align_face']

        result['face_info_list'] = face_result_list
        # video_id_file_path = face_app_conf["video_id_file_path"] + "/video_id_file.txt"
        # command_out_result = subprocess.run('echo "' + video_id + '" >> ' + video_id_file_path,
        #                                     capture_output=True, shell=True)
        # logger.info("command_out_result: " + str(command_out_result))


    except Exception as e:
        traceback.print_exc()
        # handle the exception
        result["code"] = -100
        logger.error("face_vectorization error", e)

    return result


@app.route('/api/ability/face_predict', methods=['POST'])
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
    logger.info("score:" + str(score))
    logger.info("page_num:" + str(page_num))
    logger.info("page_size:" + str(page_size))

    offset = (int(page_num) - 1) * int(page_size)


    if file:
        uuid_filename = generator.generate_unique_value()
        logger.info("uuid_filename: " + uuid_filename)

        dir_path = face_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        img1 = cv_imread(dir_path)

        # 批量检索人脸图片， 每张人脸图片只能有一张人脸
        imgs = [img1]
        start = time.time()

        search_params = {
            "metric_type": "IP",
            "offset": offset,
            "ignore_growing": False,
            "params": {"nprobe": 50}
        }
        res = core_service.search_face_image(face_model, image_faces_v1, imgs,
                                             enhance=False, score=float(score), limit=int(page_size),
                                             search_params=search_params)
        logger.info('搜索耗时: ' + str(time.time() - start))
        logger.info("搜索结果: ")
        logger.info(res)
        result['res'] = res
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"
    return jsonify(result)


@app.route('/api/ability/main_face_predict', methods=['POST'])
def main_face_predict():
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
    logger.info("score:" + str(score))
    logger.info("page_num:" + str(page_num))
    logger.info("page_size:" + str(page_size))

    offset = (int(page_num) - 1) * int(page_size)
    search_params = {
        "metric_type": "IP",
        "offset": offset,
        "ignore_growing": False,
        "params": {"nprobe": 50}
    }

    if file:
        uuid_filename = generator.generate_unique_value()
        logger.info("uuid_filename: " + uuid_filename)

        dir_path = face_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        img1 = cv_imread(dir_path)

        # 批量检索人脸图片， 每张人脸图片只能有一张人脸
        imgs = [img1]
        start = time.time()
        res = core_service.search_face_image(face_model, main_avatar_v1, imgs,
                                             enhance=False, score=float(score), limit=int(page_size),
                                             search_params=search_params)
        logger.info('搜索耗时: ' + str(time.time() - start))
        logger.info("搜索结果: ")
        logger.info(res)
        result['res'] = res
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"
    return jsonify(result)


@app.route('/api/ability/face_quality', methods=['POST'])
def face_quality():
    result = {
        "code": 0,
        "msg": "success",
    }
    file = request.files['file']  # Assuming the file input field is named 'file'


    try:

        if file:
            uuid_filename = generator.generate_unique_value()
            logger.info("uuid_filename: " + uuid_filename)

            dir_path = face_predict_dir + uuid_filename + ".jpg"
            file.save(dir_path)  # Replace with the path where you want to save the file

            img1 = cv_imread(dir_path)

            # 批量检索人脸图片， 每张人脸图片只能有一张人脸
            images = [img1]
            start = time.time()

            scores = core_service.get_face_quality_batch_img(face_model, images)
            logger.info('获取质量分数耗时: ' + str(time.time() - start))
            logger.info('质量分数: ' + str(scores))
            if len(scores) == 0:
                result["code"] = -1
                result["msg"] = "No face found"
                return jsonify(result)
            result['scores'] = str(scores[0])
        else:
            result["code"] = -1
            result["msg"] = "File uploaded Failure!"
    except Exception as e:
        traceback.print_exc()
        # handle the exception
        result["code"] = -100
        logger.error("face_quality error", e)
    return jsonify(result)

@app.route('/api/ability/determineface', methods=['POST'])
def determineface():
    result = {
        "code": 0,
        'face_found_in_image': True,
        "error_messag": "success"
    }
    file = request.files['file']  # Assuming the file input field is named 'file'
    if file:
        uuid_filename = generator.generate_unique_value()
        logger.info("uuid_filename: " + uuid_filename)

        dir_path = face_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        img = cv_imread(dir_path)

        embedding = face_model.turn2embeddings(img, enhance=False)
        if len(embedding) == 0:
            result["face_found_in_image"] = False
            result['error_message'] = 'No face found'
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"

    return jsonify(result)


@app.route('/api/ability/compute_sha256', methods=['POST'])
def compute_sha256():
    result = {
        "code": 0,
        "msg": "success",
    }
    try:
        json_data = request.get_json()
        # 必须是本地磁盘路径
        video_path = json_data["videoPath"]
        with open(video_path, 'rb') as fp:
            data = fp.read()
            result['sha256'] = hashlib.md5(data).hexdigest()
            logger.info("sha256: " + result['sha256'])
    except Exception as e:
        traceback.print_exc()
        logger.error("compute_sha256 error", e)
        # handle the exception
        result["code"] = -100
        result['msg'] = "compute sha256 error"
    return jsonify(result)


"""
插入主头像到二级索引主人像库
"""
@app.route('/api/ability/insert_main_avatar', methods=['POST'])
def insert_main_avatar():
    result = {
        "code": 0,
        "msg": "success",
    }
    file = request.files['file']  # Assuming the file input field is named 'file'

    object_id = request.form.get('objectId')
    if object_id is None:
        result["code"] = -1
        result["msg"] = "objectId is None"
        return jsonify(result)

    hdfs_path = request.form.get('hdfsPath')
    if hdfs_path is None:
        result["code"] = -1
        result["msg"] = "hdfsPath is None"
        return jsonify(result)

    score = request.form.get('score')
    if score is None:
        score = 0.4

    search_params = {
        "metric_type": "IP",
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }

    if file:
        uuid_filename = generator.generate_unique_value()
        print("uuid_filename")
        print(uuid_filename)

        dir_path = face_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        img1 = cv_imread(dir_path)
        # 批量检索人脸图片， 每张人脸图片只能有一张人脸
        imgs = [img1]
        start = time.time()
        res = core_service.search_face_image(face_model, main_avatar_v1, imgs,
                                             enhance=False, score=float(score), limit=10,
                                             search_params=search_params)
        print('主头像: ' + str(res))
        if len(res[0]) > 0:
            result["code"] = -1
            result["msg"] = "主头像已存在"
            return jsonify(result)

        embedding = face_model.turn2embeddings(img1, enhance=False)
        if len(embedding) == 0:
            result["code"] = -1
            result["msg"] = "No face found"
            return jsonify(result)

        entities = [[], [], [], []]
        entities[0].append(uuid_filename)
        entities[1].append(object_id)
        entities[2].append(embedding[0])
        entities[3].append(hdfs_path)
        main_avator_res = main_avatar_v1.insert(entities)

        print('插入结果耗时: ' + str(time.time() - start))
        print("搜索结果: ")
        print(main_avator_res)

        result["uuidFilename"] = uuid_filename
        result['res'] = "插入成功"
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"
    return jsonify(result)


"""
更新主头像到二级索引主人像库
"""
@app.route('/api/ability/update_main_avatar', methods=['POST'])
def update_main_avatar():
    result = {
        "code": 0,
        "msg": "success",
    }
    file = request.files['file']  # Assuming the file input field is named 'file'

    object_id = request.form.get('objectId')
    if object_id is None:
        result["code"] = -1
        result["msg"] = "objectId is None"
        return jsonify(result)

    hdfs_path = request.form.get('hdfsPath')
    if hdfs_path is None:
        result["code"] = -1
        result["msg"] = "hdfsPath is None"
        return jsonify(result)

    uuid_filename = request.form.get('uuidFilename')
    if uuid_filename is None:
        result["code"] = -1
        result["msg"] = "uuidFilename is None"
        return jsonify(result)

    if file:
        print("uuid_filename")
        print(uuid_filename)

        dir_path = face_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        img1 = cv_imread(dir_path)
        # 批量检索人脸图片， 每张人脸图片只能有一张人脸
        start = time.time()

        embedding = face_model.turn2embeddings(img1, enhance=False)

        entities = [[], [], [], []]
        entities[0].append(uuid_filename)
        entities[1].append(object_id)
        entities[2].append(embedding[0])
        entities[3].append(hdfs_path)
        main_avatar_v1.upsert(entities)

        print('插入结果耗时: ' + str(time.time() - start))
        print("搜索结果: ")
        result['uuidFilename'] = uuid_filename
        result['res'] = "更新成功 向量ID" + uuid_filename
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!" + uuid_filename
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='192.168.100.19', port=5010)
