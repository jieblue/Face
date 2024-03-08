import hashlib
import json
import math
import os
import time
import traceback

from flask import Flask, request, jsonify, Response

from entity.file_entity import VideoFile, ImageFile
from entity.interface_request_entity import MainFaceRequestEntity, MainFaceInsertEntity, FacePredictEntity, \
    MainFacePredictEntity, ContentFacePredictEntity, ContentVideoPredictEntity, VideoPredictEntity
from entity.union_result import UnionResult
from model.model_video import VideoModel
from service import core_service, main_avatar_service, video_service_v3, elasticsearch_service, file_service, \
    visual_algorithm_service, elasticsearch_result_converter
from service.core_service import *
from service.elasticsearch_service import image_faces_v1_index, es_client, main_avatar_v1_index, video_frames_v1_index
from utils import log_util
from utils.img_util import *
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


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
# 获取face_app的配置
face_app_conf = conf['face_app']

download_file = face_app_conf['download_file']

# Create a logger
logger = log_util.get_logger(__name__)

# 加载人脸模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)
video_model = VideoModel('./config/weights/ResNet2048_v224.onnx', gpu_id=0)
logger.info("Video model loaded successfully")

key_frames_path = './keyframes'
key_faces_path = './keyframes_faces'

hdfs_prefix = face_app_conf['hdfs_prefix']
face_predict_dir = face_app_conf['face_predict_dir']
if not os.path.exists(face_predict_dir):
    logger.info(f"face_predict_dir {face_predict_dir} not exists, create it")
    os.makedirs(face_predict_dir)
generator = UniqueGenerator()

app = Flask(__name__)


@app.route('/api/ability/main_face_list', methods=['POST'])
def main_face_list():
    result = {
        "code": 0,
        "msg": "success",
    }

    try:
        # 接收输入参数并且执行验证
        request_param = MainFaceRequestEntity(request)
        request_param.validate()
        # 转换成ESL查询
        query = request_param.to_esl_query()
        original_es_result = elasticsearch_service.main_avatar_search(request_param.saas_flag, query)
        total, construct_result = elasticsearch_result_converter.main_avatar_result_converter(original_es_result)
        result['res'] = construct_result
        result['total'] = total

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        result["code"] = -1
        result["msg"] = str(e)
        return jsonify(result)


@app.route('/api/ability/face_quality', methods=['POST'])
def face_quality():
    """
    人脸质量检测
    :return:
    """
    result = {
        "code": 0,
        "msg": "success",
    }
    file = request.files['file']  # Assuming the file input field is named 'file'

    try:

        if file:
            uuid_filename = generator.generate_unique_value()
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
            video_service_v3.delete_video_file(dir_path)
            logger.info('face_quality delete temp file: ' + dir_path)
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
def determine_face():
    result = {
        "code": 0,
        'face_found_in_image': True,
        "error_message": "success"
    }
    file = request.files['file']  # Assuming the file input field is named 'file'
    if file:
        uuid_filename = generator.generate_unique_value()

        dir_path = face_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        img = cv_imread(dir_path)

        embedding = visual_algorithm_service.turn_to_face_embedding(img, enhance=False, confidence=0.9)
        if len(embedding) == 0:
            result["face_found_in_image"] = False
            result['error_message'] = 'No face found'
        video_service_v3.delete_video_file(dir_path)
        logger.info('determineface delete temp file: ' + dir_path)
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"

    return jsonify(result)


@app.route('/api/ability/compute_sha256', methods=['POST'])
def compute_sha256():
    # 增加MD5的计算
    result = {
        "code": 0,
        "msg": "success",
    }
    try:
        json_data = request.get_json()
        # 必须是本地磁盘路径
        video_path = json_data["videoPath"]
        if not os.path.exists(video_path):
            result["code"] = -1
            result["msg"] = "videoPath is not exists"
            return jsonify(result)
        with open(video_path, 'rb') as fp:
            data = fp.read()
            result['sha256'] = hashlib.md5(data).hexdigest()
            logger.info("sha256: " + result['sha256'])
        if download_file:
            video_service_v3.delete_video_file(video_path)

    except Exception as e:
        traceback.print_exc()
        logger.error("compute_sha256 error", e)
        # handle the exception
        result["code"] = -100
        result['msg'] = "compute sha256 error"
    return jsonify(result)


@app.route('/api/ability/insert_main_avatar', methods=['POST'])
def insert_main_avatar():
    """
    Insert main avatar into Elasticsearch
    :rtype: result
    """
    result = {
        "code": 0,
        "msg": "success",
    }

    try:
        # 接收输入参数并且执行验证
        request_param = MainFaceInsertEntity(request)
        request_param.validate()
        avatar_image = cv_imread(request_param.file)
        embedding = visual_algorithm_service.turn_to_face_embedding(avatar_image, enhance=False)[0]
        # 转换成ESL查询
        exist_query = request_param.determine_face_exist_query(embedding)
        original_es_result = elasticsearch_service.main_avatar_search(request_param.saas_flag, exist_query)
        main_avatar_service.validate_insert_result(original_es_result)
        # Get embedding
        avatar_align_face = face_model.extract_face(avatar_image, enhance=False, confidence=0.99)
        face_score = face_model.tface.forward(avatar_align_face[0])
        insert_embedding = visual_algorithm_service.turn_to_face_embedding(avatar_image, enhance=False, aligned=False,
                                                                           confidence=0.99)[0]

        insert_query = request_param.insert_query(insert_embedding)
        insert_result = elasticsearch_service.main_avatar_insert(request_param.saas_flag, request_param.object_id,
                                                                 insert_query)
        logger.info(f"Insert main face {request_param.object_id} to Elasticsearch. {insert_result}")
        result["objectId"] = request_param.object_id
        result['msg'] = "Insert successful"
        result['qualityScore'] = str(float(face_score))
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        result["code"] = -1
        result["msg"] = str(e)
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
    # Validate parameters
    validate_result = main_avatar_service.validate_parameter(request)
    if validate_result["code"] < 0:
        return jsonify(validate_result)

    score = request.form.get('score')
    if score is None:
        score = 0.4

    force = request.form.get('force')
    if force is None:
        force = False
    else:
        if force == 'true':
            force = True
        else:
            force = False

    # Validate image
    avatar_image = cv_imread(request.files['file'])
    validate_image_result = main_avatar_service.validate_image(avatar_image, face_model)
    if not validate_image_result["validate"]:
        result["code"] = -1
        result["msg"] = validate_image_result["message"]
        return jsonify(result)

    # Prepare data for Elasticsearch
    object_id = request.form.get('objectId')
    hdfs_path = request.form.get('hdfsPath')
    saas_flag = request.form.get('office_code')
    index_name = elasticsearch_service.get_main_avatar_index(saas_flag)

    # Search for the main avatar in Elasticsearch
    res, total = elasticsearch_service.search_face_image(face_model, index_name, avatar_image, enhance=False,
                                                         score=float(score),
                                                         start=0, size=10)

    if len(res['hits']['hits']) == 0:
        result["code"] = -1
        result["msg"] = "该人员主头像不存在，无法更新"
        logger.info(f"该人员主头像不存在，无法更新 {object_id}, HDFS_PATH: {hdfs_path}")
        return jsonify(result)

    logger.info(f"已存在的主头像信息 {res['hits']['hits'][0]}")

    exist_object_id = res['hits']['hits'][0]['_source']['object_id']
    if exist_object_id != object_id:
        result["code"] = -1
        result["msg"] = "该人员主头像已存在，但是人员ID不一致，无法更新"
        logger.info(f"该人员主头像已存在，但是人员ID不一致，无法更新 {object_id}, HDFS_PATH: {hdfs_path}")
        return jsonify(result)

    avatar_align_face = face_model.extract_face(avatar_image, enhance=False,
                                                confidence=0.99)
    face_score = face_model.tface.forward(avatar_align_face[0])
    logger.info(f"{object_id} 人员新头像质量得分 {face_score}")

    # Check if update is needed
    if not force and float(face_score) < float(res['hits']['hits'][0]['_source']['quality_score']):
        result["code"] = -1
        result["msg"] = "该人员主头像质量得分低于已存在主头像，无法更新"
        logger.info(f"该人员主头像质量得分低于已存在主头像，无法更新 {object_id}, HDFS_PATH: {hdfs_path}")
        return jsonify(result)

    # Get embedding
    avatar_align_face = face_model.extract_face(avatar_image, enhance=False, confidence=0.99)
    face_score = face_model.tface.forward(avatar_align_face[0])
    embedding = visual_algorithm_service.turn_to_face_embedding(avatar_image, enhance=False, aligned=False,
                                                                confidence=0.99)
    embedding = core_service.squeeze_faces(embedding)[0]

    body = {
        "doc": {
            "id": object_id,
            "object_id": object_id,
            "embedding": embedding.tolist(),
            "hdfs_path": hdfs_path,
            "quality_score": face_score,
            "recognition_state": "identification"
        }
    }

    # Update in Elasticsearch
    res = es_client.update(index=index_name, id=object_id, body=body)
    logger.info(f"Update main face {object_id} in Elasticsearch. {res}")
    result['total'] = total
    result["objectId"] = object_id
    result['msg'] = "Update successful"
    result['qualityScore'] = str(float(face_score))
    return jsonify(result)


@app.route('/api/ability/update_main_avatar_object', methods=['POST'])
def update_main_avatar_object():
    result = {
        "code": 0,
        "msg": "success",
    }

    object_id = request.form.get('objectId')
    if object_id is None:
        result["code"] = -1
        result["msg"] = "object_id is None"
        return jsonify(result)

    recognition_state = request.form.get('recognitionState')
    if recognition_state is None:
        recognition_state = 'identification'

    new_object_id = request.form.get('newObjectId')
    saas_flag = request.form.get('office_code')
    index_name = elasticsearch_service.get_main_avatar_index(saas_flag)

    if recognition_state != 'identification' and recognition_state != 'unidentification':
        result["code"] = -1
        result["msg"] = "recognitionState is error, value must be identification or unidentification"
        return jsonify(result)

    # Search for the main avatar in Elasticsearch
    search_res = es_client.get(index=index_name, id=object_id)

    if not search_res['found']:
        logger.info(f"object_id: {object_id} not found")
        result["code"] = -1
        result["msg"] = "object_id not found"
        return jsonify(result)

    # Prepare data for Elasticsearch
    body = {
        "doc": {
            "id": object_id,
            "object_id": new_object_id if new_object_id else search_res['_source']['object_id'],
            "embedding": search_res['_source']['embedding'],
            "hdfs_path": search_res['_source']['hdfs_path'],
            "quality_score": search_res['_source']['quality_score'],
            "recognition_state": recognition_state
        }
    }

    # Update in Elasticsearch
    res = es_client.update(index=index_name, id=object_id, body=body)
    logger.info(f"Update main face {object_id} in Elasticsearch. {res}")

    result["objectId"] = object_id
    result['msg'] = "更新成功"
    return jsonify(result)


# =========== V3版本重构造 ==================
@app.route('/api/ability/v3/face_vectorization', methods=['POST'])
def vectorization_v3() -> Response:
    try:

        json_data = request.get_json()
        logger.info(f"face_vectorization json_data: {json_data}")
        video_path = json_data["videoPath"]
        video_id = json_data["videoId"]
        file_name = json_data["fileName"]
        tag = json_data["tag"]
        file_name = video_id
        saas_flag = json_data.get("office_code")
        video_file = VideoFile(file_name=file_name, file_path=video_path, video_id=video_id, tag=tag,
                               saas_flag=saas_flag)

        key_frame_list, face_frame_embedding_list = video_service_v3.process_video_file(video_file)
        data = {
            "key_frame_list": key_frame_list,
            "face_frame_embedding_list": face_frame_embedding_list
        }
        result = UnionResult(code=0, msg="face_vectorization success", data=data)
        json_result = {
            "code": result.code,
            "msg": result.msg,
            "data": result.data
        }
        return jsonify(json_result)

    except Exception as e:
        logger.error("face_vectorization error", e)
        # handle the exception
        result = UnionResult(code=-100, msg="vectorization_v3 error" + str(e), data=None)
        json_result = {
            "code": result.code,
            "msg": result.msg,
            "data": result.data
        }
        return jsonify(json_result)


@app.route('/api/ability/v3/image_vectorization', methods=['POST'])
def image_vectorization_v3() -> Response:
    try:

        json_data = request.get_json()
        logger.info(f"image_vectorization json_data: {json_data}")
        image_path = json_data["imagePath"]
        image_id = json_data["imageId"]
        file_name = json_data["fileName"]
        library_type = json_data["libraryType"]
        if library_type is None or library_type == "":
            raise ValueError("libraryType is empty.")

        file_image_url = json_data.get("fileImageUrl")
        logger.info(f"Need to down load file_image_url: {file_image_url}")
        if file_image_url is not None and file_image_url != "":
            file_service.download_image_file(image_path, file_image_url, image_id)
        tag = json_data["tag"]
        saas_flag = json_data.get("office_code")
        file_name = image_id
        image_file = ImageFile(file_name=file_name, file_path=image_path, video_id=image_id, tag=tag,
                               library_type=library_type, saas_flag=saas_flag)

        key_frame_list, face_frame_embedding_list = video_service_v3.process_image_file(image_file)
        data = {
            "key_frame_list": key_frame_list,
            "face_frame_embedding_list": face_frame_embedding_list
        }
        result = UnionResult(code=0, msg="image_vectorization success", data=data)
        json_result = {
            "code": result.code,
            "msg": result.msg,
            "data": result.data
        }
        return jsonify(json_result)

    except Exception as e:
        logger.error("image_vectorization error", e)
        # handle the exception
        result = UnionResult(code=-100, msg="image_vectorization v3 error" + str(e), data=None)
        json_result = {
            "code": result.code,
            "msg": result.msg,
            "data": result.data
        }
        return jsonify(json_result)


@app.route('/api/ability/face_predict', methods=['POST'])
def face_predict():
    result = {
        "code": 0,
        "msg": "success",
    }
    try:
        # 接收输入参数并且执行验证
        request_param = FacePredictEntity(request)
        request_param.validate()
        # 转换成ESL查询
        if request_param.file is not None:
            image = cv_imread(request_param.file)
            request_param.embedding_arr = visual_algorithm_service.turn_to_face_embedding(image, enhance=False)[0]
        query = request_param.to_esl_query()
        original_es_result = elasticsearch_service.image_faces_search(request_param.saas_flag, query)
        total, construct_result = elasticsearch_result_converter.face_predict_result_converter(original_es_result)
        result['res'] = construct_result
        result['total'] = total
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        result["code"] = -1
        result["msg"] = str(e)
        return jsonify(result)


@app.route('/api/ability/main_face_predict', methods=['POST'])
def main_face_predict():
    result = {
        "code": 0,
        "msg": "success",
    }
    try:
        # 接收输入参数并且执行验证
        request_param = MainFacePredictEntity(request)
        request_param.validate()
        # 转换成ESL查询
        image = cv_imread(request_param.file)
        embedding = visual_algorithm_service.turn_to_face_embedding(image, enhance=False)[0]
        query = request_param.to_esl_query(embedding)
        original_es_result = elasticsearch_service.main_avatar_search(request_param.saas_flag, query)
        total, construct_result = elasticsearch_result_converter.main_avatar_result_converter(original_es_result)
        result['res'] = construct_result
        result['total'] = total
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        result["code"] = -1
        result["msg"] = str(e)
        return jsonify(result)


video_predict_dir = '/tmp/video_predict_tmp'


@app.route('/api/ability/content_face_predict', methods=['POST'])
def content_face_predict():
    result = {
        "code": 0,
        "msg": "success",
        'total': 0
    }
    try:
        # 接收输入参数并且执行验证
        request_param = ContentFacePredictEntity(request)
        request_param.validate()
        # 转换成ESL查询
        image = cv_imread(request_param.file)
        embedding = visual_algorithm_service.turn_to_face_embedding(image, enhance=False)[0]
        query = request_param.to_esl_query(embedding)
        original_es_result = elasticsearch_service.image_faces_search(request_param.saas_flag, query)
        total, construct_result = elasticsearch_result_converter.face_predict_result_converter(original_es_result)
        result['res'] = construct_result
        result['total'] = total
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        result["code"] = -1
        result["msg"] = str(e)
        return jsonify(result)


@app.route('/api/ability/content_video_predict', methods=['POST'])
def content_video_predict():
    result = {"code": 0, "msg": "success", 'total': 0}

    try:
        # 接收输入参数并且执行验证
        request_param = ContentVideoPredictEntity(request)
        request_param.validate()
        # 转换成ESL查询
        image = cv_imread(request_param.file)
        embedding = visual_algorithm_service.get_frame_embedding(image)
        query = request_param.to_esl_query(embedding)
        original_es_result = elasticsearch_service.video_frame_search(request_param.saas_flag, query)
        total, construct_result = elasticsearch_result_converter.content_video_predict_result_converter(
            original_es_result)
        result['res'] = construct_result
        result['total'] = total
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        result["code"] = -1
        result["msg"] = str(e)
        return jsonify(result)


@app.route('/api/ability/video_predict', methods=['POST'])
def video_predict():
    result = {
        "code": 0,
        "msg": "success",
        'total': 0
    }

    try:
        # 接收输入参数并且执行验证
        request_param = VideoPredictEntity(request)
        request_param.validate()
        # 转换成ESL查询
        image = cv_imread(request_param.file)
        embedding = visual_algorithm_service.get_frame_embedding(image)
        query = request_param.to_esl_query(embedding)
        original_es_result = elasticsearch_service.video_frame_search(request_param.saas_flag, query)
        total, construct_result = elasticsearch_result_converter.video_predict_result_converter(
            original_es_result)
        result['res'] = construct_result
        result['total'] = total
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        result["code"] = -1
        result["msg"] = str(e)
        return jsonify(result)


@app.route('/api/ability/delete_relevant_data', methods=['POST'])
def delete_relevant_data():
    result = {
        "code": 0,
        "msg": "success",
    }
    json_data = request.get_json()
    video_id = json_data["videoId"]
    if video_id is None:
        result["code"] = -1
        result["msg"] = "videoId is None"
        return jsonify(result)

    body = {
        "query": {
            "bool": {
                "must": {
                    "match_all": {}
                },
                "filter": {
                    "term": {
                        "earliest_video_id": video_id
                    }
                }
            }
        }
    }

    video_frame_result = es_client.delete_by_query(index=video_frames_v1_index, body=body)
    logger.info(f"delete_relevant_data video_frame_result: {video_frame_result}")
    image_face_result = es_client.delete_by_query(index=image_faces_v1_index, body=body)
    logger.info(f"delete_relevant_data image_face_result: {image_face_result}")

    result['msg'] = "delete_relevant_data success"
    return jsonify(result)


@app.route('/api/ability/index', methods=['POST'])
def normalized_euclidean_distance(l2, dim=512):
    dim_sqrt = math.sqrt(dim)
    return 1 / (1 + l2 / dim_sqrt)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010)
