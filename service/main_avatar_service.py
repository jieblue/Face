import logging

from config.config import get_config
from model.model_onnx import Face_Onnx

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


def validate_image(image, face_model: Face_Onnx):
    """
    Validate the input image.

    Args:
        image (bytes): The image bytes to be validated.

    Returns:
        bool: Whether the image is valid.
    """
    validate_result = {
        'validate': True,
        'message': ''
    }
    message = ''
    # Check if the image is empty
    if image is None:
        message = 'Image is empty.'
        validate_result['validate'] = False
        validate_result['message'] = message
        logger.error(message)
        return validate_result

    # Check if the image is too large
    file_size_bytes = len(image)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > face_app_conf['max_image_size_mb']:
        message = 'Image is too large: {}MB. Max image size is {}MB.'.format(
            file_size_mb, face_app_conf['max_image_size_mb'])
        validate_result['validate'] = False
        validate_result['message'] = message
        logger.error(message)
        return validate_result

    align_face_list = face_model.extract_face(image, enhance=False,
                                              confidence=0.99)
    if len(align_face_list) == 0:
        message = 'No face detected.'
        validate_result['validate'] = False
        validate_result['message'] = message
        logger.error(message)
        return validate_result

    elif len(align_face_list) > 1:
        message = 'More than one face detected.'
        validate_result['validate'] = False
        validate_result['message'] = message
        logger.error(message)
        return validate_result

    return validate_result


def validate_parameter(request):
    """
    Validate the parameters of main_avatar.
    :param request:
    :return: result
    """
    result = {
        "code": 0,
        "msg": "success",
    }
    object_id = request.form.get('objectId')
    if object_id is None:
        result["code"] = -1
        result["msg"] = "objectId is None"
        return result

    hdfs_path = request.form.get('hdfsPath')
    if hdfs_path is None:
        result["code"] = -1
        result["msg"] = "hdfsPath is None"
        return result

    return result


def validate_insert_result(original_es_result):
    interior_result = original_es_result['hits']['hits']
    if len(interior_result) > 0:
        raise ValueError(f"主头像已存在:{interior_result[0]['_source']['object_id']}:"
                         f"{interior_result[0]['_source']['recognition_state']}："
                         f"{interior_result[0]['_source']['embedding']}：")


