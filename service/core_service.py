import logging
import os.path

from config.config import *
from milvus_tool.local_milvus import *
from model.model_onnx import Face_Onnx
from utils.frame_util import *
from utils.img_util import *


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


def extract_video(video_path, unique_filename):
    """
    Extracts faces from a video file.

    Args:
        video_path (str): The path to the video file.

    Returns:
        List[ndarray]: A list of NumPy arrays, each representing a face image.
    """
    # return a list of nparray(bgr image)
    # print(video_path)
    try:
        container = av.open(video_path)
    except:
        return None

    result = []
    # Signal that we only want to look at keyframes.
    stream = container.streams.video[0]

    stream.codec_context.skip_frame = 'NONKEY'
    frames = container.decode(stream)
    frame_num = 1
    for frame in frames:
        # print(stream.time_base)
        timestamp = round(frame.pts * stream.time_base)
        np_frame = av_frame2np(frame)
        result.append({
            'frame': np_frame,
            'timestamp': timestamp,
            'frame_num': frame_num,
            'unique_filename': unique_filename
        })
        frame_num = frame_num + 1

    return result


def av_frame2np(frame):
    img_frame = frame.to_image()
    np_frame = np.array(img_frame)
    np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    return np_frame


def get_align_faces_batch(model: Face_Onnx, paths,
                          enhance=False, confidence=0.99, merge=False):
    """
    Returns a batch of aligned faces from the given image paths using the specified model.

    Args:
        model (Face_Onnx): The face detection and alignment model to use.
        paths (List[str]): A list of file paths to the images to process.
        enhance (bool, optional): Whether to enhance the faces before alignment. Defaults to False.
        confidence (float, optional): The minimum confidence threshold for face detection. Defaults to 0.99.
        merge (bool, optional): Whether to merge the aligned faces into a single image. Defaults to False.

    Returns:
        List[Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]]: A list of aligned faces or tuples of aligned faces and their corresponding bounding boxes.
    """

    face_num = 0
    for keyframe_path_info in paths:
        path = keyframe_path_info['frame_file_path']
        img = cv_imread(path)
        face_num += face_num
        if img is None:
            keyframe_path_info['is_success'] = False
            keyframe_path_info['err'] = "图片读取失败"
            continue

        _align_face = model.extract_face(img, enhance=enhance,
                                         confidence=confidence)

        if merge:
            raise NotImplementedError
        else:
            keyframe_path_info['align_face'] = _align_face

            keyframe_path_info['face_num'] = face_num
            logger.info(path + " get_align_faces_batch: " + str(face_num))


def get_face_embeddings(model: Face_Onnx, paths, aligned=False,
                        enhance=False, confidence=0.99, merge=False):
    """
    Returns the face embeddings for the given image paths using the specified model.

    Args:
        model (Face_Onnx): The face recognition model to use.
        paths (List[str]): A list of file paths to the images to process.
        aligned (bool, optional): Whether to perform face alignment before computing embeddings. Defaults to True.
        merge (bool, optional): Whether to merge multiple faces in the same image into a single embedding. Defaults to False.

    Returns:
        List[np.ndarray]: A list of face embeddings, where each embedding is a numpy array of shape (128,).
    """
    for face_info in paths:
        path = face_info['face_file_path']
        img = cv_imread(path)
        if img is None:
            face_info['is_success'] = False
            face_info['err'] = '人脸读取失败'
            continue

        embedding = model.turn2embeddings(img, enhance=enhance, aligned=aligned,
                                          confidence=confidence)
        if merge:
            for single in embedding:
                raise NotImplementedError
                # embeddings.append(single)
        else:
            face_info['embedding'] = embedding


# @app.route('/api/ability/squeeze_faces', methods=['POST'])
def squeeze_faces(faces_list, threshold=0.48):
    """
    Squeezes the faces in the given list based on the given threshold.

    Args:
        faces_list (list): A list of faces to be squeezed.
        threshold (float): The threshold value for squeezing the faces. Default is 0.48.

    Returns:
        list: A list of squeezed faces.
    """
    faces = np.array(faces_list)
    _len = len(faces_list)

    # numpy to tensor
    faces_tensor = torch.from_numpy(faces).float()
    unique_vectors = []
    # ids = []
    for i, vector in enumerate(faces_tensor):

        # 检查是否与之前的向量重复
        is_duplicate = False
        vector_tensor = vector.unsqueeze(0)
        # print(vector_tensor.size())
        for x in unique_vectors:
            x_tensor = x.unsqueeze(0)
            # print(x_tensor.size())
            # 计算余弦相似度
            if torch.nn.functional.cosine_similarity(vector_tensor, x_tensor) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_vectors.append(vector)
            # ids.append(i)

    numpy_list = [t.unsqueeze(0).numpy().tolist()[0] for t in unique_vectors]
    # 从有范数的向量列表中提取没有范数的向量列表.astype(np.float32)
    return numpy_list


def save_frame_to_disk(key_frames, key_frames_path, unique_filename):
    """
    Save key frames to disk.

    Args:
        key_frames (list): A list of key frames.
        key_frames_path (str): The path to the directory where the key frames will be saved.
        unique_filename (str): A unique filename for the key frames.

    Returns:
        list: A list of file paths for the saved key frames.
    """
    # Write key frames to disk
    for i, keyframe in enumerate(key_frames):
        frame = keyframe['frame']
        timestamp = keyframe['timestamp']
        dir_path = os.path.join(key_frames_path, unique_filename)
        os.makedirs(dir_path, exist_ok=True)
        file_name = f"{i}_{timestamp}.jpg"
        file_path = os.path.join(dir_path, file_name)
        keyframe['frame_file_name'] = file_name
        keyframe['frame_file_path'] = file_path
        cv2.imwrite(file_path, frame)


def save_face_to_disk(key_frames_info_list, key_faces_path, unique_filename):
    """
    Save detected faces to disk.

    Args:
        faces (list): A list of detected faces.
        key_faces_path (str): The path to save the faces.
        unique_filename (str): A unique filename to identify the faces.

    Returns:
        list: A list of file paths where the faces are saved.
    """
    # Write key frames to disk
    face_info_list = []
    face_num = 0
    for key_frames_info in key_frames_info_list:

        for keyframes_face_info in key_frames_info['align_face']:
            face_num = face_num + 1
            face_info = {}
            face_info['frame'] = key_frames_info['frame']
            face_info['timestamp'] = key_frames_info['timestamp']
            face_info['frame_num'] = key_frames_info['frame_num']
            face_info['unique_filename'] = key_frames_info['unique_filename']

            dir_path = os.path.join(key_faces_path, unique_filename)
            os.makedirs(dir_path, exist_ok=True)

            face_embedding_id = key_frames_info['unique_filename'] + "_" + str(
                key_frames_info['frame_num']) + "_" + str(key_frames_info["timestamp"]) + "_" + str(face_num)
            face_info['face_embedding_id'] = face_embedding_id
            file_name = f"{face_embedding_id}.jpg"
            file_path = os.path.join(dir_path, file_name)
            logger.info("save_face_to_disk: " + file_path)
            face_info['face_file_path'] = file_path
            face_info_list.append(face_info)
            face_info['single_align_face'] = keyframes_face_info
            cv2.imwrite(file_path, keyframes_face_info)

    return face_info_list


def add_embeddings2milvus(collection, faces, flush=False):
    data = [[], []]
    result = []
    for single in faces:
        id = str(single['id'])
        if len(id) > 50:
            return []

    for i, face in enumerate(faces):
        id = face['id']
        id = str(id)
        embedding = face['embedding']
        data[0].append(id)
        data[1].append(embedding)

        if i == len(faces) - 1 or len(data[0]) >= 5000:
            pks = insert_data(collection, data)
            # print(pks)
            data[0].clear()
            data[1].clear()
            for pk in pks:
                result.append(
                    {"primary_key": pk,
                     'isSuccess': True})
    if flush:
        collection.flush()
    return result


# 批量人脸搜索图片
def search_face_image(model: Face_Onnx, collection, imgs,
                      enhance=False, score=0.5, limit=10, search_params=None, nprobe=50):
    embeddings = []
    for img in imgs:
        embedding = model.turn2embeddings(img, enhance=enhance)
        if len(embedding) == 0:
            embeddings.append(np.zeros([512], dtype=np.float32))
            continue
        embeddings.append(embedding[0])

    if search_params is None:
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": nprobe},
        }
    limit = 16383 if limit > 16383 else limit
    search_res = collection.search(embeddings, 'embedding', search_params,
                                   limit=limit, output_fields=['object_id', 'hdfs_path', 'quality_score'],
                                   round_decimal=4)
    result = []
    for one in search_res:
        _result = []
        for single in one:
            # print(single)
            if single.score >= score:
                tmp = {
                    # 'primary_key': single.id,
                    'id': single.entity.id,
                    'object_id': single.entity.object_id,
                    'score': single.score,
                    'hdfs_path': single.entity.hdfs_path,
                    'quality_score': str(single.entity.quality_score)
                }
                # get_search_result(single.id, single.entity.user_id, single.score)
                _result.append(tmp)
        result.append(_result)

    return result

def search_main_face_image(model: Face_Onnx, collection, imgs,
                      enhance=False, score=0.5, limit=10, search_params=None, nprobe=50):
    embeddings = []
    for img in imgs:
        embedding = model.turn2embeddings(img, enhance=enhance)
        if len(embedding) == 0:
            embeddings.append(np.zeros([512], dtype=np.float32))
            continue
        embeddings.append(embedding[0])

    if search_params is None:
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": nprobe},
        }
    limit = 16383 if limit > 16383 else limit
    search_res = collection.search(embeddings, 'embedding', search_params,
                                   limit=limit, output_fields=['object_id', 'hdfs_path', 'quality_score', 'recognition_state'],
                                   round_decimal=4)
    result = []
    for one in search_res:
        _result = []
        for single in one:
            # print(single)
            if single.score >= score:
                tmp = {
                    # 'primary_key': single.id,
                    'id': single.entity.id,
                    'object_id': single.entity.object_id,
                    'score': single.score,
                    'hdfs_path': single.entity.hdfs_path,
                    'quality_score': str(single.entity.quality_score),
                    'recognition_state': single.entity.recognition_state
                }
                # get_search_result(single.id, single.entity.user_id, single.score)
                _result.append(tmp)
        result.append(_result)

    return result


def search_vectors(collection, field, vectors, output_fields,
                   search_params=None, limit=3, nprobe=50):
    if search_params is None:
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": nprobe},
        }
    limit = 16383 if limit > 16383 else limit
    res = collection.search(vectors, field, search_params,
                            limit=limit, output_fields=output_fields, round_decimal=4)
    return res


# 批获取人脸图片的质量分数
# 返回result 和 err, err记录读取出错的图片路径
# 批获取人脸图片的质量分数
# imgs为list，list中的每个img是经过人脸检测得到的人脸图片
# 返回result 和 err, err记录错误信息，暂时为空list
def get_face_quality_batch_img(model: Face_Onnx, imgs):
    result = []
    for img in imgs:
        score = model.tface.forward(img)
        result.append(score)
    return result


# 批获取人脸图片的质量分数
# 返回result 和 err, err记录读取出错的图片路径
# 批获取人脸图片的质量分数
# imgs为list，list中的每个img是经过人脸检测得到的人脸图片
# 返回result 和 err, err记录错误信息，暂时为空list
def get_face_quality_single_img(model: Face_Onnx, image_path):
    img = cv_imread(image_path)
    score = model.tface.forward(img)
    return score


