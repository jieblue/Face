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

# 加载人脸模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)

# 加载视频模型
# video_model = Video_Model('./config/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', gpu_id=0)


# 获取Milvus的配置
milvus_conf = conf['milvus']
print("Milvus 配置信息： " + str(conf['milvus']))
print(f"start connecting to Milvus")

connections.connect("default", host="192.168.104.9", port="19530")

image_faces_v1_name = "image_faces_v1"

has = utility.has_collection(image_faces_v1_name)
print(f"Does collection {image_faces_v1_name} exist in Milvus: {has}")


print("Milvus 链接成功")

fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
            FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="hdfs_path", dtype=DataType.VARCHAR, max_length=256),

        ]
schema = CollectionSchema(fields, "image_faces_v1 is the simplest demo to introduce the APIs")
image_faces_v1 = Collection("image_faces_v1", schema)
# 主头像二级人像库
main_avator_v1 = Collection("main_avator_v1", schema)

index = {
    "index_type": 'IVF_SQ8',
    "metric_type": "IP",
    "params": {"nlist": 100},
}

image_faces_v1.create_index("embedding", index)
main_avator_v1.create_index("embedding", index)
print("Milvus embedding index  创建成功")

image_faces_v1.load()
main_avator_v1.load()

print("Milvus collection 加载成功")

key_frames_path = './keyframes'
key_faces_path = './keyframes_faces'
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
        # 必须是本地磁盘路径
        video_path = json_data["videoPath"]
        video_id = json_data["videoId"]
        if not os.path.exists(video_path):
            result["code"] = -1
            result["msg"] = "视频路径不存在"
            return jsonify(result)
        # 获取文件名唯一标识
        unique_filename = video_id
        print(f"Processing video file: {unique_filename}")


        # Extract key frames from video
        key_frames_info_list = core_service.extract_video(video_path, unique_filename)
        print(f"Extracted {len(key_frames_info_list)} key frames from video in {time.time() - start_time:.2f} seconds")

        
        # Write key frames to disk
        core_service.save_frame_to_disk(key_frames_info_list, key_frames_path, unique_filename)
        print(f"Wrote {len(key_frames_info_list)} key frames to disk in {time.time() - start_time:.2f} seconds")

        
        # Detect faces in key frames
        core_service.get_align_faces_batch(face_model, key_frames_info_list, True, 0.99)
        face_num = 0
        for key_frame_info in key_frames_info_list:
            face_num += len(key_frame_info['align_face'])
        print(f"Detected {str(face_num)} faces in key frames in {time.time() - start_time:.2f} seconds")

        # Write faces to disk
        faces_paths = core_service.save_face_to_disk(key_frames_info_list, key_faces_path, unique_filename)
        print(f"Wrote {face_num} faces to disk in {time.time() - start_time:.2f} seconds")

        print("filepahts:" + str(faces_paths))
        # Extract facial embeddings from faces
        core_service.get_face_embeddings(face_model, faces_paths, aligned=False, enhance=False, confidence=0.99, merge=False)

        # 0 face_embedding_id
        # 1 object_id
        # 2 embedding
        # 3 hdfs_path
        hdfs_prefix = "/VIDEO_FACE_TEST/face/"
        entities = [[],[], [], []]

        face_result_list = []
        for i, face_info in enumerate(faces_paths):
            if (len(face_info['embedding'])) > 0:
                entities[0].append(face_info['face_embedding_id'])
                entities[1].append('unidentification')
                entities[2].append(core_service.squeeze_faces(face_info['embedding'])[0])
                hdfs_file = hdfs_prefix + face_info['unique_filename'] + "/" + face_info['face_embedding_id'] + ".jpg"
                face_info['hdfs_path'] = hdfs_file
                entities[3].append(hdfs_file)
                face_result_list.append(face_info)
                
        # Insert embeddings into Milvus
        res = image_faces_v1.insert(entities)
        print(res)
        print(f"Inserted vectors into Milvus in {time.time() - start_time:.2f} seconds")

        result["msg"] = f"Processed video file {unique_filename} in {time.time() - start_time:.2f} seconds"
        # 删除一些不必要的元素
        for i, face_info in enumerate(face_result_list):
            del face_info['frame']
            del face_info['embedding']
            del face_info['single_align_face']

        print(faces_paths[0])

        # result['milvs_msg'] = res
        result['face_info_list'] = face_result_list
    except Exception as e:
        traceback.print_exc()
        # handle the exception
        result["code"] = -100
        result['unique_filename'] = unique_filename
    
    return result


face_predict_dir = '/tmp/face_predict_tmp'
generator = UniqueGenerator()

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

    offset = (int(page_num) - 1) * int(page_size)
    search_params = {
        "metric_type": "IP", 
        "offset": offset, 
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
        res = core_service.search_face_image(face_model, image_faces_v1, imgs,
                                                    enhance=False, score=float(score), limit=int(page_size), 
                                                    search_params=search_params)
        print('搜索耗时: ' + str(time.time()-start))
        print("搜索结果: ")
        print(res)
        result['res'] = res
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"
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
        print("uuid_filename")
        print(uuid_filename)

        dir_path = face_predict_dir + uuid_filename + ".jpg"
        file.save(dir_path)  # Replace with the path where you want to save the file

        img = cv_imread(dir_path)

        embedding = face_model.turn2embeddings(img, enhance=False)
        if len(embedding)==0:
            result["face_found_in_image"] = False
            result['error_messag'] = 'No face found'
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"

    
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
        res = core_service.search_face_image(face_model, main_avator_v1, imgs,
                                                    enhance=False, score=float(score), limit=10, 
                                                    search_params=search_params)
        print('主头像: ' + str(res))
        if len(res[0]) > 0:
            result["code"] = -1
            result["msg"] = "主头像已存在"
            return jsonify(result)
        
        embedding = face_model.turn2embeddings(img1, enhance=False)
        if len(embedding)==0:
            result["code"] = -1
            result["msg"] = "No face found"
            return jsonify(result)
        
        entities = [[],[], [], []]
        entities[0].append(uuid_filename)
        entities[1].append(object_id)
        entities[2].append(embedding[0])
        entities[3].append(hdfs_path)
        main_avator_res = main_avator_v1.insert(entities)
        
        print('插入结果耗时: ' + str(time.time()-start))
        print("搜索结果: ")
        print(main_avator_res)
        
        result['res'] = "插入成功"
    else:
        result["code"] = -1
        result["msg"] = "File uploaded Failure!"
    return jsonify(result)






if __name__ == '__main__':
    app.run(host='192.168.100.19', port=5010)


