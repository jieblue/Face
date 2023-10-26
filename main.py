import os.path
import time

from model.model_onnx import *
from service import face_service
from utils.img_util import *
from config.config import *
from milvus_tool import local_milvus


# 获取config信息
conf = get_config()

# 加载模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)

# 获取Milvus的配置
milvus_conf = conf['milvus']

#创建和Milvus的连接
# db_name 表示连接的数据库名称，之前使用的是default, 考虑到拓展性，
# 使用initialize.py 重新初始化创建了Face_Search数据库
# con = local_milvus.create_connection(host=milvus_conf['host'], port=milvus_conf['port'],
#                         user=milvus_conf['user'], password=milvus_conf['password'], db_name='Face_Search')
#
#
# 获取Milvus中的collection，并且加载到内存中
# image_collection = local_milvus.get_collection("image_faces", load=True)
# video_collection = local_milvus.get_collection("video_faces", load=True)


img1_path ='./test_img/0.jpg'
img2_path ='./test_img/1.jpg'
img3_path ='./test_img/2.png'
img_paths = [img1_path, img2_path, img3_path]


# 提取人脸并且对齐人脸
start = time.time()
extracted_faces, err = face_service.get_align_faces_batch(face_model, img_paths,
                                                   enhance=False, confidence=0.99)
print('提取人脸耗时:' + str(time.time()-start))
print(err)
# 把结果写在./res_img目录下查看
res_path = './res_img'
res_paths = []
for i, single in enumerate(extracted_faces):
    for j, face in enumerate(single):
        name = str(i) +'_' +str(j)+'.jpg'
        path = os.path.join(res_path, name)
        res_paths.append(path)
        cv2.imwrite(path, face)

# 增强提取处来的人脸
start = time.time()
enhanceed_faces, err = face_service.enhance_face_batch(face_model, res_paths, aligned=True)
print('增强人脸耗时:' + str(time.time()-start))
print(err)
#把增强后的人脸写在 /res_img下
for i , face in enumerate(enhanceed_faces):
    name = res_paths[i].split('.jpg') [0] + 'enhanceed.jpg'
    cv2.imwrite(name, face)


res_paths.clear()
res_list = os.listdir('./res_img')
for single in res_list:
    path = os.path.join(res_path, single)
    res_paths.append(path)

# 获取人脸图片的质量分数
start = time.time()
scores, err = face_service.get_face_quality_batch(face_model, res_paths, aligned=True)
print('获取质量分数耗时:' + str(time.time()-start))
print('质量分数:' + str(scores))
print(err)

# 获取对齐后的人脸向量， 一张图片就是一个人脸图片，
start = time.time()
embeddings, err = face_service.get_face_embeddings(face_model, res_paths, aligned=True)
print('获取对齐后的人脸向量耗时:' + str(time.time()-start))
print(err)
#打印向量维度 512维
print(len(embeddings[0][0]))

# 获取图片中的人脸向量， 每个图片可以有多张人脸， 这个过程包括提取人脸图片
'''
get_face_embeddings(aligned=False) =
get_align_faces_batch() + get_face_embeddings(aligned=True)

'''
start = time.time()
embeddings, err = face_service.get_face_embeddings(face_model, img_paths, aligned=False)
print('获取图片中的人脸向量耗时:' + str(time.time()-start))
print(len(embeddings[0]))
print(len(embeddings[2]))
print(err)

'''
把特征人脸特征向量插入Milvus， 图片人脸就插入image_collection,
视频人脸就插入video_collection
id 是标识每个人脸的id， 可以是视频id， 图片id， 人员id
embedding 是一个人脸特征向量
'''
# faces = []
# for i, single in enumerate(embeddings):
#     for embedding in single:
#         faces.append({
#             'id': str(i),
#             'embedding': embedding
#         })
#
# res = face_service.add_embeddings2milvus(image_collection, faces)
# print(res)

# 把collection 从内存中释放， Milvus的查询需要把collection加载到内存中，程序结束时才要释放
# image_collection.release()
# video_collection.release()