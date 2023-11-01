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
#default
con = local_milvus.create_connection(host=milvus_conf['host'], port=milvus_conf['port'],
                        user=milvus_conf['user'], password=milvus_conf['password'], db_name='Face_Search')


# 获取Milvus中的collection，并且加载到内存中
image_collection = local_milvus.get_collection("image_faces", load=True)
video_collection = local_milvus.get_collection("video_faces", load=True)


img1_path ='D:/dataset/face/szfdata/SZF/W020230406421816120026.jpg'
img2_path ='./test_img/2.png'

img1 = cv_imread(img1_path)
img2 = cv_imread(img2_path)

# 批量检索人脸图片， 每张人脸图片只能有一张人脸
imgs = [img1, img2]
start = time.time()
res = face_service.search_face_image(face_model, image_collection, imgs,
                                            enhance=False, score=0.4, limit=2, nprobe=50, offset=0)
print('搜索耗时: ' + str(time.time()-start))
print("搜索结果: ")
print(res)

# object_ids = ['3214412', '1', '2', 'dadadada', 'dad32fdsfds']
# start = time.time()
# res = face_service.delete_face_by_object_id(image_collection, object_ids)
# print('删除耗时: ' + str(time.time()-start))
# print(res)
# #
# #
#
# # 根据priimary_key 批量删除
# base = 444134326410189121
# primary_keys = []
#
# for i in range(10000):
#     primary_keys.append(++base)
# res = face_service.delete_face_by_primary_key_batch(image_collection, primary_keys)
#
# image_collection.release()
# video_collection.release()



