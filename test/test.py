import glob
import os
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
from config.config import get_config
from service.face_service import *




config = get_config()
model = Face_Onnx(config['model'])
milvus_conf = config['milvus']



# con = create_connection(host=milvus_conf['host'], port=milvus_conf['port'],
#                         user=milvus_conf['user'], password=milvus_conf['password'])
#
# #
# image_collection = get_collection("image_faces")
# video_collection = get_collection("video_faces")
#
# image_collection.release()
# video_collection.release()
#
# img1 = cv_imread('C:/codeproject/face/Face/img/tyk.png')
# img_path = 'C:/codeproject/face/Face/img/tyk.png'
# embeddings = get_face_embeddings(model, [img_path])
# face = []
# for single in embeddings:
#     for embedding in single:
#         face.append({
#             'id': '3214412',
#             'embedding': embedding
#         })
# start = time.time()
# res = add_embeddings2milvus(image_collection ,face, False)
# print(time.time()-start)
# start = time.time()
# res = search_image_by_face(model, image_collection, img)
# print(time.time()-start)
# print(res)
# image_dir = 'D:/dataset/face/szfdata/SZF'
#
# video_dir = 'D:/dataset/face/szfdata/video'
# # video_dir = 'C:/Users/jieblue/Documents/stuff/nls1/横向/海博视频检索资料/科研团队资料/样例视频'
# res_dir = 'D:/dataset/face/szfdata/keyframes'
# # # list_path = glob.glob(os.path.join(image_dir, "*.jpg"))
# frame_dir = 'D:/dataset/face/szfdata/keyframes'
# image_list = []
# video_list = []
# frame_list = []
# name_list = []
# for single in os.listdir(image_dir):
#     # print(sigle)
#     name = (single.split('\\'))[-1].split('.')[0]
#     # img_type = (single.split('\\'))[-1].split('.')[1]
#     # # print(name)
#     name_list.append(name)
#     # print(sigle)
#     _path = os.path.join(image_dir, single)
#     # print(image_path)
#     image_list.append(_path)
#
#
# embeddings = get_face_embeddings(model, image_list)
#
# data = []
# for i, single in enumerate(embeddings):
#     if len(single) == 0:
#         continue
#
#     for embedding in single:
#         data.append({
#             'id': name_list[i],
#             'embedding': embedding
#         })
#
# add_embeddings2milvus(image_collection, data)
