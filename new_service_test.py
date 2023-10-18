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

#
#
imgs = []
img1_path ='./test_img/0.jpg'
img2_path ='./test_img/1.jpg'
img3_path ='./test_img/2.png'
img_paths = [img1_path, img2_path, img3_path]

for img_path in img_paths:
    img = cv_imread(img_path)
    imgs.append(img)

# 提取人脸并且对齐人脸
start = time.time()
extracted_faces, err = face_service.get_align_faces_batch_img(face_model, imgs,
                                                   enhance=False, confidence=0.99)
print('提取人脸耗时:' + str(time.time()-start))
print(err)

total_faces = []
for faces in extracted_faces:
    total_faces += faces
# 把结果写在./res_img目录下查看
res_path = './res_img'

for i, face in enumerate(total_faces):
    name = str(i) +'.jpg'
    path = os.path.join(res_path, name)
    cv2.imwrite(path, face)

# 增强提取出来的人脸
start = time.time()
enhanceed_faces, err = face_service.enhance_face_batch_img(face_model, total_faces)
print('增强人脸耗时:' + str(time.time()-start))
print(err)
#把增强后的人脸写在 /res_img下
for i , face in enumerate(enhanceed_faces):
    name = res_path +'/' +str(i) + 'enhanced.jpg'
    cv2.imwrite(name, face)


# 获取人脸图片的质量分数
start = time.time()
scores, err = face_service.get_face_quality_batch_img(face_model, total_faces)
print('获取质量分数耗时:' + str(time.time()-start))
print('质量分数:' + str(scores))
print(err)

# 获取对齐后的人脸向量， 输入是人脸检测得到的图片
start = time.time()
embeddings, err = face_service.get_face_embeddings_img(face_model, total_faces, aligned=True)
print('获取对齐后的人脸向量耗时:' + str(time.time()-start))
print(err)
#打印向量维度 512维
print(len(embeddings))
print(len(embeddings[0]))
print(len(embeddings[0][0]))

# 获取图片中的人脸向量， 每个图片可以有多张人脸， 这个过程包括提取人脸图片
'''
get_face_embeddings_img(aligned=False) =
get_align_faces_batch_img() + get_face_embeddings_img(aligned=True)

'''
start = time.time()
embeddings, err = face_service.get_face_embeddings_img(face_model, imgs, aligned=False)
print('获取图片中的人脸向量耗时:' + str(time.time()-start))
print(len(embeddings[0]))
print(len(embeddings[2]))
print(err)


# 关键帧路径
keyframes_dir0 = './keyframes/0'
keyframes_dir1 = './keyframes/1'
keyframes_img0 = []
keyframes_img1 = []
keyframes_faces_dir = './keyframes_faces'

for name in os.listdir(keyframes_dir0):
    path = os.path.join(keyframes_dir0, name)
    img = cv_imread(path)
    keyframes_img0.append(img)

for name in os.listdir(keyframes_dir1):
    path = os.path.join(keyframes_dir1, name)
    img = cv_imread(path)
    keyframes_img1.append(img)

# 关键帧图片list
keyframes_imgs = [keyframes_img0, keyframes_img1]
# 获取关键帧的人脸
faces, err = face_service.get_keyframes_faces_img(face_model, keyframes_imgs, confidence=0.9)
for i, single in enumerate(faces):
    for j, face in enumerate(single):

        name = '/' + str(i) + '/' + str(j) + '.jpg'
        path = keyframes_faces_dir + name
        cv2.imwrite(path, face)

# 获取关键帧的人脸向量
embeddings, err = face_service.get_keyframes_faces_imgembedding_img(face_model, faces)
print(len(embeddings))
print(len(embeddings[0]))
print(len(embeddings[1]))