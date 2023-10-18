import os.path
import time

from model.model_onnx import *
from service import face_service
from utils.img_util import *
from config.config import *

# 获取config信息
conf = get_config()

# 加载模型 加载模型会耗时比较长
face_model = Face_Onnx(conf['model'], gpu_id=0)

# 测试视频路径
video1_path = 'C:/Users/jieblue/Documents/stuff/nls1/横向/海博视频检索资料/科研团队资料/样例视频/0.mp4'
video2_path = 'C:/Users/jieblue/Documents/stuff/nls1/横向/海博视频检索资料/科研团队资料/样例视频/1.mp4'
res_path = './keyframes'

# 视频路径数组
video_paths = [video1_path, video2_path]

# 提取视频关键帧
# 17s 视频耗时 0.2217ms
# 32分09秒 视频耗时 22s
start = time.time()
keyframes, err = face_service.extract_key_frames_batch(video_paths)
print("提取视频关键帧耗时: " + str(time.time() - start))
print(err)
# 把提取出的视频关键帧写入本地
for i, single in enumerate(keyframes):
    for j, keyframe in enumerate(single):
        frame = keyframe['frame']
        timestamp = keyframe['timestamp']
        dir = os.path.join(res_path, str(i))
        os.makedirs(dir, exist_ok=True)
        name = str(j) + '_' + str(timestamp) +'.jpg'
        path = os.path.join(dir, name)
        cv2.imwrite(path, frame)

list_path = os.listdir('./keyframes')
keyframes_dirs = []
for single in list_path:
    dir = os.path.join('./keyframes', single)

    keyframes_dirs.append(dir)

# 提取关键帧中的人脸
faces, err = face_service.get_videos_faces(face_model, keyframes_dirs)
print(err)

for i, single in enumerate(faces):
    for j, face in enumerate(single):
        dir = os.path.join('./keyframes_faces', str(i))
        os.makedirs(dir, exist_ok=True)
        name = str(j) + '.jpg'
        path = os.path.join(dir, name)
        cv2.imwrite(path, face)

faces_path1 = './keyframes_faces/0'
faces_path2 = './keyframes_faces/1'
faces_paths = [faces_path1, faces_path2]

# 获取关键帧提取出的人脸的特征向量 输入是目录数组，每个目录存取自己视频提取出的人脸
embeddings, err = face_service.get_video_extracted_face_embedding(face_model, faces_paths, 0.5)
print("第1个视频的特征向量个数："+ str(len(embeddings[0])))
print("第2个视频的特征向量个数："+ str(len(embeddings[1])))
print(err)

# 直接获取关键帧中的人脸特征向量 输入是目录数组，每个目录存取自己视频的关键帧
'''
get_videos_face_embedding() = 
get_videos_faces() + get_video_extracted_face_embedding()
'''
embeddings, err = face_service.get_videos_face_embedding(face_model, keyframes_dirs, threshold=0.5)
print("第1个视频的特征向量个数："+ str(len(embeddings[0])))
print("第2个视频的特征向量个数："+ str(len(embeddings[1])))
print(err)


