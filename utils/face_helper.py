import cv2
import numpy as np
import torch

face_template_512 = np.array(
    [[192.98138, 239.94708],
     [318.90277, 240.1936],
     [256.63416, 314.01935],
     [201.26117, 371.41043],
     [313.08905, 371.15118]])


def align_face(img, landmark, face_size, border_mode = 'constant',save_path=None, reorder=True):
    if reorder:
        landmark = np.float32([[landmark[i], landmark[i + 5]] for i in range(5)])


    face_template = face_template_512 * (face_size / 512.0)
    # print(face_template)
    # print(landmark)
    affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
    # warp and crop faces
    if border_mode == 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode == 'reflect101':
        border_mode = cv2.BORDER_REFLECT101
    elif border_mode == 'reflect':
        border_mode = cv2.BORDER_REFLECT

    face = cv2.warpAffine(
        img, affine_matrix, (face_size, face_size), borderMode=border_mode, borderValue=(135, 133, 132))  # gray

    # save the cropped face
    if save_path is not None:
        cv2.imwrite(save_path, face)

    return face


def squeeze_faces(faces_list, threshold=0.48):
    _len = len(faces_list)
    unique_vectors = []
    # ids = []
    for i, vector in enumerate(faces_list):

        # 检查是否与之前的向量重复
        is_duplicate = False
        for x in unique_vectors:
            # print(x_tensor.size())
            # 计算余弦相似度
            if cul_similarity(vector, x) > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_vectors.append(vector)
            # ids.append(i)

    return unique_vectors


def grouping_face(faces_list, threshold=0.55):
# input is a list of face, every face is a dict {id:....., embedding: .....}
# return is a list, containing some list, [[face1, face2, ], [....]....]
    res = []
    for face in faces_list:
        is_duplicate = False
        for i, single in enumerate(res):
            if cul_similarity(single[0]['embedding'],
                              face['embedding']) >threshold:
                res[i].append(face)
                is_duplicate = True
                break

        if not is_duplicate:
            res.append([face])

    return res


# 计算余弦相似度
def cul_similarity(face_x, face_y, noraml=True):
    # list 2 numpy
    # normal 表示输入向量是否归一化
    np_fx = np.array(face_x)
    np_fy = np.array(face_y)
    if noraml:
        return np.dot(np_fx, np_fy)
    else:
        return np.dot(np_fx, np_fy) / (np.linalg.norm(np_fx) * np.linalg.norm(np_fy))

    # retur
    #n torch.nn.functional.cosine_similarity(t_fx, t_fy)

#计算图像梯度
def calculate_sharpness_score(image):
    # 读取图像
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的梯度
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 上限400
    score  = min(score, 400)
    return score