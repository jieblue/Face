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


def cul_similarity(face_x, face_y):
    # list 2 numpy
    np_fx = np.array(face_x)
    np_fy = np.array(face_y)
    return np.dot(np_fx, np_fy)
    # #numpy to tensor
    # t_fx = torch.from_numpy(np_fx).float().unsqueeze(0)
    # t_fy = torch.from_numpy(np_fy).float().unsqueeze(0)
    # return torch.nn.functional.cosine_similarity(t_fx, t_fy)