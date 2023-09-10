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
    faces = np.array(faces_list)
    _len = len(faces_list)

    #numpy to tensor
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
    # 从有范数的向量列表中提取没有范数的向量列表
    return numpy_list


