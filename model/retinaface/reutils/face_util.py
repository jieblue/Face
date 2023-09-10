import numpy as np
import cv2


def face_quality_assessment(face, points):
    quality_score = 0

    # 人脸大小评估
    face_size = face.shape[0] * face.shape[1]
    if face_size >= 1600:
        quality_score += 1

    # 返回的结果是一个浮点数，代表了输入图像的拉普拉斯变换的方差。根据方差的大小，
    # 可以判断图像的清晰度。方差越小，说明图像越模糊，而方差越大，说明图像越清晰。
    # blurred = cv2.Laplacian(face, cv2.CV_64F).var()
    # if blurred > 100:
    #     quality_score += 1

    # 根据关键点进行姿态估计
    # 获取目标人脸区域的关键点坐标
    # left_eye = [int(points[0]), int(points[5])]
    # right_eye = [int(points[1]), int(points[6])]
    # nose = [int(points[2]), int(points[7])]
    # mouth_left = [int(points[3]), int(points[8])]
    # mouth_right = [int(points[4]), int(points[9])]
    # # print(right_eye)
    # # 计算眼睛之间的角度
    # eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
    #
    # # 根据眼睛之间的角度评估姿态
    # if abs(eye_angle) < 10:
    #     quality_score += 1

    return quality_score