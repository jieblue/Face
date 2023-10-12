import numpy
import onnxruntime as ort

import cv2

import numpy as np


# from utils.face_helper import pre_process, post_process

# TFace的ONNX模型， 用于得出质量分数
class TFace_Onnx:
    def __init__(self, path, cuda=True):
        #使用GPU
        if cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device = 'cuda'
        else:
            providers = ['CPUExecutionProvider']
            self.device = 'cpu'

        so = ort.SessionOptions()
        so.log_severity_level = 3
        # 加载onnx模型到onnxruntime的推理
        self.session = ort.InferenceSession(path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        #模型预热
        self.warmup()

    # 得到模型输出
    def forward(self, img):
        img = to_input(img)
        inputs = {self.input_name: img}
        res = self.session.run(None, inputs)[0][0][0]
        return res


    # 模型预热
    def warmup(self):
        x = numpy.random.random([112, 112, 3])
        self.forward(x)


# 转为模型输入要求
def to_input(img):
    # img is a bgr numpy array range(0, 255)
    img = img / 255.0
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 0] = (img[:, :, 0] - 0.5) / 0.5
    img[:, :, 1] = (img[:, :, 1] - 0.5) / 0.5
    img[:, :, 2] = (img[:, :, 2] - 0.5) / 0.5
    img = np.float32(img[np.newaxis, :, :, :])
    img = img.transpose(0, 3, 1, 2)
    return img
