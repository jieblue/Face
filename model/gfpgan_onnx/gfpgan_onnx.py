import torch.onnx
import onnxruntime as ort
import time
import uuid
import cv2
import os
import numpy as np
import onnx


class GFPGAN_Onnx:
    def __init__(self, path, cuda=True):
        if cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.device = 'cuda'
        else:
            providers = ['CPUExecutionProvider']
            self.device = 'cpu'

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name


    def forward(self, img, resize=False):
        img = pre_process(img)
        inputs = {self.input_name: img}
        res = self.session.run(None, inputs)[0][0]
        res = post_process(res)
        if resize:
            res = cv2.resize(res, (112, 112))
        return res


def pre_process(img):
    # img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    # img = cv2.resize(img, (512, self.face_size))
    img = img / 255.0
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[:, :, 0] = (img[:, :, 0] - 0.5) / 0.5
    img[:, :, 1] = (img[:, :, 1] - 0.5) / 0.5
    img[:, :, 2] = (img[:, :, 2] - 0.5) / 0.5
    img = np.float32(img[np.newaxis, :, :, :])
    img = img.transpose(0, 3, 1, 2)
    return img

def post_process(output):
    output = output.clip(-1,1)
    output = (output + 1) / 2
    output = output.transpose(1, 2, 0)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output = (output * 255.0).round()
    output = output.astype(np.uint8)

    return output