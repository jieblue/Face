import torch.onnx
import onnxruntime as ort
import time
import uuid

import cv2
import os
import numpy as np
import onnx

class Adaface_Onnx:
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


    def forward(self, img):
        input_img = to_input(img)
        inputs = {self.input_name: input_img}
        # start = time.time()
        embeddings = self.session.run(None, inputs)
        # end = time.time()
        # print('adaface infer: ', str(end-start))
        # print(len(embeddings[0][0]))
        return embeddings[0][0]



def to_input(img):
    # cv2 bgr img 0-255
    img = ((img / 255.) - 0.5) / 0.5
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    # tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
    return img

