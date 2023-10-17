import onnxruntime as ort
import cv2
import numpy as np


# gfpgan 的 ONNX模型使用
# 人脸增强
class GFPGAN_Onnx:
    def __init__(self, path, gpu_id=0):
        #使用GPU
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': gpu_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                # 'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]

        so = ort.SessionOptions()
        so.log_severity_level = 3
        # # 加载 onnx模型到onnxruntime的推理
        self.session = ort.InferenceSession(path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    # 获得模型的输出
    # resize=True会把增强后的图像转为112x112分辨率，用于adaface 的 tface的输入
    def forward(self, img, resize=False):
        img = pre_process(img)
        inputs = {self.input_name: img}
        res = self.session.run(None, inputs)[0][0]
        res = post_process(res)
        if resize:
            res = cv2.resize(res, (112, 112))
        return res


# 输入图像的前置处理
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

#输出结果的后置处理
def post_process(output):
    output = output.clip(-1,1)
    output = (output + 1) / 2
    output = output.transpose(1, 2, 0)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output = (output * 255.0).round()
    output = output.astype(np.uint8)

    return output