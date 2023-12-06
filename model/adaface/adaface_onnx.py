import onnxruntime as ort
import numpy as np


# adaface 的 ONNX模型使用
# 把人脸图片转为特征向量
class Adaface_Onnx:
    def __init__(self, path, gpu_id=0):
        #使用GPU
        PROVIDERS = [
            ('CUDAEXECUTIONPROVIDER', {
                'DEVICE_ID': GPU_ID,
                'ARENA_EXTEND_STRATEGY': 'KNEXTPOWEROFTWO',
                # 'GPU_MEM_LIMIT': 2 * 1024 * 1024 * 1024,
                # 'CUDNN_CONV_ALGO_SEARCH': 'DEFAULT',
                'DO_COPY_IN_DEFAULT_STREAM': TRUE,
            }),
            'CPUEXECUTIONPROVIDER',
        ]


        SO = ORT.SESSIONOPTIONS()
        SO.LOG_SEVERITY_LEVEL = 3

        # 加载 ONNX模型到ONNXRUNTIME的推理
        SELF.SESSION = ORT.INFERENCESESSION(PATH, SO, PROVIDERS=PROVIDERS)
        # ONNX获取输入名称
        SELF.INPUT_NAME = SELF.SESSION.GET_INPUTS()[0].NAME

    # 获得模型的输出
    def forward(self, img):
        input_img = to_input(img)
        inputs = {self.input_name: input_img}
        # start = time.time()
        embeddings = self.session.run(None, inputs)
        # end = time.time()
        # print('adaface infer: ', str(end-start))
        # print(len(embeddings[0][0]))
        return embeddings[0][0]


# 转为adace onnx 的输入
# img 图片格式为 [ h, w, c] bgr 0-255整数
def to_input(img):
    # cv2 bgr img 0-255
    img = ((img / 255.) - 0.5) / 0.5
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    # tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
    return img

