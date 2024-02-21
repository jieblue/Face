from PIL import Image
import torch
import torchvision.transforms as transforms
import time
import onnxruntime as ort

from utils import log_util

# Create a logger
logger = log_util.get_logger(__name__)


class VideoModel:
    def __init__(self, config, gpu_id=0):
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': gpu_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]

        so = ort.SessionOptions()
        so.log_severity_level = 3
        # 加载onnx模型到onnxruntime的推理
        self.session = ort.InferenceSession(config, so, providers=providers)
        # self.session = ort.InferenceSession(config, so)
        self.input_name = self.session.get_inputs()[0].name
        self.preprocess = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # with tf.device('/gpu:{}'.format(gpu_id) if gpu_id >= 0 else '/cpu:0'):
        #     self.model = tf.keras.applications.ResNet50(include_top=False, weights=None)
        #     self.model.load_weights(config)

    def get_frame_embedding_byte(self, frame_image):

        # 提取每个帧的特征向量
        start_time = time.time()
        # 打开图像并进行预处理
        # image = Image.open(frame_path).convert("RGB")
        image = self.preprocess(frame_image).unsqueeze(0).numpy()
        # 使用模型提取特征
        # with torch.no_grad():
        # 输入模型进行推理
        feature = self.session.run(None, {self.input_name: image})
        feature = torch.from_numpy(feature[0])
        feature = torch.nn.functional.adaptive_avg_pool2d(feature, [1, 1])
        feature = feature.reshape((1, 4, 512))
        feature = torch.mean(feature, dim=1)
        feature = feature.flatten().numpy()
        feature = feature.tolist()

        end_time = time.time()
        logger.info("Extract feature time: {}".format(end_time - start_time))
        return feature

    def get_frame_embedding_byte_256(self, frame_image):

        # 提取每个帧的特征向量
        start_time = time.time()
        # 打开图像并进行预处理
        # image = Image.open(frame_path).convert("RGB")
        image = self.preprocess(frame_image).unsqueeze(0).numpy()
        # 使用模型提取特征
        # with torch.no_grad():
        # 输入模型进行推理
        feature = self.session.run(None, {self.input_name: image})
        feature = torch.from_numpy(feature[0])
        feature = torch.nn.functional.adaptive_avg_pool2d(feature, [1, 1])
        feature = feature.reshape((1, 8, 256))
        feature = torch.mean(feature, dim=1)
        feature = feature.flatten().numpy()
        feature = feature.tolist()

        end_time = time.time()
        logger.info("Extract feature time: {}".format(end_time - start_time))
        return feature

    def get_frame_embedding(self, frame_image):
        # 提取每个帧的特征向量
        # 打开图像并进行预处理
        if isinstance(frame_image, str):
            image = Image.open(frame_image).convert("RGB")
            return self.get_frame_embedding_byte(image)
        else:
            return self.get_frame_embedding_byte(frame_image.to_image())

    def get_frame_embedding_256(self, frame_image):
        # 提取每个帧的特征向量
        # 打开图像并进行预处理
        if isinstance(frame_image, str):
            image = Image.open(frame_image).convert("RGB")
            return self.get_frame_embedding_byte_256(image)
        else:
            return self.get_frame_embedding_byte_256(frame_image.to_image())
