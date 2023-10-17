from model.tface.tface_onnx import TFace_Onnx
from utils.img_util import read_img
from model.adaface.adaface_onnx import Adaface_Onnx
from model.gfpgan_onnx.gfpgan_onnx import GFPGAN_Onnx
from model.retinaface.retinafce_onnx import Retinaface_Onnx


# Face_Onnx 是.model包下四个模型的整合
class Face_Onnx:
    def __init__(self, config, gpu_id=0):
        # 模型初始化 config为配置文件， gpu_id是指定gpu
        self.retinaface = Retinaface_Onnx(config['retinaface'], gpu_id=gpu_id)
        self.adaface = Adaface_Onnx(config['adaface'], gpu_id=gpu_id)
        self.gfpgan = GFPGAN_Onnx(config['gfpgan'], gpu_id=gpu_id)
        self.tface = TFace_Onnx(config['tface'], gpu_id=gpu_id)
        # 预热模型
        self.warw_up(config['img'])

    # 预热
    def warw_up(self, path):
        img = read_img(path, False)
        self.turn2embeddings(img, enhance=True)


    # 提取图片中的人脸同时增强
    def extract_faces_enhance(self, img, confidence=0.99):
        enhance_faces = []
        faces = self.retinaface.extract_face(img, 512, confidence=confidence)
        for face in faces:
            enhance_face = self.gfpgan.forward(face, True)
            enhance_faces.append(enhance_face)
        return enhance_faces

    # 提取图片中的人脸， enhance表示是否增强人脸, confidence是人脸置信度
    def extract_face(self, img, enhance=False, confidence=0.99):
        if not enhance:
            return self.retinaface.extract_face(img, 112, confidence)

        else:
            return self.extract_faces_enhance(img, confidence)

    # 把人脸图片转为特征向量
    # enhacne 表示是否对人脸增强，
    # aligned表示人脸图片是否对齐，True表示是经过extract_face得到的人脸图片，
    # False表示输入的是原始图片同extract_face的输入
    def turn2embeddings(self, img, enhance=False, aligned=False,
                        confidence=0.99):
        face_size = 512 if enhance else 112
        if aligned:
            faces = [img]
        else:
            faces = self.retinaface.extract_face(img, face_size,
                                             confidence=confidence)

        if enhance:
            for i, face in enumerate(faces):
                enhance_face = self.gfpgan.forward(face, resize=True)
                faces[i] = enhance_face

        embeddings = []
        for i, face in enumerate(faces):

            embedding = self.adaface.forward(face)
            embeddings.append(embedding)

        return embeddings


