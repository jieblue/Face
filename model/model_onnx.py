from model.tface.tface_onnx import TFace_Onnx
from utils.img_util import read_img
from model.adaface.adaface_onnx import Adaface_Onnx
from model.gfpgan_onnx.gfpgan_onnx import GFPGAN_Onnx
from model.retinaface.retinafce_onnx import Retinaface_Onnx


class Face_Onnx:
    def __init__(self, config, cuda=True):
        self.retinaface = Retinaface_Onnx(config['retinaface'], cuda=cuda)
        self.adaface = Adaface_Onnx(config['adaface'], cuda=cuda)
        self.gfpgan = GFPGAN_Onnx(config['gfpgan'], cuda=cuda)
        self.tface = TFace_Onnx(config['tface'], cuda=cuda)
        self.warw_up(config['img'])


    def warw_up(self, path):
        img = read_img(path, False)
        self.turn2embeddings(img, enhance=True)


    def extract_faces_enhance(self, img, confidence=0.99):
        enhance_faces = []
        faces = self.retinaface.extract_face(img, 512, confidence=confidence)
        for face in faces:
            enhance_face = self.gfpgan.forward(face, True)
            enhance_faces.append(enhance_face)
        return enhance_faces


    def extract_face(self, img, enhance=False, confidence=0.99):
        if not enhance:
            return self.retinaface.extract_face(img, 112, confidence)

        else:
            return self.extract_faces_enhance(img, confidence)

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


