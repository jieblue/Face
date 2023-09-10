import uuid
import urllib.request
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import io

def np2pil(img):
    ## numpy aray bgr --> PIL.Image
    return Image.fromarray(np.uint8(img[:, :, ::-1]))


def pil2np(img):
    return np.array(img)[:, :, ::-1]


def get_uuid():
    return uuid.uuid4().int & (1<<64)-1



def np2tensor(img, normal=False):
    image_rgb = img[:, :, ::-1]
    tensor_rgb = torch.from_numpy(image_rgb.copy().transpose((2, 0, 1))).float()
    # print(tensor_rgb)
    if normal:
        tensor_rgb = tensor_rgb/255.
    return tensor_rgb


def tensor2np(img):
    return img.cpu().numpy()[:, :, ::-1]


def read_img(path, is_url=False):
    #return numpy bgr
    # print(path)
    if is_url:
        with urllib.request.urlopen(path) as url_response:
            img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            # print(img.shape)
    else:
        try:
            img = cv_imread(path)
        except:
            return None
        # print(img)

    return img



def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)
    return cv_img

def down_image(img):

    height, width = img.shape[:2]
    # print(height, width)
    if width > height:
        scale = 1920 / width
    else:
        scale = 1920 / height

    if scale >=1:
        return img

    new_hight = int(height * scale)
    new_width = int(width * scale)

    img = cv2.resize(img, (new_width, new_hight) )
    return img


def imgfile2np(file):
    img = Image.open(io.BytesIO(file))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return pil2np(img)