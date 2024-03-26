import uuid
import urllib.request
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import io


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
    try:
        cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),cv2.IMREAD_COLOR)
    except:
        cv_img = None
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

#判断图片是否112x112分辨率
def judge_img112(img):
    # input image shape (H, W, C)
    # if H=W=112 return True else False
    height, width = img.shape[:2]
    if height==width and height==112:
        return True
    return False

# 写入压缩图片
def write_compression_img(save_path, img, bit_rate=50, size=None):
    # save_path 存储路径  img输入图片
    # bit_rate压缩比特率 默认50，降低约一半空间占用
    # size 默认为None，表示是否修改图片的分辨率，如果修改分辨率, eg. size=(1920, 1080)
    if size is not None:
        img = cv2.resize(img, size)
    # 设置压缩参数（比特率）
    compression_params = [cv2.IMWRITE_JPEG_QUALITY, bit_rate]
    # 保存压缩后的图像
    cv2.imwrite(save_path, img, compression_params)