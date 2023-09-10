import os

import yaml


def get_config():
    abs_path = os.path.abspath(__file__)
    parent_dir_path = os.path.dirname(abs_path)
    config_path = os.path.join(parent_dir_path, 'config.yml')


    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    cfg['model'] = get_model_path()

    return cfg


def get_model_path():
    abs_path = os.path.abspath(__file__)
    parent_dir_path = os.path.dirname(abs_path)
    img_path = os.path.join(parent_dir_path, 'warm_up.jpg')
    son_dir_path = os.path.join(parent_dir_path, 'weights')
    retinaface_path = os.path.join(son_dir_path, 'mobileface.onnx')
    adaface_path = os.path.join(son_dir_path, 'adaface.onnx')
    gfpgan_path = os.path.join(son_dir_path, 'gfpgan.onnx')
    tface_path = os.path.join(son_dir_path, 'tface.onnx')
    model_config = {
        'retinaface': retinaface_path,
        'adaface': adaface_path,
        'gfpgan': gfpgan_path,
        'tface': tface_path,
        'img': img_path
    }

    return model_config