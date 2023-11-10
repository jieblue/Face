import tensorflow as tf
from PIL import Image
import numpy as np



class Video_Model:
    def __init__(self, config, gpu_id=0):
        with tf.device('/gpu:{}'.format(gpu_id) if gpu_id >= 0 else '/cpu:0'):
            self.model = tf.keras.applications.ResNet50(include_top=False,weights=None)
            self.model.load_weights(config)


    
    def get_frame_embedding(self, frame_path):
        print(frame_path)
        # if distant:
        #     content = requests.get(url, stream=True).content
        #     byteStream = io.BytesIO(content)
        #     image = Image.open(byteStream)
        # else:
        #     image = Image.open(url)
        image=Image.open(frame_path)
        image = image.resize([224, 224]).convert('RGB')
        y = tf.keras.preprocessing.image.img_to_array(image)
        y = np.expand_dims(y, axis=0)
        # print(y)
        y = tf.keras.applications.resnet50.preprocess_input(y)
        # print(type(y), y)
        y = self.model.predict(y)
        result = tf.keras.layers.GlobalAveragePooling2D()(y)
        feature = [x for x in result.numpy()[0].tolist()]

        return feature


