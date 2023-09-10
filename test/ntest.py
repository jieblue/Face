import time
import cv2
import os
import numpy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

limit = 3
limit = 16383 if limit >16383 else limit
print(limit)
# data = [ *single for single in raw]
# print(data)

