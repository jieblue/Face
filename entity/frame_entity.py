import numpy as np


class KeyFrame:
    def __init__(self, frame, timestamp, frame_num, unique_filename):
        self.frame = frame
        self.timestamp = timestamp
        self.frame_num = frame_num
        self.unique_filename = unique_filename
