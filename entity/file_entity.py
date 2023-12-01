class FileData:
    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path


class VideoFile(FileData):
    def __init__(self, *, file_name, file_path, video_id, tag):
        super().__init__(file_name, file_path)
        self.tag = tag
        self.video_id = video_id


class ImageFile(FileData):
    def __init__(self, file_name, file_path, image_id):
        super().__init__(file_name, file_path)
        self.image_id = image_id
        self.tag = None
