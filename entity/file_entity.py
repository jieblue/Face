class FileData:
    def __init__(self, unique_name, file_path):
        self.file_name = unique_name
        self.file_path = file_path


class VideoFile(FileData):
    def __init__(self, unique_name, file_path, video_id):
        super().__init__(unique_name, file_path)
        self.video_id = video_id


class ImageFile(FileData):
    def __init__(self, unique_name, file_path, image_id):
        super().__init__(unique_name, file_path)
        self.image_id = image_id
