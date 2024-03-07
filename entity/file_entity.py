class FileData:
    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path
        self.library_type = None


class VideoFile(FileData):
    def __init__(self, *, file_name, file_path, video_id, tag, saas_flag):
        super().__init__(file_name, file_path)
        self.tag = tag
        self.video_id = video_id
        self.local_disk = False
        self.saas_flag = saas_flag



class ImageFile(FileData):
    def __init__(self, *, file_name, file_path, video_id, tag, library_type, saas_flag):
        super().__init__(file_name, file_path)
        self.video_id = video_id
        self.image_id = video_id
        self.tag = tag
        self.library_type = library_type
        self.saas_flag = saas_flag

