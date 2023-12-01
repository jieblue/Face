from entity.file_entity import VideoFile
import os


def video_validate(video_file: VideoFile):
    validate_file_path(video_file.file_path)

    if video_file.file_name is None or video_file.file_name == "":
        raise ValueError("File path is empty.")

    if video_file.video_id is None or video_file.video_id == "":
        raise ValueError("Video id is empty.")

    if video_file.tag is None or video_file.tag == "":
        raise ValueError("Tag is empty.")


def validate_file_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError("Path does not exists.")
