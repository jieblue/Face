from entity.file_entity import VideoFile, ImageFile
import os


def video_validate(video_file: VideoFile):
    validate_file_path(video_file)
    if video_file.video_id is None or video_file.video_id == "":
        raise ValueError("Video id is empty.")


def image_validate(image_file: ImageFile):
    validate_file_path(image_file)

    if not os.path.isdir(image_file.file_path):
        raise NotADirectoryError("Path is not a directory.")

    if image_file.image_id is None or image_file.image_id == "":
        raise ValueError("Image id is empty.")


two_gb_bytes = 2 * (1024 ** 3)


def validate_file_path(file):
    # Get file size in bytes
    file_size_bytes = os.path.getsize(file.file_path)
    # Check if file size is greater than 2GB
    if file_size_bytes > two_gb_bytes:
        raise ValueError("File size is greater than 2GB.")
    if not os.path.exists(file.file_path):
        raise FileNotFoundError("Path does not exists.")
    if file.file_name is None or file.file_name == "":
        raise ValueError("File path is empty.")

    if file.tag is None or file.tag == "":
        raise ValueError("Tag is empty.")
