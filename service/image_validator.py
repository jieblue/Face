from entity.file_entity import ImageFile
import os


def image_validate(image_file: ImageFile):
    validate_file_path(image_file.file_path)


def validate_file_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError("Path does not exists.")