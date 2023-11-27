from config.config import get_config
import logging


def get_logger(class_name: str):
    # 获取config信息
    conf = get_config()

    # Create a logger
    logger = logging.getLogger(class_name)
    logger.setLevel(logging.INFO)

    # 获取face_app的配置
    face_app_conf = conf['face_app']
    log_file = face_app_conf['log_file']

    # Define the log file and format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a file handler and set the log format
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)

    # Create a stream handler to print log messages to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
