import redis

from utils import log_util

logger = log_util.get_logger(__name__)


redis_host = "10.10.38.99"
redis_port = 6379
redis_password = "ec#TsnS8YFfMSSQ#"
redis_db = 9
logger.info("redis_client init")
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, db=redis_db)
logger.info(f"redis_client init success, host: {redis_host}, port: {redis_port}, password: {redis_password}, db: {redis_db}")


def set_expire_after_24hours(key, value):
    """
    Set the key-value pair to the Redis database with a specific expiration time.

    Args:
        key (str): The key of the key-value pair.
        value (str): The value of the key-value pair.

    Returns:
        str: The result of setting the key-value pair.
    """
    return redis_client.setex(key, 24*60*60, value)


def get_value_by_key(key):
    """
    Get the value of the key-value pair from the Redis database.

    Args:
        key (str): The key of the key-value pair.

    Returns:
        str: The value of the key-value pair.
    """
    return redis_client.get(key)




