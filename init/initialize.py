from milvus_tool.local_milvus import *
from config.config import *
from pymilvus import utility


conf = get_config()
milvus_conf = conf['milvus']
# print(milvus_conf)

con = create_connection(host=milvus_conf['host'], port=milvus_conf['port'],
                        user=milvus_conf['user'], password=milvus_conf['password'])


fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
        ]
image_collection = create_collection('image_faces','face of images', fields)


video_collection = create_collection('video_faces','face of videos', fields)



create_index(image_collection, 'embedding')
create_index(video_collection, 'embedding')
