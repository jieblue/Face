from milvus_tool.local_milvus import *
from config.config import *


# 获取配置
conf = get_config()
milvus_conf = conf['milvus']
# print(milvus_conf)
#建立连接
con = create_connection(host=milvus_conf['host'], port=milvus_conf['port'],
                        user=milvus_conf['user'], password=milvus_conf['password'],
                        db_name='default')


#创建名为 Face_Search 的数据库
# 项目之前使用的是default数据库，使用特定数据库方便管理
db_name = 'Face_Search'
create_database(db_name)
db.using_database(db_name)

# collection的字段参数
fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
        ]

# 创建collection
image_collection = create_collection('image_faces','face of images', fields)
video_collection = create_collection('video_faces','face of videos', fields)


# 创建索引
create_index(image_collection, 'embedding')
create_index(video_collection, 'embedding')
