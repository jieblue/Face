from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db,

)

# 这个文件是对Milvus操作的相关方法， 详见 https://milvus.io/docs

# 创建 collection的连接 db_name 是连接的数据库名称
def create_connection(host="127.0.0.1", port="19530",
                      user='root', password='Milvus', db_name='Face_Search'):
    return connections.connect('default', host=host, port=port,
                               user=user, password=password, db_name=db_name)

def create_connection_without_auth(host="127.0.0.1", port="19530"):
    return connections.connect('default', host=host, port=port)


# 创建 数据库 database_name是数据库名称
def create_database(database_name):

    db.create_database(database_name)

# 创建 collection collection_name 为collection的名字， description为描述，
# filed为参数详细见 https://milvus.io/docs/create_collection.md
# 返回Collection对象
def create_collection(collection_name, description='',
                      fields=None):
    if has_collection(collection_name):
        # 创建成功后， 跳过
        raise Exception('Collection has already existed!')

    if fields is None:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)
        ]
    schema = CollectionSchema(fields, description)
    my_collection = Collection(collection_name, schema)
    return my_collection


# 判断是否存在某个 collection
def has_collection(collection_name):
    return utility.has_collection(collection_name)

# 判断是否存在某个 collection
def has_database(database_name):
    database_arr = db.list_databases("")
    if database_name in database_arr:
        return True
    return False


# 获得某个 collection 对象，
# collection_name 为 collection的名字， load表示是否把collection加载到内存
def get_collection(collection_name, load=True):
    collection = Collection(collection_name)
    if load:
        collection.load()
    return collection


# 对一个collection创建索引 collection为Collection对象
# field 为 要创建索引的字段名称 str
# index_type 为索引类型 , metric_type 为计算向量距离的方式，nlist为每个索引簇的大小
# 详见 https://milvus.io/docs/build_index.md
def create_index(collection, field, index_type="IVF_SQ8",
                 metric_type="IP", nlist=100):
    index_params = {
        "index_type": index_type,
        "metric_type": metric_type,
        "params": {"nlist": nlist},
    }
    collection.create_index(field, index_params)




def insert_data(collection: Collection, data):
    res = collection.insert(data)
    return res.primary_keys
    #


def search_vectors(collection, field, vectors, output_fields,
                   search_params=None, limit=3, nprobe=50):

    if search_params is None:
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": nprobe},
        }
    limit = 16383 if limit > 16383 else limit
    res = collection.search(vectors, field, search_params,
                            limit=limit, output_fields=output_fields, round_decimal=4)
    return res



def delete_by_pks(collection: Collection, pks):
    res = []
    for pk in pks:
        del_expr = 'id in ['

        _expr = str(pk) + ']'

        del_expr = del_expr + _expr
        # print(del_expr)
        delete_res = collection.delete(del_expr)
        # print(err_count)
        err_count = delete_res.err_count
        isSuccess = True if (err_count is None or err_count==0) else False
        res.append({
            'primary_key': pk,
            'isSuccess': isSuccess
        })
    # collection.flush()
    return res




def delete_by_pks_batch(collection: Collection, pks):
    del_expr = 'id in ['
    _len = len(pks)
    for i, pk in enumerate(pks):
        _id = pk
        _expr = str(_id) + ']' if i == (_len - 1) else str(_id) + ','
        del_expr = del_expr + _expr

    delete_res = collection.delete(del_expr)

    return delete_res.err_count


def delete_by_filed(collection: Collection, field, value):
    expr = field+'=="' + value+'"'
    # print(expr)
    entitys = collection.query(expr)
    if entitys is None or len(entitys) == 0:
        return 0

    del_expr = 'id in ['
    _len = len(entitys)
    for i, entity in enumerate(entitys):
        _id = entity['id']
        _expr = str(_id) + ']' if i == (_len-1) else str(_id) + ','
        del_expr = del_expr + _expr
        # print(id)

    # print(del_expr)
    delete_res = collection.delete(del_expr)
    # collection.flush()

    return delete_res.err_count



