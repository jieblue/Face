from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,

)




def create_connection(host="127.0.0.1", port="19530",
                      user='root', password='Milvus'):
    return connections.connect('default', host=host, port=port,
                               user=user, password=password)


def create_collection(collection_name, description='',
                      fields=None):
    if has_collection(collection_name):
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


def has_collection(collection_name):
    return utility.has_collection(collection_name)


def get_collection(collection_name, load=True):
    collection = Collection(collection_name)
    if load:
        collection.load()
    return collection


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



