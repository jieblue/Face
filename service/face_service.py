import numpy

from model.model_onnx import Face_Onnx
from utils.img_util import *
from milvus_tool.local_milvus import *
from utils.frame_util import *
from utils.face_helper import *



#批量提取视频关键帧
#返回result 和 err, err记录读取出错的视频路径
def extract_key_frames_batch(paths):
    result = []
    err = []
    for path in paths:
        key_frames = extract_video(path)
        if key_frames is None:
            err.append(path)
        result.append(key_frames)
    return result, err


#批量获取视频关键帧的人脸图片
#video_spaths为list，list中的每个path是一个视频的关键帧的(文件夹)目录 dir
#返回result 和 err, err记录读取出错的图片路径
def get_videos_faces(model: Face_Onnx, video_paths, enhance=False,
                     confidence =0.99):
    result = []
    err = []
    for keyframes_path in video_paths:
        path_list = []
        for file_name in os.listdir(keyframes_path):
            path_list.append(os.path.join(keyframes_path, file_name))
        # print(path_list)
        _faces, _err = get_align_faces_batch(model, path_list ,enhance=enhance,
                                      confidence=confidence, merge=True)
        err += _err
        # print(len(_faces))
        result.append(_faces)
        # print(len(result[0]))
    return result, err

# 批量获取关键帧中提取出的人脸的向量
#faces_spaths为list，list中的每个path是一个视频的人脸(文件夹)目录 dir
#返回result 和 err, err记录读取出错的图片路径
def get_video_extracted_face_embedding(model: Face_Onnx, faces_paths, threshold=0.5):
    result = []
    err = []
    for faces_path in faces_paths:
        path_list = []
        for file_name in os.listdir(faces_path):
            path_list.append(os.path.join(faces_path, file_name))
        raw_embeddings, _err = get_face_embeddings(model, path_list, aligned=True, merge=True)
        err += _err

        embeddings = squeeze_faces(raw_embeddings, threshold)
        result.append(embeddings)

    return result, err


#批量获取视频关键帧的人脸向量，包含同个视频不同关键帧的人脸去重
#video_spaths为list，list中的每个path是一个视频的关键帧的(文件夹)目录 dir
#返回result 和 err, err记录读取出错的图片路径
def get_videos_face_embedding(model: Face_Onnx, video_paths, enhance=False,
                              confidence =0.99, threshold=0.5):
    result = []
    err = []
    for keyframes_path in video_paths:
        path_list = []
        for file_name in os.listdir(keyframes_path):
            path_list.append(os.path.join(keyframes_path, file_name))
        # print(path_list)
        raw_embeddings, _err = get_face_embeddings(model, path_list, False,
                                      enhance, confidence, True)
        err += _err
        # print('raw: ' + str(len(raw_embeddings)))
        embeddings = squeeze_faces(raw_embeddings, threshold)
        # print('fine: ' + str(len(embeddings)))
        result.append(embeddings)
    return result, err



#批量获取对齐的人脸
#paths为list，list中的每个path是一张图片的路径
#返回result 和 err, err记录读取出错的图片路径
def get_align_faces_batch(model: Face_Onnx, paths,
                          enhance=False, confidence=0.99, merge=False):
    align_faces = []
    err = []
    for path in paths:
        img = cv_imread(path)
        if img is None:
            err.append(path)
            continue

        _align_face = model.extract_face(img, enhance=enhance,
                                         confidence=confidence)

        if merge:
            align_faces += _align_face
        else:
            align_faces.append(_align_face)
    return align_faces, err


#批量获取图片中的人脸向量
#paths为list，list中的每个path是一张图片的路径
#返回result 和 err, err记录读取出错的图片路径
def get_face_embeddings(model: Face_Onnx, paths, aligned=False,
                        enhance=False, confidence=0.99, merge=False):
    embeddings = []
    err = []
    for path in paths:
        img = cv_imread(path)
        if img is None:
            err.append(path)
            continue

        embedding = model.turn2embeddings(img, enhance=enhance, aligned=aligned,
                                          confidence=confidence)
        if merge:
            for single in embedding:
                embeddings.append(single)
        else:
            embeddings.append(embedding)

    return embeddings, err


#人脸向量插入Milvus
def add_embeddings2milvus(collection, faces, flush=False):
    data = [[], []]
    result = []
    for single in faces:
        id = str(single['id'])
        if len(id)>50:
            return []

    for i, face in enumerate(faces):
        id = face['id']
        id = str(id)
        embedding = face['embedding']
        data[0].append(id)
        data[1].append(embedding)

        if i == len(faces)-1 or len(data[0])>=5000:
            pks = insert_data(collection, data)
            # print(pks)
            data[0].clear()
            data[1].clear()
            for pk in pks:
                result.append(
                    {"primary_key": pk ,
                     'isSuccess': True} )
    if flush:
        collection.flush()
    return  result

#按记录id批删除向量， 执行n次删除语句
def delete_face_by_primary_key(collection, primary_keys):
    return delete_by_pks(collection, primary_keys)


#按照记录id批删除向量 只执行一次删除语句
def delete_face_by_primary_key_batch(collection, primary_keys):
    return delete_by_pks_batch(collection, primary_keys)



#按向量对应的人脸图片/视频id批删除向量
def delete_face_by_object_id(collection, object_ids):
    result = []
    for id in object_ids:
        if (type(id) != str):
            id = str(id)
        err_count = delete_by_filed(collection, 'object_id', id)
        isSuccess = True if err_count == 0 else False
        result.append({
            'id': id,
            'isSuccess': isSuccess
        })
    return result


#批量人脸搜索图片
#offset为 跳过搜索结果的前n个entity，用于分页返回的话 offset = 页大小 * 前置页数
def search_face_image(model: Face_Onnx, collection, imgs,
                      enhance=False, score=0.5, limit=10, nprobe=50, offset=0):
    embeddings = []
    for img in imgs:
        embedding = model.turn2embeddings(img, enhance=enhance)
        if len(embedding)==0:
            embeddings.append(numpy.zeros([512], dtype=numpy.float32))
            continue
        embeddings.append(embedding[0])

    search_res = search_vectors(collection, 'embedding', embeddings,
                                output_fields=['object_id'], limit=limit, nprobe=nprobe, offset=offset)
    result = []
    for one in search_res:
        _result = []
        for single in one:
            # print(single)
            if single.score >= score:
                tmp = {
                    # 'primary_key': single.id,
                    'id': single.entity.object_id,
                    'score': single.score
                 }#get_search_result(single.id, single.entity.user_id, single.score)
                _result.append(tmp)
        result.append(_result)

    return result


#批量人脸搜索视频
def search_face_video(model: Face_Onnx, collection, imgs, enhance=False,
                      score=0.5, limit=50, nprobe=50):
    embeddings = []
    for img in imgs:
        embedding = model.turn2embeddings(img, enhance=enhance)
        embeddings.append(embedding[0])

    search_res = search_vectors(collection, 'embedding', embeddings,
                                output_fields=['object_id'], nprobe=nprobe, limit=limit)
    result = []

    # print(len(search_res))
    for one in search_res:
        _result = []
        res_dict = {}
        for single in one:
            # print(single)
            if single.score >= score:
                object_id = str(single.entity.object_id)
                _score = single.score
                # print(_score)
                if object_id in res_dict:
                    if res_dict[object_id] < _score:
                        res_dict[object_id] = _score

                else:
                    res_dict[object_id] = _score

        for item in res_dict.items():
            tmp = {'id': item[0],
                   'score': item[1]}

            _result.append(tmp)

        result.append(_result)

    return result


#批获取人脸图片的质量分数
#返回result 和 err, err记录读取出错的图片路径
def get_face_quality_batch(model: Face_Onnx, paths,
                           aligned=False):
    faces = []
    err = []
    if not aligned:
        _faces, _err = get_align_faces_batch(model, paths, enhance=False, confidence=0.99)
        # print(_faces)
        for single in _faces:
            faces.append(single[0])

    else:
        for path in paths:
            img = cv_imread(path)
            if img is None:
                err.append(path)
                continue
            faces.append(img)


    scores = []
    for face in faces:
        score = model.tface.forward(face)
        scores.append(score)
    return scores, err


# 批获取增强后的对齐人脸图片
#返回result 和 err, err记录读取出错的图片路径
def enhance_face_batch(model: Face_Onnx, paths,
                       aligned=False):
    aligned_faces = []
    enhance_faces = []
    err = []
    if not aligned:
        _aligned_faces, _err = get_align_faces_batch(model, paths, True, 0.99)
        err += _err
        for face in _aligned_faces:
            aligned_faces.append(face[0])

    else:
        for path in paths:
            img = cv_imread(path)
            if img is None:
                err.append(path)
                continue
            img = cv2.resize(img, (512, 512))
            aligned_faces.append(img)

    for face in aligned_faces:
        _face = model.gfpgan.forward(face, True)
        enhance_faces.append(_face)
    return enhance_faces, err


#批量获取对齐的人脸
#imgs为list，list中的每个img是一张opencv格式的图片
#返回result 和 err, err记录错误信息
def get_align_faces_batch_img(model: Face_Onnx, imgs,
                              enhance=False, confidence=0.99, merge=False):
    align_faces = []
    err = []
    for img in imgs:
        _align_face = model.extract_face(img, enhance=enhance,
                                         confidence=confidence)
        if merge:
            align_faces += _align_face
        else:
            align_faces.append(_align_face)
    return align_faces, err


#批量获取图片中的人脸向量
#imgs为list，list中的每个img是一张opencv格式的图片
#返回result 和 err, err记录错误信息，暂时为空list
def get_face_embeddings_img(model: Face_Onnx, imgs, aligned=False,
                            enhance=False, confidence=0.99, merge=False):
    embeddings = []
    err = []
    for img in imgs:

        embedding = model.turn2embeddings(img, enhance=enhance, aligned=aligned,
                                          confidence=confidence)
        if merge:
            for single in embedding:
                embeddings.append(single)
        else:
            embeddings.append(embedding)

    return embeddings, err


# 批获取增强后的对齐人脸图片
#imgs为list，list中的每个img是经过人脸检测得到的人脸图片
#返回result 和 err, err记录错误信息，暂时为空list
def enhance_face_batch_img(model: Face_Onnx, imgs):
    enhance_faces = []
    err = []

    for img in imgs:
        img = cv2.resize(img, (512, 512))
        face = model.gfpgan.forward(img, True)
        enhance_faces.append(face)
    return enhance_faces, err



#批获取人脸图片的质量分数
#imgs为list，list中的每个img是经过人脸检测得到的人脸图片
#返回result 和 err, err记录错误信息，暂时为空list
def get_face_quality_batch_img(model: Face_Onnx, imgs):
    err = []
    result = []
    for img in imgs:
        score = model.tface.forward(img)
        result.append(score)
    return result, err



#批量获取视频关键帧的人脸图片
#imgs_list 是list的list，[ [...], [...], ...]，每个子list包含来自同个视频的关键帧图片
#图片是opencv格式的表示
#返回result 和 err, err记录错误信息，暂时为空list
def get_keyframes_faces_img(model: Face_Onnx, imgs_list, enhance=False,
                            confidence =0.9):
    result = []
    err = []
    for imgs in imgs_list:
        # print(path_list)
        _faces, _err = get_align_faces_batch_img(model, imgs ,enhance=enhance,
                                                 confidence=confidence, merge=True)
        # print(len(_faces))
        err += _err
        # print(len(_faces))
        result.append(_faces)
        # print(len(result[0]))
    return result, err


# 批量获取关键帧中提取出的人脸的向量
#imgs_list 是list的list，[ [...], [...], ...]，每个子list包含来自同个视频的关键帧中提取的人脸图片
#图片是opencv格式的表示
#返回result 和 err, err记录错误信息，暂时为空list
def get_keyframes_faces_imgembedding_img(model: Face_Onnx, imgs_list, threshold=0.5):
    result = []
    err = []
    for imgs in imgs_list:
        raw_embeddings, _err = get_face_embeddings_img(model, imgs, aligned=True, merge=True)
        err += _err

        embeddings = squeeze_faces(raw_embeddings, threshold)
        result.append(embeddings)

    return result, err