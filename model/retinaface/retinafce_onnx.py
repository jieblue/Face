from model.retinaface.data import cfg_mnet
from model.retinaface.layers.functions.prior_box import PriorBox
from model.retinaface.reutils.box_utils import decode, decode_landm
from model.retinaface.reutils.nms.py_cpu_nms import py_cpu_nms
from utils.face_helper import *
import torch.onnx
import onnxruntime as ort
from utils.img_util import down_image


# retinaface 的ONNX模型，用于人脸检测
class Retinaface_Onnx:
    def __init__(self, path, gpu_id=0):

            # 设置CUDAExecutionProvider, 因为模型输入大小不固定
            # 要设置加速算法为默认
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': gpu_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
        device_name = 'cuda:' + str(gpu_id)
        self.device = torch.device("mps:0")


        so = ort.SessionOptions()
        so.log_severity_level = 3
        # 加载onnx模型到onnxruntime的推理
        self.session = ort.InferenceSession(path, so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    #人脸检测方法
    def detect_faces(self, img_raw, confidence_threshold=0.99,
                     top_k=5000, nms_threshold=0.4, keep_top_k=750, resize=1):

        img = np.float32(img_raw)

        im_height, im_width = img.shape[:2]

        # img = down_image(img)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        # img = torch.from_numpy(img).unsqueeze(0)
        # img = img.to(self.device)
        scale = scale.to(self.device)

        # tic = time.time()

        # with torch.no_grad():
        inputs = {self.input_name: img}
        # start = time.time()
        loc, conf, landms = self.session.run(None, inputs)  # forward pass
        # end = time.time()
        # print('infer time: '+str(end - start))

        loc = to_tensor(loc).to(self.device)
        conf = to_tensor(conf).to(self.device)
        landms = to_tensor(landms).to(self.device)

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print(landms.shape)
        landms = landms.reshape((-1, 5, 2))
        # print(landms.shape)
        landms = landms.transpose((0, 2, 1))
        # print(landms.shape)
        landms = landms.reshape(-1, 10, )
        # print(landms.shape)

        return dets, landms

    # 人脸提取方法， 根据人脸检测得到的landmarks仿射变换人脸部分到face_size大小
    # 返回校准后的人脸列表
    def extract_face(self, img, face_size = 112, confidence=0.99):
        # input is numpy img bgr
        # return is numpy img bgr
        if img is None:
            return []
        img = down_image(img)
        # face_size = 112 if not enhance else 512
        # start = time.time()
        bounding_boxes, landmarks = self.detect_faces(img, confidence_threshold=confidence)
        # end = time.time()
        # print("detect time: " + str(end - start))

        align_faces = []
        if len(bounding_boxes) == 0:
            return align_faces
        raw_img = img.copy()

        for i, (box, lam) in enumerate(zip(bounding_boxes, landmarks)):

            box = box.astype(int)
            face = raw_img[box[1]:box[3], box[0]:box[2]]

            res = align_face(raw_img, lam, face_size=face_size, reorder=True)

            align_faces.append(res)
            #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            #     # draw the landmarks
            #     # cv2.circle(img, (int(lam[0]), int(lam[5])), 1, (0, 0, 255), 4)
            #     # cv2.circle(img, (int(lam[1]), int(lam[6])), 1, (0, 0, 255), 4)
            #     # cv2.circle(img, (int(lam[2]), int(lam[7])), 1, (0, 0, 255), 4)
            #     # cv2.circle(img, (int(lam[3]), int(lam[8])), 1, (0, 0, 255), 4)
            #     # cv2.circle(img, (int(lam[4]), int(lam[9])), 1, (0, 0, 255), 4)
            #     cv2.imwrite(os.path.join(result_path, _id + '_' + str(i) + '.jpg'), res)


        return align_faces


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# np2tensor
def to_tensor(np_array):
    return torch.from_numpy(np_array) #.contiguous()



