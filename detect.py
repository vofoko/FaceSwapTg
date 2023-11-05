import onnxruntime
import numpy as np
import cv2
import requests


#import torchvision
import time
#import torch

#def box_iou(box1, box2, eps=1e-7):
#    """
#    Return intersection-over-union (Jaccard index) of boxes.
#    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
##    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
#    Arguments:
#        box1 (Tensor[N, 4])
#        box2 (Tensor[M, 4])
#        eps
#    Returns:
#        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
 #   """

#    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
#    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

#    # IoU = inter / (area1 + area2 - inter)
#    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.
    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    #y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y




def py_cpu_nms(dets, scores, thresh = 0.98):
    # dets:(m,5)  thresh:scaler

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    #scores = dets[:, 4]
    keep = []

    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h


        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1


    return keep


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels
    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output


    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    # Settings
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose([1, 0])[xc[xi]]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box = x[:, :4]
        cls = x[:, -1].reshape((box.shape[0], -1))
        mask = np.zeros((box.shape[0], 0)) # x[:, 5:]
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

        conf = np.max(cls, axis=1).reshape(box.shape[0], -1)
        j = np.argmax(cls, axis=1).reshape(box.shape[0], -1)
        x = np.concatenate([box, conf, j, mask], 1)

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        x = x[x[:, 4].argsort()[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = py_cpu_nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections


        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output



def detect_face(image=None, SIZE = 320, show_time = False, debug = False):
    if image is None and debug:
        image = cv2.imread('faces.jpg')

    factor_resize = max(image.shape)/SIZE
    image = cv2.resize(image, (SIZE, SIZE))
    image = np.transpose(image, [2, 0, 1])/255
    x = np.array([image]).astype(np.float32)

    if show_time:
        import time
        start_time = time.time()

    if debug:
        ort_session = onnxruntime.InferenceSession("weights/yolov8n_face.onnx")
    else:
        response = requests.get('https://storage.yandexcloud.net/face-networks/yolov8n_face.onnx')
        ort_session = onnxruntime.InferenceSession(response.content)

    if show_time:
        print("--- %s seconds ---" % (time.time() - start_time))


    #def to_numpy(tensor):
    #    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    #np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    if show_time:
        print("--- %s seconds ---" % (time.time() - start_time))

    faces=non_max_suppression(np.array(ort_outs[0]), nc=1, max_det=10)[0]
    return faces[:, :4] * factor_resize




def predict_keypoints(image, SIZE = 112, show_time = False, debug = False):
    if image is None and debug:
        image = cv2.imread('1.jpg')

    factor_resize = max(image.shape)/SIZE
    image = cv2.resize(image, (SIZE, SIZE))
    x = np.transpose(image, [2, 0, 1])/255
    x = np.array([x]).astype(np.float32)

    if show_time:
        import time
        start_time = time.time()

    if debug:
        ort_session = onnxruntime.InferenceSession("weights/pfld_pretrained.onnx")
    else:
        response = requests.get('https://storage.yandexcloud.net/face-networks/model_landmark.onnx')
        ort_session = onnxruntime.InferenceSession(response.content)

    if show_time:
        print("--- %s seconds ---" % (time.time() - start_time))

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outs = ort_session.run(None, ort_inputs)

    keypoints = ort_outs[1][0] * SIZE * factor_resize

    #visual_keypoints(image, keypoints[192:196]) #100
    return keypoints

#predict_keypoints_onnx()


# 1 - 32 ([0:66] points) contour of face
# 33 - 41 ([66:84] points) - left brown
# 42 - 50 ([84:102] points) - right brown
# 51 - 59 ([102:120] points) - nose
# 60 - 67 ([120:136] points) - left eye
# 68 - 75 ([120:152] points) - right eye
# 76 - 75 ([120:152] points) - right eye
# 76 - 95 ([152:196] points) - mouse
# 96 [192:194] - left zrachOk
# 97 [194:196] - left zrachOk