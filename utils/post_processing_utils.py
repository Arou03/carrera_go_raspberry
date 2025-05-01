""
"""

yolo_utils.py - Utility Functions for YOLOv5 Model Post-Processing

This module includes essential functions to process YOLO model outputs and image

preprocessing. These utilities cover:

- Non-Maximum Suppression (NMS) for filtering overlapping boxes

- Rescaling boxes to match original image dimensions

- Letterboxing (resize + pad) for inference compatibility

- Format conversions between bounding box representations

Author: Manda Andriamaromanana

"""

import torch

import numpy as np

import cv2

import time

import torchvision

def box_iou(box1, box2):
    """
    Compute the IoU between each pair of boxes from box1 and box2.

    Parameters
    ----------
    box1 : torch.Tensor
        Shape (N, 4) in [x1, y1, x2, y2] format.
    box2 : torch.Tensor
        Shape (M, 4) in [x1, y1, x2, y2] format.

    Returns
    -------
    torch.Tensor
        IoU matrix of shape (N, M).
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2 - inter_area
    return inter_area / union_area

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping detection boxes.

    Parameters
    ----------
    prediction : torch.Tensor
        Output tensor from YOLO model, shape (batch_size, num_preds, 85).
    conf_thres : float
        Confidence score threshold to retain boxes.
    iou_thres : float
        IOU threshold for suppression.
    classes : list of int, optional
        Filter by class index. If None, all classes are used.
    agnostic : bool
        If True, NMS is class-agnostic.
    multi_label : bool
        If True, allow multi-label per box.
    labels : list, optional
        Ground truth labels to add to predictions.
    max_det : int
        Maximum number of boxes to keep per image.
    nm : int
        Number of masks or extra channels.

    Returns
    -------
    list of torch.Tensor
        One list per image, each tensor shape (N, 6), with [x1, y1, x2, y2, conf, cls].
    """
    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    device = prediction.device
    mps = "mps" in device.type
    if mps:
        prediction = prediction.cpu()
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        if merge and (1 < n < 3e3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            break

    return output

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescale bounding boxes from processed image shape to original image shape.

    Parameters
    ----------
    img1_shape : tuple
        Shape of the preprocessed image (height, width).
    boxes : np.ndarray or torch.Tensor
        Bounding boxes in format [x1, y1, x2, y2].
    img0_shape : tuple
        Shape of the original image (height, width).
    ratio_pad : tuple, optional
        Tuple containing (gain, pad) used during preprocessing.

    Returns
    -------
    np.ndarray or torch.Tensor
        Rescaled bounding boxes.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image to match desired shape with stride alignment.

    Parameters
    ----------
    im : np.ndarray
        Input image.
    new_shape : tuple of int
        Target shape (height, width).
    color : tuple of int
        Padding color.
    auto : bool
        Whether to use automatic padding.
    scaleFill : bool
        Stretch image to fill new_shape.
    scaleup : bool
        Allow upscaling.
    stride : int
        Stride multiple constraint.

    Returns
    -------
    tuple
        Resized image, scale ratio, and padding (dw, dh).
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def xyxy2xywh(x):
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height].

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Input bounding boxes.

    Returns
    -------
    np.ndarray or torch.Tensor
        Converted bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def xywh2xyxy(x):
    """
    Convert bounding boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2].

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Input bounding boxes.

    Returns
    -------
    np.ndarray or torch.Tensor
        Converted bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def clip_boxes(boxes, shape):
    """
    Clip bounding box coordinates to stay within image boundaries.

    Parameters
    ----------
    boxes : np.ndarray or torch.Tensor
        Bounding boxes in format [x1, y1, x2, y2].
    shape : tuple
        Image shape as (height, width).

    Returns
    -------
    None
        Modifies boxes in-place.
    """
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])
