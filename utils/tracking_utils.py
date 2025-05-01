"""
tracking_utils.py - Inference and Tracking Utilities for ONNX YOLOv5

Provides helper functions and classes to:
- Load and run ONNX-format YOLOv5 models
- Preprocess video frames for inference
- Track object detections over time with persistent IDs
- Draw annotations for visual debugging

Author: Manda Andriamaromanana
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import onnxruntime as ort
import yaml


def load_model(model_path):
    """
    Load an ONNX model using ONNX Runtime.

    Parameters
    ----------
    model_path : str or Path
        Path to the ONNX model file.

    Returns
    -------
    onnxruntime.InferenceSession
        Loaded ONNX model session.
    """
    return ort.InferenceSession(str(model_path))


def onnx_inference(session, image):
    """
    Run inference on a single image using an ONNX model.

    Parameters
    ----------
    session : onnxruntime.InferenceSession
        Loaded ONNX model session.
    image : np.ndarray
        Input image of shape (1, 3, H, W) with float32 pixel values in [0, 1].

    Returns
    -------
    np.ndarray
        Model output predictions.
    """
    inputs = {session.get_inputs()[0].name: image.astype(np.float32)}
    outputs = session.run(None, inputs)
    return outputs[0]


def load_class_names(yaml_path):
    """
    Load class names from a YOLOv5 data YAML file.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the .yaml file.

    Returns
    -------
    list of str
        List of class names.
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']


def preprocess_frame(frame, size=(640, 640)):
    """
    Resize a video frame to a fixed size.

    Parameters
    ----------
    frame : np.ndarray
        Original image frame.
    size : tuple of int
        Target size as (width, height).

    Returns
    -------
    np.ndarray
        Resized image frame.
    """
    return cv2.resize(frame, size)


def prepare_image(frame, size=640):
    """
    Convert image to input format required by YOLOv5 ONNX model.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.
    size : int
        Target image size (square shape).

    Returns
    -------
    np.ndarray
        Image in shape (1, 3, size, size), normalized to [0, 1].
    """
    from .post_processing_utils import letterbox

    img = letterbox(frame, new_shape=(size, size))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def draw_box_label(frame, bbox, label, color=(255, 0, 0), thickness=2):
    """
    Draw a labeled bounding box on a frame.

    Parameters
    ----------
    frame : np.ndarray
        The target image frame.
    bbox : list of int
        Bounding box coordinates [x1, y1, x2, y2].
    label : str
        Text label to show.
    color : tuple of int, optional
        Color for the box and text (default is red).
    thickness : int, optional
        Line thickness.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, color, thickness, cv2.LINE_AA)


class CentroidTracker:
    """
    Simple object tracker that assigns persistent IDs using centroid distance.

    Attributes
    ----------
    max_distance : float
        Maximum allowed centroid distance for ID matching.
    next_object_id : int
        ID counter for assigning new object IDs.
    objects : dict
        Dictionary mapping object IDs to bounding boxes.
    """

    def __init__(self, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # object_id -> bbox
        self.max_distance = max_distance

    def update(self, detections):
        """
        Match new detections to existing objects using centroid proximity.

        Parameters
        ----------
        detections : list of list
            List of bounding boxes [x1, y1, x2, y2].

        Returns
        -------
        dict
            Updated mapping of object_id -> bbox.
        """
        if len(detections) == 0:
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self.objects[self.next_object_id] = det
                self.next_object_id += 1
            return self.objects

        object_ids = list(self.objects.keys())
        object_boxes = list(self.objects.values())

        D = cdist(
            np.array([self._center(box) for box in object_boxes]),
            np.array([self._center(box) for box in detections])
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        updated = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = detections[col]
            updated[object_id] = detections[col]
            used_rows.add(row)
            used_cols.add(col)

        for i, det in enumerate(detections):
            if i not in used_cols:
                self.objects[self.next_object_id] = det
                updated[self.next_object_id] = det
                self.next_object_id += 1

        return updated

    def _center(self, bbox):
        """
        Compute center of bounding box.

        Parameters
        ----------
        bbox : list of int
            Bounding box [x1, y1, x2, y2].

        Returns
        -------
        list of float
            Center [cx, cy].
        """
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
