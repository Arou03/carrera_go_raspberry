"""
main.py - Real-Time Object Tracking and Dark Area Detection

This script performs object detection and tracking on a video stream (either from a webcam or a test video file).
It also detects dark zones in the video using HSV masking and overlays the results. A YOLOv5 model in ONNX format
is used for object detection, and the results are tracked using a centroid-based tracker. Additional information
such as whether an object is on track (inside a dark zone) is calculated, and data can optionally be sent to
an external server.

Usage:
    python main.py [--config CONFIG_PATH] [--show] [--send N] [--test]

Arguments:
    --config     Path to YAML config file with parameters (default: config.yaml)
    --show       Display the video feed with overlays in a window
    --send       Interval (in frames) for sending tracking data to server (0 disables)
    --test       Use test video instead of webcam

Dependencies:
    - OpenCV
    - PyTorch
    - ONNX Runtime
    - NumPy
    - YAML
    - Custom modules: yolo_utils, dark_area_utils, tracking_utils, fps_utils, utils_send_info

Author: Manda Andriamaromanana
"""

import sys
import cv2
import torch
import time
import argparse
import numpy as np
import yaml
from pathlib import Path
from collections import deque, defaultdict

YOLOV5_PATH = Path("yolov5")
sys.path.append(str(YOLOV5_PATH))

from utils.utils_send_info import send_position_http
from utils.post_processing_utils import non_max_suppression, scale_boxes

from utils.dark_area_utils import (
    get_dark_mask, overlay_mask, merge_contours_by_distance
)
from utils.tracking_utils import (
    load_class_names, load_model, onnx_inference, prepare_image, preprocess_frame, draw_box_label, CentroidTracker
)
from utils.fps_utils import calculate_fps, display_fps

fps_buffer = None
dark_mask_history = None

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Configuration parameters as a dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def is_on_track(
    object_bbox,
    final_mask,
    threshold=-2,
    velocity=None,
    w_dark=0.5,
    w_speed=0.5,
    speed_threshold=1.0,
    score_threshold=0.5
):
    """
    Assess whether an object is considered 'on track' based on a weighted combination of
    its spatial location (distance to dark zone contours) and its motion (speed).

    The function computes two normalized scores:
    - `dark_score`: Based on the distance from the object's center to the nearest contour
      of a dark zone. Higher if deeper inside, lower if near the edge or outside.
    - `motion_score`: Based on the speed of the object relative to a speed threshold.

    The final decision is based on a weighted sum of these scores compared to a global
    `score_threshold`.

    Parameters
    ----------
    object_bbox : list[int]
        Bounding box coordinates as [x1, y1, x2, y2].
    final_mask : np.ndarray
        Binary mask (0 or 255) representing the current dark zones.
    threshold : float, optional
        Unused in this version but preserved for compatibility (default: -2).
    velocity : tuple[float, float], optional
        Velocity vector (dx, dy) representing object movement between frames.
    w_dark : float, optional
        Weight assigned to the dark area score (default: 0.5).
    w_speed : float, optional
        Weight assigned to the motion score (default: 0.5).
    speed_threshold : float, optional
        Speed value that corresponds to a maximum motion score of 1.0 (default: 1.0).
    score_threshold : float, optional
        Minimum combined score required to consider the object 'on track' (default: 0.5).

    Returns
    -------
    bool
        True if the weighted score exceeds or equals the `score_threshold`, otherwise False.

    Notes
    -----
    - The function assumes that a higher motion score can compensate for being outside the
      dark area, and vice versa, depending on the chosen weights.
    - If no velocity is provided, motion_score is set to 0.
    - You can tune `w_dark`, `w_speed`, and `score_threshold` to control sensitivity.

    Example
    -------
    >>> is_on_track([100, 100, 120, 120], mask, velocity=(3, 2),
    ...             w_dark=0.7, w_speed=0.3, score_threshold=0.6)
    True
    """

    x1, y1, x2, y2 = object_bbox
    object_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    # Find distance to nearest contour (positive if inside, negative if outside)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_dist = float('-inf')
    for contour in contours:
        dist = cv2.pointPolygonTest(contour, object_center, measureDist=True)
        max_dist = max(max_dist, dist)

    # Normalize dark score: inside = 1, near edge = 0.5, outside = 0
    dark_score_max_dist = 10

    if max_dist >= 0:
        dark_score = min(max_dist / dark_score_max_dist, 1.0)
    else:
        dark_score = max(max_dist / dark_score_max_dist, -1.0)

    # Motion score: normalized between 0 and 1
    motion_score = 0.0
    if velocity is not None:
        speed = (velocity[0]**2 + velocity[1]**2)**0.5
        motion_score = min(speed / speed_threshold, 1.0)

    # Weighted combination
    score = w_dark * dark_score + w_speed * motion_score
    # print(f'score = {score} ; dark = {dark_score} ; motion = {motion_score}')
    return score >= score_threshold


def main(config, show_window=False):
    """
    Run the main loop for object detection, tracking, and visualization.
    Only applies "on track" logic to the specified target class.
    """
    resize_size = config['resize_size']
    conf_threshold = config['conf_threshold']
    iou_threshold = config['iou_threshold']
    max_fps_window = config['max_fps_window']
    mask_alpha = config['mask_alpha']
    contour_update_interval = config['contour_update_interval']
    send_interval = config['send_interval']
    upper_black = config.get('upper_black', [179, 75, 95])
    dark_mask_history_length = config.get('dark_mask_history_length', 10)
    dark_area_threshold = config.get('dark_area_threshold', -5)
    min_area = config.get('min_area', 10000)
    dark_score_max_dist = config.get('dark_score_max_distance', 20.0)

    # Load class labels and determine the target class ID
    class_names = load_class_names(config["data_yaml_path"])
    target_class = config.get("target_class", None)
    target_class_id = class_names.index(target_class) if target_class in class_names else None

    fps_buffer = deque(maxlen=max_fps_window)
    dark_mask_history = deque(maxlen=dark_mask_history_length)
    trackers = {} # CentroidTracker(max_distance=config.get('tracker_max_distance', 50))
    model = load_model(config['model_path'])

    video_source = 'video_test/sample1.mp4' if config.get("use_test_video") else 0
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resize_size)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resize_size)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    frame_resized_first = preprocess_frame(first_frame)
    scale_x = frame_resized_first.shape[1] / resize_size
    scale_y = frame_resized_first.shape[0] / resize_size
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    total_fps = 0
    send_count = 0
    final_mask = np.zeros((resize_size, resize_size), dtype=np.uint8)
    merged_contours = None
    timing_data = defaultdict(float)
    prev_time = time.time()

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = preprocess_frame(frame)
        t1 = time.perf_counter()
        timing_data["frame_acquisition"] += t1 - t0

        # Dark area detection and smoothing
        t0 = time.perf_counter()
        raw_mask, contours = get_dark_mask(frame_resized, min_area=min_area, upper_black=upper_black)
        dark_mask_history.append(raw_mask)
        avg_mask = np.mean(dark_mask_history, axis=0).astype(np.uint8)
        _, avg_mask = cv2.threshold(avg_mask, 127, 255, cv2.THRESH_BINARY)

        if frame_count % contour_update_interval == 0:
            contours, _ = cv2.findContours(avg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            merged_contours = merge_contours_by_distance(contours)

        final_mask.fill(0)
        if merged_contours:
            for contour in merged_contours:
                cv2.drawContours(final_mask, [contour], -1, 255, thickness=cv2.FILLED)
        t1 = time.perf_counter()
        timing_data["dark_area_processing"] += t1 - t0

        # Object detection
        t0 = time.perf_counter()
        img_numpy = prepare_image(frame_resized, size=resize_size)
        preds_raw = onnx_inference(model, img_numpy)
        preds = non_max_suppression(torch.tensor(preds_raw), conf_thres=conf_threshold, iou_thres=iou_threshold)
        t1 = time.perf_counter()
        timing_data["yolo_inference"] += t1 - t0

        # Object tracking
        t0 = time.perf_counter()
        detection_classes = []  # parallel list
        detections_by_class = defaultdict(list)
        for det in preds:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img_numpy.shape[2:], det[:, :4], (resize_size, resize_size)).round()
                for (*xyxy, conf, cls) in det:
                    bbox = [
                        int(xyxy[0] * scale_x), int(xyxy[1] * scale_y),
                        int(xyxy[2] * scale_x), int(xyxy[3] * scale_y)
                    ]
                    class_id = int(cls)
                    detections_by_class[class_id].append(bbox)
                    detection_classes.append(int(cls))  # used later for labeling

        tracked_objects = {}  # class_id -> {object_id -> bbox}
        for class_id, detections in detections_by_class.items():
            if class_id not in trackers:
                trackers[class_id] = CentroidTracker(max_distance=config.get('tracker_max_distance', 50))
            tracked = trackers[class_id].update(detections)
            tracked_objects[class_id] = tracked

        t1 = time.perf_counter()
        timing_data["tracking_update"] += t1 - t0

        # Annotation and optional data sending
        for class_id, objects in tracked_objects.items():
            class_name = class_names[class_id]
            for object_id, bbox in objects.items():
                velocity = None
                track_history = trackers[class_id].tracks.get(object_id, [])
                if len(track_history) >= 2:
                    dx = track_history[-1][0] - track_history[-2][0]
                    dy = track_history[-1][1] - track_history[-2][1]
                    velocity = (dx, dy)

                if target_class_id is not None and class_id == target_class_id:
                    is_on = is_on_track(
                        bbox,
                        final_mask,
                        velocity=velocity,
                        w_dark=config.get("weight_dark", 0.5),
                        w_speed=config.get("weight_speed", 0.5),
                        speed_threshold=config.get("speed_threshold", 1.0),
                        score_threshold=config.get("on_track_score_threshold", 0.5)
                    )
                    color = (0, 255, 0) if is_on else (0, 0, 255)
                    label = f"{class_name} ID {object_id}"
                else:
                    is_on = None
                    color = (100, 100, 100)
                    label = f"{class_name}"

                draw_box_label(frame_resized, bbox, label, color=color)

                if is_on is not None and send_interval > 0:
                    send_count += 1
                    if send_count % send_interval == 0:
                        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                        send_position_http("http://localhost:5000/track", center, is_on, frame_id=frame_count)

        # FPS calc
        t0 = time.perf_counter()
        current_time = time.time()
        avg_fps, prev_time = calculate_fps(prev_time, current_time, fps_buffer)
        t1 = time.perf_counter()
        timing_data["fps_calc"] += t1 - t0

        # Final overlay
        overlayed = overlay_mask(frame_resized.copy(), final_mask, alpha=mask_alpha)
        final_frame = display_fps(overlayed, avg_fps)

        if show_window:
            cv2.imshow("Tracking + Dark Zone", final_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        total_fps += avg_fps
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        print(f"\n[INFO] Average FPS over session: {total_fps / frame_count:.2f}")
        print("\n[PROFILE] Total time spent per block (seconds):")
        for k, v in timing_data.items():
            print(f"  {k:30s}: {v:.3f} sec total ({v/frame_count:.4f} sec/frame)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking + Dark Area Detection Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--show", action="store_true", help="Display video window while processing")
    parser.add_argument("--test", action="store_true", help="Use test video instead of webcam")
    args = parser.parse_args()

    config = load_config(args.config)
    config["use_test_video"] = args.test
    main(config, show_window=args.show)
