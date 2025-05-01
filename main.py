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
    load_model, onnx_inference, prepare_image, preprocess_frame, draw_box_label, CentroidTracker
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

def is_on_track(object_bbox, final_mask, threshold=-5):
    """
    Check if the center of a bounding box lies within a detected dark area.

    Parameters
    ----------
    object_bbox : list of int
        Bounding box [x1, y1, x2, y2].
    final_mask : np.ndarray
        Binary mask of the dark area.
    threshold : float, optional
        Margin threshold for contour inclusion (default is -5).

    Returns
    -------
    bool
        True if the object is inside the dark area, else False.
    """
    x1, y1, x2, y2 = object_bbox
    object_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.pointPolygonTest(contour, object_center, measureDist=False) > threshold for contour in contours)

def main(config, show_window=False):
    """
    Run the main loop for object detection, tracking, and visualization.

    Parameters
    ----------
    config : dict
        Configuration parameters loaded from YAML.
    show_window : bool
        If True, displays the annotated video stream in a window.
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

    fps_buffer = deque(maxlen=max_fps_window)
    dark_mask_history = deque(maxlen=dark_mask_history_length)
    tracker = CentroidTracker(max_distance=config.get('tracker_max_distance', 50))
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

        # Dark area detection and mask smoothing
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
        detections = []
        for det in preds:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img_numpy.shape[2:], det[:, :4], (resize_size, resize_size)).round()
                for (*xyxy, conf, cls) in det:
                    bbox = [
                        int(xyxy[0] * scale_x), int(xyxy[1] * scale_y),
                        int(xyxy[2] * scale_x), int(xyxy[3] * scale_y)
                    ]
                    detections.append(bbox)
        tracked_objects = tracker.update(detections)
        t1 = time.perf_counter()
        timing_data["tracking_update"] += t1 - t0

        # Annotation and optional data sending
        for object_id, bbox in tracked_objects.items():
            is_on = is_on_track(bbox, final_mask, dark_area_threshold)
            color = (0, 255, 0) if is_on else (0, 0, 255)
            draw_box_label(frame_resized, bbox, f"ID {object_id}", color=color)

            if send_interval > 0:
                send_count += 1
                if send_count % send_interval == 0:
                    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    send_position_http("http://localhost:5000/track", center, is_on, frame_id=frame_count)

        # FPS calculation
        t0 = time.perf_counter()
        current_time = time.time()
        avg_fps, prev_time = calculate_fps(prev_time, current_time, fps_buffer)
        t1 = time.perf_counter()
        timing_data["fps_calc"] += t1 - t0

        # Final overlay and optional display
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
    parser.add_argument("--send", type=int, default=0, help="Send data every N detections (0 to disable sending)")
    parser.add_argument("--test", action="store_true", help="Use test video instead of webcam")
    args = parser.parse_args()

    config = load_config(args.config)
    config["use_test_video"] = args.test
    main(config, show_window=args.show)
