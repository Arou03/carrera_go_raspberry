import sys
import cv2
import torch
import time
import argparse
import numpy as np
import yaml
import re
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

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def is_on_track(object_bbox, final_mask, threshold=-2, velocity=None,
                w_dark=0.5, w_speed=0.5, speed_threshold=1.0, score_threshold=0.5):
    x1, y1, x2, y2 = object_bbox
    object_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_dist = float('-inf')
    for contour in contours:
        dist = cv2.pointPolygonTest(contour, object_center, measureDist=True)
        max_dist = max(max_dist, dist)

    dark_score_max_dist = 10
    dark_score = min(max_dist / dark_score_max_dist, 1.0) if max_dist >= 0 else max(max_dist / dark_score_max_dist, -1.0)
    motion_score = 0.0
    if velocity is not None:
        speed = (velocity[0]**2 + velocity[1]**2)**0.5
        motion_score = min(speed / speed_threshold, 1.0)

    score = w_dark * dark_score + w_speed * motion_score
    return score >= score_threshold


def main(config, show_window=False, frame_skip=0):
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

    class_names = load_class_names(config["data_yaml_path"])
    target_class = config.get("target_class", None)
    target_class_id = class_names.index(target_class) if target_class in class_names else None

    fps_buffer = deque(maxlen=max_fps_window)
    dark_mask_history = deque(maxlen=dark_mask_history_length)
    trackers = {}
    model = load_model(config['model_path'])

    use_test_video = config.get("use_test_video", False)
    video_source = 'video_test/sample1.mp4' if use_test_video else 0
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

    # 🎥 Create VideoWriter if test mode
    out_writer = None
    if use_test_video:
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        existing = list(results_dir.glob("test_result_*.mp4"))
        indices = [int(re.findall(r"(\d{3})", f.stem)[0]) for f in existing if re.findall(r"(\d{3})", f.stem)]
        next_index = max(indices) + 1 if indices else 0
        output_path = results_dir / f"test_result_{next_index:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (resize_size, resize_size))

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
        for _ in range(frame_skip):
            cap.read()
        if not ret:
            break
        frame_resized = preprocess_frame(frame)
        t1 = time.perf_counter()
        timing_data["frame_acquisition"] += t1 - t0

        # Dark zone
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

        # Detection
        t0 = time.perf_counter()
        img_numpy = prepare_image(frame_resized, size=resize_size)
        preds_raw = onnx_inference(model, img_numpy)
        preds = non_max_suppression(torch.tensor(preds_raw), conf_thres=conf_threshold, iou_thres=iou_threshold)
        t1 = time.perf_counter()
        timing_data["yolo_inference"] += t1 - t0

        # Tracking
        t0 = time.perf_counter()
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
        tracked_objects = {}
        for class_id, detections in detections_by_class.items():
            if class_id not in trackers:
                trackers[class_id] = CentroidTracker(max_distance=config.get('tracker_max_distance', 50))
            tracked_objects[class_id] = trackers[class_id].update(detections)
        t1 = time.perf_counter()
        timing_data["tracking_update"] += t1 - t0

        # Annotations
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
                        bbox, final_mask, velocity=velocity,
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

        # FPS & display
        t0 = time.perf_counter()
        current_time = time.time()
        avg_fps, prev_time = calculate_fps(prev_time, current_time, fps_buffer)
        t1 = time.perf_counter()
        timing_data["fps_calc"] += t1 - t0

        overlayed = overlay_mask(frame_resized.copy(), final_mask, alpha=mask_alpha)
        final_frame = display_fps(overlayed, avg_fps)

        if out_writer:
            out_writer.write(final_frame)
        if show_window:
            cv2.imshow("Tracking + Dark Zone", final_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        total_fps += avg_fps
        frame_count += 1

    cap.release()
    if out_writer:
        out_writer.release()
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
    parser.add_argument("--skip", type=int, default=0, help="Number of frames to skip between processed frames")
    args = parser.parse_args()

    config = load_config(args.config)
    config["use_test_video"] = args.test
    main(config, show_window=args.show, frame_skip=args.skip)
