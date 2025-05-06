# Object Tracking & On-Track Verification System

## Project Overview
This project provides a complete pipeline for real-time object detection and tracking, combined with a custom method to verify if an object remains within a predefined "track" area. It is designed to work with a YOLOv5 model converted to ONNX format and supports live webcam feeds or prerecorded test videos.

⚠️ **Context**:  
This system is intended to **generate real-time data to feed into a reinforcement learning (RL) algorithm**. Specifically, the RL agent will control a **toy car that accelerates based on electric impulses** or a continuous control signal. This project represents the **perception and tracking half** of a broader RL-based system designed to make such cars drive themselves.

---

## 📑 Table of Contents

- [🏎️ Object Tracking & On-Track Verification System](#-object-tracking--on-track-verification-system)
  - [📌 Project Overview](#project-overview)
  - [🧠 Methodology](#methodology)
  - [📂 Project Structure](#project-structure)
  - [🔍 Description of `main.py`](#description-of-mainpy)
    - [Configuration Load](#configuration-load)
    - [Model Loading](#model-loading)
    - [Video Capture Setup](#video-capture-setup)
    - [Dark Area Extraction](#dark-area-extraction)
    - [YOLO Inference](#yolo-inference)
    - [Tracking](#tracking)
    - [On-Track Check](#on-track-check)
    - [Display](#display)
  - [🚀 How to Launch](#how-to-launch)
  - [⚙️ Explanation of Arguments](#️explanation-of-arguments)
  - [📁 Configuration (`config.yaml`)](#configuration-configyaml)
  - [🔧 Suggested Improvements](#suggested-improvements)
  - [👤 Author](#author)
  - [📚 Références techniques](#références-techniques)


---

## Methodology
The system consists of the following components:

- **YOLOv5 Object Detection**: Detects objects (e.g., toy cars) using a trained model exported to ONNX.
- **Centroid Tracking**: Assigns persistent IDs to detected objects across frames.
- **Dark Area Segmentation**: Approximates a race track by detecting dark zones in the environment.
- **On-Track Verification**: Determines if an object is inside the valid dark region (i.e., on track).
- **Real-Time FPS Monitoring**: Calculates and displays live FPS.
- **Optional HTTP Sender**: Sends tracked positions and on-track status via HTTP to another system (e.g., an RL agent).

---

## Project Structure
```
project/
│
├── main.py                      # Main script for detection + tracking + dark zone detection
├── config.yaml                  # All parameters and thresholds
├── utils/
│   ├── yolo_utils.py           # NMS, box scaling, letterboxing
│   ├── tracking_utils.py       # ONNX model loading, tracker class
│   ├── dark_area_utils.py      # Track detection using dark areas
│   ├── fps_utils.py            # FPS display and smoothing
│   └── utils_send_info.py      # HTTP sender for inference metadata
└── video_test/
    └── sample1.mp4             # Test video example

```

---

## Description of `main.py`

### Configuration Load
```python
config = load_config(args.config)
```

### Model Loading
```python
model = load_model(config['model_path'])
```

### Video Capture Setup
```python
video_source = 'video_test/sample1.mp4' if config.get("use_test_video") else 0
cap = cv2.VideoCapture(video_source)
```

### Dark Area Extraction
```python
raw_mask, contours = get_dark_mask(frame_resized, ...)
```

### YOLO Inference
```python
img_numpy = prepare_image(frame_resized)
preds_raw = onnx_inference(model, img_numpy)
```

### Tracking
```python
tracked_objects = tracker.update(detections)
```

### On-Track Check
```python
is_on = is_on_track(bbox, final_mask, velocity=..., w_dark=..., w_speed=...)
```

### Display
```python
display_fps(frame, fps)
```

---

## How to Launch

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the script with webcam:
```bash
python main.py --show
```

### 3. Run the script with a test video:
```bash
python main.py --test --show
```

### 4. Send detections to an HTTP server (e.g., Flask backend):
```bash
python main.py --send 10 --show
```

---

## Explanation of Arguments

### `--show`
Displays a real-time window with annotations (bounding boxes, FPS, track overlay).

### `--test`
Uses the video file at `video_test/sample1.mp4` instead of the webcam input.

### `--send N`
Enables HTTP POST requests to a local server every N frames.

---

## Configuration (`config.yaml`)
All parameters are adjustable in the `config.yaml` file:

```yaml
model_path: "model/best.onnx"          # Path to ONNX model
resize_size: 640                        # Input frame size (for both width and height)
conf_threshold: 0.4                     # YOLO detection confidence threshold
iou_threshold: 0.5                      # IOU threshold for NMS
mask_alpha: 0.4                         # Opacity of track overlay
contour_update_interval: 1             # Update track contours every N frames
send_interval: 0                        # 0 disables HTTP sending, else send every N frames
dark_area_threshold: -1                # Point-in-contour threshold for on-track detection
upper_black: [179, 50, 90]             # HSV values to define black/dark regions
min_area: 35000                         # Minimum area to consider a contour as part of the track
target_class: "mario"                  # Class to apply the on-track logic (e.g., "mario")
weight_dark: 0.101                     # Importance of proximity to dark zone
weight_speed: 0.7                      # Importance of speed (motion)
on_track_score_threshold: 0.599        # Score above which object is "on track"
tracker_max_distance: 50               # Used to avoid class mixups by using separate trackers
```

---

## Suggested Improvements

- **Add Training & Evaluation Phase**  
  Include scripts to train YOLOv5 from scratch on your dataset, and test it in simulation or real environments.

- **Trajectory Learning with KD-Tree**  
  During testing, build a KD-tree of valid (x, y) positions to learn and model the valid track layout. Use this structure to correct `is_on_track()` logic or detect anomalies.

- **Embedded Deployment**  
  Convert the ONNX model to NCNN, TensorRT, or similar formats for low-power deployment (e.g., Raspberry Pi, Jetson Nano). Enables onboard inference for real-world toy cars.

- **Use as RL Feedback**  
  This perception pipeline can be extended to work hand-in-hand with reinforcement learning algorithms. The output (position, velocity, on/off track) can directly serve as observation space input.

- **Path Prediction / Safety Heuristics**  
  Add prediction layers such as a Kalman filter or GRU to infer next likely positions and support proactive control.

- **Web Visualization Interface**  
  Build a lightweight dashboard to monitor the toy car's status, trajectory, and whether it's off-track, using Flask + WebSockets.

---

## Author  
**Manda Andriamaromanana**  
Université Paris-Saclay — Master 1 ISD  

### Acknowledgments  
This project was supervised by **Steven Martin**, and I thank **Thomas Gerald** for his helpful guidance on how to get started.

---

## Références techniques

Ce document compile les principales ressources et outils utilisés dans le cadre de ce projet de détection et suivi d’objets pour une voiture miniature pilotée automatiquement via un signal électrique.

---

### 🔍 YOLOv5 (You Only Look Once, version 5)

- **Lien officiel** : [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **Documentation** : [https://docs.ultralytics.com/models/yolov5](https://docs.ultralytics.com/models/yolov5/)
- **Description** : YOLOv5 est un modèle de détection d’objets en temps réel développé par Ultralytics. Il est basé sur PyTorch et se distingue par sa vitesse d’inférence et sa précision. Il prend en charge l’exportation vers des formats comme ONNX ou TensorRT, ce qui le rend idéal pour l’embarqué.

---

### 🧰 Roboflow

- **Site web** : [https://roboflow.com/](https://roboflow.com/)
- **GitHub** : [https://github.com/roboflow](https://github.com/roboflow)
- **Description** : Roboflow est une plateforme en ligne permettant de créer, annoter, nettoyer et exporter facilement des jeux de données pour l'entraînement de modèles de vision par ordinateur. Très utile pour organiser les classes et exporter dans des formats compatibles avec YOLO.

---

### 🧠 ChatGPT (OpenAI)

- **Page officielle** : [https://openai.com/chatgpt](https://openai.com/chatgpt)
- **Documentation API** : [https://platform.openai.com/docs](https://platform.openai.com/docs)
- **Description** : ChatGPT est un modèle de langage développé par OpenAI. Il a été utilisé pour générer, corriger, documenter et organiser le code Python du projet, ainsi que pour proposer des optimisations et rédiger la documentation technique.

---

### 🏁 Carrera GO!!!

- **Site officiel** : [https://www.carrera-toys.com/](https://www.carrera-toys.com/)
- **Boutique & modèles** : [https://www.carreraslots.com/](https://www.carreraslots.com/)
- **Description** : Carrera GO!!! est une gamme de circuits de course pour voitures miniatures (échelle 1:43). Ces voitures sont alimentées par impulsions électriques via les rails. Le projet utilise cette base physique comme environnement pour tester le système de suivi automatique de véhicules.

---

### 📚 Autres ressources utiles

- **ONNX Runtime** : [https://onnxruntime.ai/](https://onnxruntime.ai/)
- **Conversion ONNX vers FP16 / NCNN** : [https://github.com/onnx/onnx](https://github.com/onnx/onnx)
- **OpenCV (vision par ordinateur)** : [https://opencv.org/](https://opencv.org/)
- **SciPy (KD-Tree)** : [https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html)

