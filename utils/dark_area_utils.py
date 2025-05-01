"""
dark_area_utils.py - Dark Zone Segmentation and Visualization Utilities

Provides utility functions for detecting, processing, and overlaying dark regions in video frames.
These regions are typically defined in the HSV color space as low-light or low-saturation areas,
useful for applications such as autonomous driving on tracks.

Functions include:
- Preprocessing and resizing frames.
- HSV-based dark area segmentation.
- Temporal smoothing of binary masks.
- Visualization via overlay.
- Merging spatially-close contours for mask simplification.

Author: Manda Andriamaromanana
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree
from collections import deque

DEFAULT_MIN_AREA = 35000
DEFAULT_RESIZE = (640, 640)
DEFAULT_ALPHA = 0.5
WAIT_TIME_MS = 1
CONTOUR_DISTANCE_THRESHOLD = 250

history_masks = deque(maxlen=30)

def preprocess_frame(frame, size=DEFAULT_RESIZE):
    """
    Resize a frame to a fixed size.

    Parameters
    ----------
    frame : np.ndarray
        The input image/frame.
    size : tuple of int, optional
        Target size (width, height) for resizing.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    return cv2.resize(frame, size)

def get_dark_mask(frame, min_area=DEFAULT_MIN_AREA, upper_black=(360, 75, 95)):
    """
    Detect dark areas in a frame based on HSV thresholding.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame.
    min_area : int, optional
        Minimum area (in pixels) for a contour to be considered valid.
    upper_black : tuple of int, optional
        Upper HSV bound for dark detection.

    Returns
    -------
    output_mask : np.ndarray
        Binary mask with valid dark contours filled.
    valid_contours : list of np.ndarray
        List of filtered contours.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array(upper_black, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    output_mask = np.zeros_like(mask)
    cv2.drawContours(output_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
    return output_mask, valid_contours

def smooth_mask(current_mask, history):
    """
    Smooth binary mask using temporal history and dilation.

    Parameters
    ----------
    current_mask : np.ndarray
        Binary mask for the current frame.
    history : deque of np.ndarray
        Buffer storing previous masks.

    Returns
    -------
    np.ndarray
        Smoothed and dilated binary mask.
    """
    history.append(current_mask)
    accumulated = np.sum(np.stack(history), axis=0)
    binary_mask = np.clip(accumulated, 0, 255).astype(np.uint8)
    return cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)

def overlay_mask(frame, mask, alpha=DEFAULT_ALPHA):
    """
    Overlay a colored mask on the original frame.

    Parameters
    ----------
    frame : np.ndarray
        Original frame.
    mask : np.ndarray
        Binary mask to overlay.
    alpha : float, optional
        Blending factor for the overlay (default is 0.5).

    Returns
    -------
    np.ndarray
        Frame with overlay.
    """
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    highlight = np.zeros_like(frame)
    highlight[mask_resized != 0] = (0, 255, 0)
    return cv2.addWeighted(frame, 1.0, highlight, alpha, 0)

def merge_contours_by_distance(contours, distance_threshold=CONTOUR_DISTANCE_THRESHOLD):
    """
    Merge contours whose centers are within a specified distance.

    Parameters
    ----------
    contours : list of np.ndarray
        List of contours to merge.
    distance_threshold : float, optional
        Maximum center-to-center distance for merging (default is 250).

    Returns
    -------
    list of np.ndarray
        List of merged contours.
    """
    if len(contours) > 3:
        return contours

    centers = np.array([
        (x + w // 2, y + h // 2)
        for x, y, w, h in (cv2.boundingRect(cnt) for cnt in contours)
    ], dtype=np.float32)

    if len(centers) == 0:
        return []

    tree = cKDTree(centers)
    merged_contours = []
    visited = set()

    for i, contour in enumerate(contours):
        if i in visited:
            continue

        nearby_indices = tree.query_ball_point(centers[i], distance_threshold)
        merged_contour = np.vstack([contours[j] for j in nearby_indices])
        visited.update(nearby_indices)
        merged_contours.append(merged_contour)

    return merged_contours
