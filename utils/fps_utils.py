"""
fps_utils.py - Frame Rate Monitoring Utilities

Provides functions for real-time measurement and overlay of frames-per-second (FPS) in a video stream.
Useful for profiling and monitoring performance in computer vision pipelines.

Functions include:
- Calculating smoothed FPS from timestamps.
- Displaying FPS as text overlay on video frames.

Author: Manda Andriamaromanana
"""
import cv2

def calculate_fps(prev_time, current_time, fps_buffer):
    """
    Calculate the average frames per second (FPS) over a buffer of recent values.

    Parameters
    ----------
    prev_time : float
        Timestamp of the previous frame (in seconds).
    current_time : float
        Timestamp of the current frame (in seconds).
    fps_buffer : list of float
        A list storing recent FPS values for temporal smoothing.

    Returns
    -------
    avg_fps : float
        Averaged FPS over the buffer.
    current_time : float
        Updated previous timestamp (used in next frame).
    """
    fps = 1.0 / (current_time - prev_time)
    fps_buffer.append(fps)
    avg_fps = sum(fps_buffer) / len(fps_buffer)
    return avg_fps, current_time

def display_fps(frame, fps):
    """
    Draw the FPS value on a video frame.

    Parameters
    ----------
    frame : np.ndarray
        Input image frame in BGR format.
    fps : float
        Frames per second value to be rendered on the image.

    Returns
    -------
    np.ndarray
        Annotated frame with FPS text overlay.
    """
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame
