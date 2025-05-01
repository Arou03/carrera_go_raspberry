"""
utils_send_info.py - HTTP Utility for Sending Object Tracking Data

This module provides a utility function to send tracking information of detected objects
(e.g., position and track status) to a remote HTTP server via a POST request. This is 
useful in scenarios such as real-time monitoring of autonomous agents, vehicle telemetry,
or experiment logging in computer vision pipelines.

Function
--------
send_position_http : Send position and state of a tracked object to a specified HTTP server.

Author: Manda Andriamaromanana
"""

import requests

def send_position_http(url, object_center, on_track, frame_id=None):
	"""
	Send the position and status of an object via HTTP POST request in JSON format.

	This function is typically used to report the position of a tracked object (e.g., a car or agent)
	in a video processing pipeline to a local or remote server, for monitoring or analytics purposes.

	Parameters
	----------
	url : str
		The target endpoint where tracking data should be sent. Example: 'http://localhost:5000/track'
	object_center : tuple of int
		The (x, y) coordinates of the center of the detected object.
	on_track : bool
		Boolean flag indicating whether the object is currently on the expected path (e.g., on a track).
	frame_id : int or None, optional
		Frame number associated with the current detection (for logging or synchronization).

	Returns
	-------
	None

	Notes
	-----
	- The payload is sent as JSON with keys: 'x', 'y', 'on_track', and 'frame'.
	- A timeout of 1.0 second is applied to the request to avoid blocking the main process.
	- Status messages are printed to the console indicating success, warning, or error states.

	Examples
	--------
	>>> send_position_http("http://localhost:5000/track", (320, 240), True, frame_id=42)

	This will send the following JSON to the specified URL:
	{
		"x": 320,
		"y": 240,
		"on_track": true,
		"frame": 42
	}
	"""
	payload = {
		"x": object_center[0],
		"y": object_center[1],
		"on_track": on_track,
		"frame": frame_id
	}
	try:
		response = requests.post(url, json=payload, timeout=1.0)
		if response.status_code == 200:
			print(f"[INFO] Position sent successfully: {payload}")
		else:
			print(f"[WARN] Failed to send position: {response.status_code} - {response.text}")
	except requests.RequestException as e:
		print(f"[ERROR] HTTP request failed: {e}")
