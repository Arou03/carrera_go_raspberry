import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("yolov5n_custom.onnx")
class_names = ['mario', 'yoshi']

def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
	h, w = image.shape[:2]
	r = min(new_shape[0] / h, new_shape[1] / w)
	resized = cv2.resize(image, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)

	# Padding
	dw = new_shape[1] - resized.shape[1]
	dh = new_shape[0] - resized.shape[0]
	top, bottom = dh // 2, dh - (dh // 2)
	left, right = dw // 2, dw - (dw // 2)

	padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	return padded, r, left, top

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	img_input, r, pad_x, pad_y = letterbox(frame, (640, 640))
	blob = cv2.dnn.blobFromImage(img_input, 1/255.0, (640, 640), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward()
	output = outputs[0]  # (25200, 7)

	best_detections = {}

	for det in output:
		cx, cy, w, h = det[:4]
		obj_conf = det[4]
		class_scores = det[5:]
		class_id = np.argmax(class_scores)
		class_conf = class_scores[class_id]
		confidence = obj_conf * class_conf

		if confidence > 0.4:
			# Dépadding et mise à l’échelle inverse
			cx = (cx - pad_x) / r
			cy = (cy - pad_y) / r
			w = w / r
			h = h / r
			x = int(cx - w / 2)
			y = int(cy - h / 2)
			w = int(w)
			h = int(h)

			if class_id not in best_detections or confidence > best_detections[class_id][0]:
				best_detections[class_id] = (confidence, (x, y, w, h))

	for class_id, (conf, (x, y, w, h)) in best_detections.items():
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
		cv2.putText(frame, class_names[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (0, 255, 0), 2)

	cv2.imshow("Detection", frame)
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
