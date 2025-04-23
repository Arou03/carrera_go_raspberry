import cv2
import numpy as np
import ncnn  # Assure-toi que le binding Python pour NCNN est installé

# Chargement du modèle NCNN
net = ncnn.Net()
net.opt.num_threads = 8  # Ajuste selon ton CPU
net.load_param("best.ncnn.param")
net.load_model("best.ncnn.bin")
print("Model loaded successfully.")

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

fps = 30
# Normalisation
mean_vals = [0.0, 0.0, 0.0]
norm_vals = [1/255.0, 1/255.0, 1/255.0]

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 320)


prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prétraitement de l'image
    img_input, r, pad_x, pad_y = letterbox(frame, (640, 640))
    img_input = img_input.astype(np.uint8)  # Assure-toi d'être en uint8

    # Conversion avec NCNN en précisant la largeur et la hauteur
    in_mat = ncnn.Mat.from_pixels(img_input, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_input.shape[1], img_input.shape[0])
    in_mat.substract_mean_normalize(mean_vals, norm_vals)

    # Définition de l'entrée pour l'extracteur
    ex = net.create_extractor()
    ex.input("in0", in_mat)
    
    # Exécution de l'inférence et extraction de la sortie
    ret, out = ex.extract("out0")
    if ret != 0:
        print("Erreur lors de l'extraction de la sortie")
        continue
    
    # On suppose que la sortie contient 25200 x 7 (1, 25200, 7) comme avec le modèle ONNX
    # Convertir la sortie NCNN.Mat en numpy
    num_detections = out.h  # nombre de lignes
    output = np.array([out.row(i) for i in range(num_detections)])

    best_detections = {}

    for det in output:
        cx, cy, w, h = det[:4]
        obj_conf = det[4]
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        class_conf = class_scores[class_id]
        confidence = obj_conf * class_conf

        if confidence > 0.2:  # Ajuste ce seuil si nécessaire
            # Dépadding et mise à l’échelle inverse
            cx = (cx - pad_x) / r
            cy = (cy - pad_y) / r
            w = w / r
            h = h / r
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            w = int(w)
            h = int(h)

            # Garder la meilleure détection pour chaque classe
            if class_id not in best_detections or confidence > best_detections[class_id][0]:
                best_detections[class_id] = (confidence, (x, y, w, h))


    for class_id, (conf, (x, y, w, h)) in best_detections.items():
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{class_names[class_id]}: {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    current_time = cv2.getTickCount()
    time_diff = (current_time - prev_time) / cv2.getTickFrequency()

    if time_diff >= 1 / fps:
        cv2.imshow("NCNN Detection", frame)
        prev_time = current_time

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
