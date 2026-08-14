import cv2
import numpy as np
from datetime import datetime
import os
import json
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

CAPTURE_FOLDER = "captured"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

MODEL_PATH = "densenet_leaf_model.keras"
LABELS_PATH = "labels.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)


def capture_image():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        raise Exception("Could not open laptop camera")

    print("Press SPACE to capture image. Press ESC to exit.")

    image_path = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break 

        cv2.imshow("Leaf Capture", frame)
        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            filename = f"leaf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.join(CAPTURE_FOLDER, filename)
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            break

        elif key == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()
    return image_path


def analyze_leaf(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Could not read image")

    # Preprocess image for DenseNet
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Predict
    preds = model.predict(image, verbose=0)[0]
    print("RAW PREDICTIONS:", preds)

    class_index = int(np.argmax(preds))
    confidence = float(preds[class_index])

    predicted_class = class_labels.get(str(class_index), "Unknown")
    print("DEBUG CLASS INDEX:", class_index)
    print("DEBUG CLASS NAME:", predicted_class)
    print("DEBUG CONFIDENCE:", confidence)

    # Map class to user-friendly result
    if predicted_class == "Tomato_healthy":
        status = "Healthy"
        risk = "Low"
        alert = "No Alert"

    elif predicted_class == "Tomato_Bacterial_spot":
        status = "Bacterial Spot Detected"
        risk = "High"
        alert = "Immediate Alert"

    elif predicted_class == "Tomato__Tomato_YellowLeaf__Curl_Virus":
        status = "Yellow Leaf Curl Virus Detected"
        risk = "High"
        alert = "Immediate Alert"

    else:
        status = "Uncertain"
        risk = "Medium"
        alert = "Warning Alert"

    return {
        "status": status,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "risk": risk,
        "alert": alert,
        "model_name": "DenseNet121"
    }
