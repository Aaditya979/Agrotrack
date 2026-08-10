import json
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "densenet_leaf_model.keras"
LABELS_PATH = "labels.json"
IMAGE_PATH = "sample_leaf.jpg"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise Exception("Could not read sample image.")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = np.array(image, dtype=np.float32)
image = tf.keras.applications.densenet.preprocess_input(image)
image = np.expand_dims(image, axis=0)

preds = model.predict(image, verbose=0)[0]
class_index = int(np.argmax(preds))
confidence = float(preds[class_index])
predicted_label = labels[str(class_index)]

print("Predicted class:", predicted_label)
print("Confidence:", round(confidence, 4))