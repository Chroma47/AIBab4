# Example 4.14 (Modified) - Webcam Classification with ResNet50 AND DenseNet121

import cv2
import numpy as np
import time

from tensorflow.keras.applications import resnet50, densenet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions as resnet_decode
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import decode_predictions as densenet_decode

# ---------------------------------------------------------------
# Initialize both models
# ---------------------------------------------------------------
print("Loading ResNet50...")
resnet_model = ResNet50(weights='imagenet')

print("Loading DenseNet121...")
densenet_model = DenseNet121(weights='imagenet')

# Optional: show summaries (very long, uncomment if needed)
# print(resnet_model.summary())
# print(densenet_model.summary())

# Open webcam
camera = cv2.VideoCapture(0)

# Image size required by both ResNet50 and DenseNet121
image_size = 224

# Simple running averages for timing
resnet_times = []
densenet_times = []

while camera.isOpened():

    # Read frame from webcam
    ok, cam_frame = camera.read()

    if not ok:
        print("Failed to read from webcam.")
        break

    # Resize frame
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add batch dimension
    image_batch = np.expand_dims(rgb_frame, axis=0)

    # -----------------------------------------------------------
    # ResNet50 prediction
    # -----------------------------------------------------------
    resnet_input = resnet50.preprocess_input(image_batch.copy())

    t0 = time.perf_counter()
    resnet_preds = resnet_model.predict(resnet_input, verbose=0)
    resnet_ms = (time.perf_counter() - t0) * 1000

    resnet_label = resnet_decode(resnet_preds, top=1)
    resnet_name = resnet_label[0][0][1]
    resnet_conf = resnet_label[0][0][2] * 100

    # -----------------------------------------------------------
    # DenseNet121 prediction
    # -----------------------------------------------------------
    densenet_input = densenet.preprocess_input(image_batch.copy())

    t0 = time.perf_counter()
    densenet_preds = densenet_model.predict(densenet_input, verbose=0)
    densenet_ms = (time.perf_counter() - t0) * 1000

    densenet_label = densenet_decode(densenet_preds, top=1)
    densenet_name = densenet_label[0][0][1]
    densenet_conf = densenet_label[0][0][2] * 100

    # -----------------------------------------------------------
    # Overlay results
    # -----------------------------------------------------------
    resnet_text = f"ResNet50:    {resnet_name} ({resnet_conf:.1f}%)  {resnet_ms:.1f} ms"
    densenet_text = f"DenseNet121: {densenet_name} ({densenet_conf:.1f}%)  {densenet_ms:.1f} ms"

    cv2.putText(cam_frame, resnet_text,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

    cv2.putText(cam_frame, densenet_text,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 180, 0), 2)

    # Show webcam
    cv2.imshow("ResNet50 vs DenseNet121 Webcam Classification", cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release webcam
camera.release()
cv2.destroyAllWindows()