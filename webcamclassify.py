# Example 4.13 - Webcam Image Classification with VGG16

import cv2
import numpy as np

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions
)

# Load VGG16 model with ImageNet weights
model = VGG16(weights='imagenet')

# Open the webcam
camera = cv2.VideoCapture(0)

# Image size required by VGG16
image_size = 224

while True:

    # Capture frame from webcam
    ok, cam_frame = camera.read()

    if not ok:
        print("Failed to read from webcam.")
        break

    # Resize frame
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # Convert BGR (OpenCV) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert image to NumPy array
    image_batch = np.expand_dims(rgb_frame, axis=0)

    # Preprocess image for VGG16
    processed_image = preprocess_input(
        image_batch.astype(np.float32)
    )

    # Get predictions
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Decode predictions
    label = decode_predictions(
        predictions,
        top=1
    )

    # Get predicted class name
    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100

    # Display prediction on original webcam frame
    text = f"VGG16: {class_name} ({confidence:.1f}%)"

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Show webcam
    cv2.imshow("VGG16 Webcam Classification", cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(1)

    if key == 27:
        break

# Release webcam
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()