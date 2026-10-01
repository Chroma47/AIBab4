# Example 4.15 - Webcam Classification

import cv2
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import (
    preprocess_input,
    decode_predictions
)

# Set up the model
model = VGG16(
    weights='imagenet',
    classes=1000
)

# Image size for VGG16
image_size = 224

# Display model summary
model.summary()

# Open webcam
camera = cv2.VideoCapture(0)

while camera.isOpened():

    # Read frame from webcam
    ret, cam_frame = camera.read()

    if not ret:
        print("Failed to read from webcam.")
        break

    # Resize image
    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    # Convert BGR to RGB
    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to NumPy array
    image = np.asarray(image)

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # Preprocess image
    image = preprocess_input(image)

    # Predict
    preds = model.predict(
        image,
        verbose=0
    )

    # Decode predictions
    label = decode_predictions(
        preds,
        top=1
    )

    # Get result
    class_name = label[0][0][1]
    confidence = label[0][0][2]

    # Display result
    text = "{} : {:.1f}%".format(
        class_name,
        confidence * 100
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow(
        "Classification",
        cam_frame
    )

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break

# Release webcam
camera.release()

# Close OpenCV windows
cv2.destroyAllWindows()