# Example 4.13 - Webcam Classification with ResNet50

import cv2
import numpy as np

from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

# Initialize the model
model = ResNet50(weights='imagenet')

# Display model summary
print(model.summary())

# Open webcam
camera = cv2.VideoCapture(0)

# Image size required by ResNet50
image_size = 224

while camera.isOpened():

    # Read frame from webcam
    ok, cam_frame = camera.read()

    if not ok:
        print("Failed to read from webcam.")
        break

    # Resize frame
    frame = cv2.resize(
        cam_frame,
        (image_size, image_size)
    )

    # Convert image to array
    numpy_image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Add batch dimension
    image_batch = np.expand_dims(
        numpy_image,
        axis=0
    )

    # Preprocess image
    processed_image = resnet50.preprocess_input(
        image_batch.copy()
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

    # Get class name and confidence
    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100

    # Display prediction
    text = f"ResNet50: {class_name} ({confidence:.1f}%)"

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
    cv2.imshow(
        "ResNet50 Webcam Classification",
        cam_frame
    )

    # Press ESC to quit
    key = cv2.waitKey(1)

    if key == 27:
        break

# Release webcam
camera.release()

# Close OpenCV windows
cv2.destroyAllWindows()