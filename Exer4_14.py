# Example 4.15 (Modified) - Webcam Classification with if-else model selection

import cv2
import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101,
    MobileNet, MobileNetV2,
    InceptionV3,
    DenseNet121,
    EfficientNetB0,
    NASNetMobile,
)

# ------------------------------------------------------------------
# Per-model preprocessing / decoding helpers
# ------------------------------------------------------------------
from tensorflow.keras.applications.vgg16      import preprocess_input as pp_vgg16,      decode_predictions as dp_vgg16
from tensorflow.keras.applications.vgg19      import preprocess_input as pp_vgg19,      decode_predictions as dp_vgg19
from tensorflow.keras.applications.resnet50   import preprocess_input as pp_resnet50,   decode_predictions as dp_resnet50
from tensorflow.keras.applications.resnet     import preprocess_input as pp_resnet101,  decode_predictions as dp_resnet101
from tensorflow.keras.applications.mobilenet  import preprocess_input as pp_mobilenet,  decode_predictions as dp_mobilenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp_mobilenetv2, decode_predictions as dp_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as pp_inception, decode_predictions as dp_inception
from tensorflow.keras.applications.densenet   import preprocess_input as pp_densenet,   decode_predictions as dp_densenet
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_effnet,   decode_predictions as dp_effnet
from tensorflow.keras.applications.nasnet     import preprocess_input as pp_nasnet,     decode_predictions as dp_nasnet

# ------------------------------------------------------------------
# >>> CHOOSE THE MODEL HERE <<<
# Valid values:
#   "VGG16", "VGG19", "ResNet50", "ResNet101",
#   "MobileNet", "MobileNetV2", "InceptionV3",
#   "DenseNet121", "EfficientNetB0", "NASNetMobile"
# ------------------------------------------------------------------
MODEL_CHOICE = "ResNet50"

# ------------------------------------------------------------------
# if-elif-else selection of the model
# ------------------------------------------------------------------
if MODEL_CHOICE == "VGG16":
    model = VGG16(weights='imagenet')
    preprocess_input = pp_vgg16
    decode_predictions = dp_vgg16
    image_size = 224

elif MODEL_CHOICE == "VGG19":
    model = VGG19(weights='imagenet')
    preprocess_input = pp_vgg19
    decode_predictions = dp_vgg19
    image_size = 224

elif MODEL_CHOICE == "ResNet50":
    model = ResNet50(weights='imagenet')
    preprocess_input = pp_resnet50
    decode_predictions = dp_resnet50
    image_size = 224

elif MODEL_CHOICE == "ResNet101":
    model = ResNet101(weights='imagenet')
    preprocess_input = pp_resnet101
    decode_predictions = dp_resnet101
    image_size = 224

elif MODEL_CHOICE == "MobileNet":
    model = MobileNet(weights='imagenet')
    preprocess_input = pp_mobilenet
    decode_predictions = dp_mobilenet
    image_size = 224

elif MODEL_CHOICE == "MobileNetV2":
    model = MobileNetV2(weights='imagenet')
    preprocess_input = pp_mobilenetv2
    decode_predictions = dp_mobilenetv2
    image_size = 224

elif MODEL_CHOICE == "InceptionV3":
    model = InceptionV3(weights='imagenet')
    preprocess_input = pp_inception
    decode_predictions = dp_inception
    image_size = 299   # InceptionV3 requires 299x299

elif MODEL_CHOICE == "DenseNet121":
    model = DenseNet121(weights='imagenet')
    preprocess_input = pp_densenet
    decode_predictions = dp_densenet
    image_size = 224

elif MODEL_CHOICE == "EfficientNetB0":
    model = EfficientNetB0(weights='imagenet')
    preprocess_input = pp_effnet
    decode_predictions = dp_effnet
    image_size = 224

elif MODEL_CHOICE == "NASNetMobile":
    model = NASNetMobile(weights='imagenet')
    preprocess_input = pp_nasnet
    decode_predictions = dp_nasnet
    image_size = 224

else:
    raise ValueError(
        f"Unknown MODEL_CHOICE: {MODEL_CHOICE!r}. "
        "Pick one of: VGG16, VGG19, ResNet50, ResNet101, "
        "MobileNet, MobileNetV2, InceptionV3, DenseNet121, "
        "EfficientNetB0, NASNetMobile."
    )

print(f"Selected model: {MODEL_CHOICE}  (input {image_size}x{image_size})")
model.summary()

# ------------------------------------------------------------------
# Webcam loop
# ------------------------------------------------------------------
camera = cv2.VideoCapture(0)

frame_count = 0
total_time = 0.0

while camera.isOpened():

    ret, cam_frame = camera.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # Resize to the size required by the selected model
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add batch dimension
    image = np.expand_dims(rgb, axis=0).astype(np.float32)

    # Preprocess using the selected model's normalizer
    image = preprocess_input(image)

    # Predict + time it
    t0 = time.perf_counter()
    preds = model.predict(image, verbose=0)
    ms = (time.perf_counter() - t0) * 1000.0
    frame_count += 1
    total_time += ms

    # Decode
    label = decode_predictions(preds, top=1)
    class_name = label[0][0][1]
    confidence = label[0][0][2] * 100

    avg_ms = total_time / frame_count

    # Overlay
    text1 = f"{MODEL_CHOICE}: {class_name} ({confidence:.1f}%)"
    text2 = f"Inference: {ms:.1f} ms  |  Avg: {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)"

    cv2.putText(cam_frame, text1,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(cam_frame, text2,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 200, 255), 2, cv2.LINE_AA)

    cv2.imshow(f"{MODEL_CHOICE} Webcam Classification", cam_frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------
print(f"\n=== Summary for {MODEL_CHOICE} ===")
print(f"Frames processed : {frame_count}")
print(f"Average latency  : {total_time/frame_count:.1f} ms")
print(f"Average FPS      : {1000.0/(total_time/frame_count):.1f}")