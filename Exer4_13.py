# Example 4.15 (Modified) - Multi-Model Webcam Classification
# Uncomment / add any model in the MODELS_TO_RUN list below to enable it.

import cv2
import numpy as np
import time
from collections import deque

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
# Registry: name -> (constructor, preprocess_input, decode_predictions, input_size)
# ------------------------------------------------------------------
from tensorflow.keras.applications.vgg16     import preprocess_input as pp_vgg16,     decode_predictions as dp_vgg16
from tensorflow.keras.applications.vgg19     import preprocess_input as pp_vgg19,     decode_predictions as dp_vgg19
from tensorflow.keras.applications.resnet50  import preprocess_input as pp_resnet50,  decode_predictions as dp_resnet50
from tensorflow.keras.applications.resnet    import preprocess_input as pp_resnet101, decode_predictions as dp_resnet101
from tensorflow.keras.applications.mobilenet import preprocess_input as pp_mobilenet, decode_predictions as dp_mobilenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pp_mobilenetv2, decode_predictions as dp_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as pp_inception, decode_predictions as dp_inception
from tensorflow.keras.applications.densenet  import preprocess_input as pp_densenet,  decode_predictions as dp_densenet
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_effnet, decode_predictions as dp_effnet
from tensorflow.keras.applications.nasnet    import preprocess_input as pp_nasnet,    decode_predictions as dp_nasnet

MODEL_REGISTRY = {
    "VGG16":        (VGG16,          pp_vgg16,      dp_vgg16,      224),
    "VGG19":        (VGG19,          pp_vgg19,      dp_vgg19,      224),
    "ResNet50":     (ResNet50,       pp_resnet50,   dp_resnet50,   224),
    "ResNet101":    (ResNet101,      pp_resnet101,  dp_resnet101,  224),
    "MobileNet":    (MobileNet,      pp_mobilenet,  dp_mobilenet,  224),
    "MobileNetV2":  (MobileNetV2,    pp_mobilenetv2,dp_mobilenetv2,224),
    "InceptionV3":  (InceptionV3,    pp_inception,  dp_inception,  299),
    "DenseNet121":  (DenseNet121,    pp_densenet,   dp_densenet,   224),
    "EfficientNetB0":(EfficientNetB0,pp_effnet,     dp_effnet,     224),
    "NASNetMobile": (NASNetMobile,   pp_nasnet,     dp_nasnet,     224),
}

# ------------------------------------------------------------------
# >>> UNCOMMENT the models you want to run <<<
# ------------------------------------------------------------------
MODELS_TO_RUN = [
    "VGG16",
    "VGG19",
    "ResNet50",
    "MobileNetV2",
    "DenseNet121",
    "EfficientNetB0",
    # "InceptionV3",
    # "NASNetMobile",
    # "ResNet101",
    # "MobileNet",
]

# ------------------------------------------------------------------
# Load selected models
# ------------------------------------------------------------------
loaded_models = {}
for name in MODELS_TO_RUN:
    print(f"Loading {name}...")
    ctor, pp, dp, size = MODEL_REGISTRY[name]
    model = ctor(weights='imagenet')
    loaded_models[name] = {
        "model": model,
        "preprocess": pp,
        "decode": dp,
        "size": size,
        "times": deque(maxlen=30),   # rolling window for average latency
        "last": ("", 0.0, 0.0),      # (class, conf, ms)
    }

# Open webcam
camera = cv2.VideoCapture(0)

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 200, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
    (128, 0, 255), (0, 128, 255), (255, 128, 0), (128, 255, 0),
]

while camera.isOpened():
    ret, cam_frame = camera.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # Run each enabled model on the same frame
    y = 30
    for idx, (name, entry) in enumerate(loaded_models.items()):
        size = entry["size"]

        # Prepare input for THIS model's expected size
        frame = cv2.resize(cam_frame, (size, size))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = np.expand_dims(rgb, axis=0).astype(np.float32)
        batch = entry["preprocess"](batch)

        # Time inference
        t0 = time.perf_counter()
        preds = entry["model"].predict(batch, verbose=0)
        ms = (time.perf_counter() - t0) * 1000.0
        entry["times"].append(ms)

        # Decode
        label = entry["decode"](preds, top=1)
        cls_name = label[0][0][1]
        conf     = label[0][0][2] * 100
        entry["last"] = (cls_name, conf, ms)

        avg_ms = sum(entry["times"]) / len(entry["times"])

        text = f"{name:<14} {cls_name:<20} {conf:5.1f}%  {ms:6.1f} ms (avg {avg_ms:6.1f})"
        cv2.putText(
            cam_frame, text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLORS[idx % len(COLORS)],
            1,
            cv2.LINE_AA,
        )
        y += 26

    cv2.imshow("Multi-Model Webcam Classification (ESC to quit)", cam_frame)
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------
# Final summary printed to console
# ------------------------------------------------------------------
print("\n===== Final Averages over last 30 frames =====")
print(f"{'Model':<16}{'Avg ms':>10}{'Approx FPS':>12}")
for name, entry in loaded_models.items():
    if entry["times"]:
        avg = sum(entry["times"]) / len(entry["times"])
        print(f"{name:<16}{avg:>10.1f}{1000/avg:>12.1f}")