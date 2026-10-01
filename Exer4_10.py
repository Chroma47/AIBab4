# Example 4.11 (Modified)
# Classify the same image with both VGG16 and VGG19, compare performance

import time
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input as preprocess_vgg16, decode_predictions as decode_vgg16)
from tensorflow.keras.applications.vgg19 import (
    VGG19, preprocess_input as preprocess_vgg19, decode_predictions as decode_vgg19)

# ---- Image path -------------------------------------------------
img_path = 'elephant.webp'

# Load and prepare the image once (same for both models)
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


# ---- Helper: run one model --------------------------------------
def classify(model, preprocess_fn, decode_fn, x, name):
    x_pre = preprocess_fn(x.copy())

    # Warm-up (first call includes graph construction overhead)
    _ = model.predict(x_pre, verbose=0)

    # Timed inference (average of 3 runs)
    times = []
    for _ in range(3):
        t0 = time.time()
        preds = model.predict(x_pre, verbose=0)
        times.append(time.time() - t0)
    avg_time = sum(times) / len(times)

    results = decode_fn(preds, top=5)[0]

    print(f"\n=== {name} ===")
    print(f"Parameters   : {model.count_params():,}")
    print(f"Inference    : {avg_time*1000:.1f} ms (avg of 3 runs)")
    print("Top-5 predictions:")
    for i, (imagenet_id, label, score) in enumerate(results):
        print(f"  {i+1}. {label:<30} ({score*100:.2f}%)")

    return {
        'name': name,
        'params': model.count_params(),
        'time': avg_time,
        'top1_label': results[0][1],
        'top1_score': results[0][2],
        'results': results,
    }


# ---- Load and run VGG16 -----------------------------------------
print("Loading VGG16...")
vgg16 = VGG16(weights='imagenet')
r16 = classify(vgg16, preprocess_vgg16, decode_vgg16, x, "VGG16")

# ---- Load and run VGG19 -----------------------------------------
print("\nLoading VGG19...")
vgg19 = VGG19(weights='imagenet')
r19 = classify(vgg19, preprocess_vgg19, decode_vgg19, x, "VGG19")


# ---- Comparison summary -----------------------------------------
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"{'model':<8} | {'params':>12} | {'time (ms)':>10} | "
      f"{'top-1 label':<20} | {'score':>7}")
print("-" * 70)
for r in [r16, r19]:
    print(f"{r['name']:<8} | {r['params']:>12,} | "
          f"{r['time']*1000:>10.1f} | {r['top1_label']:<20} | "
          f"{r['top1_score']*100:>6.2f}%")

# Side-by-side top-5 comparison
print("\nTop-5 side by side:")
print(f"{'rank':>4} | {'VGG16 label':<25} {'score':>7} | "
      f"{'VGG19 label':<25} {'score':>7}")
print("-" * 80)
for i in range(5):
    l16, s16 = r16['results'][i][1], r16['results'][i][2]
    l19, s19 = r19['results'][i][1], r19['results'][i][2]
    print(f"{i+1:>4} | {l16:<25} {s16*100:>6.2f}% | "
          f"{l19:<25} {s19*100:>6.2f}%")