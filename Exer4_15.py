# Example 4.17 (Modified)
# CNN Layers Visualization - VGG16 AND VGG19
# Filter and Feature-Map Visualization with selectable model

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input as pp_vgg16
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# =========================================================
# MODEL SELECTION (if-elif-else)
# =========================================================
MODEL_CHOICE = "VGG19"      # <-- change to "VGG16" to switch

if MODEL_CHOICE == "VGG16":
    model = VGG16(weights="imagenet")
    preprocess_input = pp_vgg16
elif MODEL_CHOICE == "VGG19":
    model = VGG19(weights="imagenet")
    preprocess_input = pp_vgg19
else:
    raise ValueError(f"Unknown MODEL_CHOICE: {MODEL_CHOICE!r}")

print(f"Loaded {MODEL_CHOICE}")
model.summary()

# =========================================================
# PART 2 - List every layer with its index
# =========================================================
print("\n" + "=" * 70)
print(f"{MODEL_CHOICE} LAYERS")
print("=" * 70)

for n, layer in enumerate(model.layers):
    print(f"{n:3d}  {layer.name:<20}  {layer.__class__.__name__}")

# =========================================================
# PART 3 - Show conv filter shapes and bias shapes
# =========================================================
print("\n" + "=" * 70)
print("CONVOLUTIONAL LAYER FILTER SHAPES")
print("=" * 70)

for n, layer in enumerate(model.layers):
    if "conv" in layer.name:
        filters, biases = layer.get_weights()
        print(f"{n:3d}  {layer.name:<15} "
              f"filters={filters.shape}  biases={biases.shape}")

# =========================================================
# PART 4 - Visualize filters from MULTIPLE conv layers
# =========================================================
def show_filters(layer_name, n_filters=4, cmap="gray"):
    layer = model.get_layer(layer_name)
    filters, _ = layer.get_weights()

    # Normalize to 0-1 for display
    fmin, fmax = filters.min(), filters.max()
    filters = (filters - fmin) / (fmax - fmin + 1e-8)

    n_show = min(n_filters, filters.shape[-1])

    plt.figure(figsize=(2.2 * n_show, 2.6))
    for i in range(n_show):
        f = filters[:, :, :, i]
        # Collapse RGB to grayscale (only meaningful for block1)
        if f.shape[-1] == 3:
            f = np.mean(f, axis=2)
        else:
            f = f[:, :, 0]  # show first input channel for deeper layers

        ax = plt.subplot(1, n_show, i + 1)
        ax.imshow(f, cmap=cmap)
        ax.set_title(f"F{i+1}", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"{MODEL_CHOICE} filters – {layer_name} "
                 f"({filters.shape[2]}x{filters.shape[3]} kernels)")
    plt.tight_layout()
    plt.show()

# Visualize filters for each block's first conv layer
filter_layers = ["block1_conv1", "block2_conv1",
                 "block3_conv1", "block4_conv1", "block5_conv1"]

for lname in filter_layers:
    show_filters(lname, n_filters=4)

# =========================================================
# PART 5 - Load an image
# =========================================================
image_path = "elephant.webp"          # <-- change to your own image
print("\nLoading image:", image_path)

image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
print("Image shape:", image.shape)

# =========================================================
# PART 6 - Visualize feature maps + compute statistics
# =========================================================
layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

outputs = [model.get_layer(name).output for name in layer_names]
feature_model = Model(inputs=model.input, outputs=outputs)

print("\nComputing feature maps...")
feature_maps = feature_model.predict(image, verbose=1)

print("\n" + "=" * 70)
print("FEATURE-MAP STATISTICS PER LAYER")
print("=" * 70)
print(f"{'Layer':<15}{'Shape':<20}{'Mean':>10}{'Std':>10}{'% Zero':>10}")

for lname, fmap in zip(layer_names, feature_maps):
    shape_str = f"{fmap.shape[1]}x{fmap.shape[2]}x{fmap.shape[3]}"
    mean_val  = float(fmap.mean())
    std_val   = float(fmap.std())
    pct_zero  = 100.0 * float((fmap == 0).mean())
    print(f"{lname:<15}{shape_str:<20}{mean_val:>10.3f}"
          f"{std_val:>10.3f}{pct_zero:>9.1f}%")

# Visualize a grid of feature maps per layer
for lname, fmap in zip(layer_names, feature_maps):
    n_features = min(fmap.shape[-1], 64)
    square = int(np.ceil(np.sqrt(n_features)))

    plt.figure(figsize=(12, 12))
    for i in range(n_features):
        ax = plt.subplot(square, square, i + 1)
        ax.imshow(fmap[0, :, :, i], cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"{MODEL_CHOICE} – Feature maps: {lname} "
                 f"({fmap.shape[1]}x{fmap.shape[2]}, {fmap.shape[-1]} channels)")
    plt.tight_layout()
    plt.show()

print("\nVisualization completed!")