# Example 4.17 (Modified)
# VGG19 Filter & Feature-Map Visualization with selectable conv layer n
# Change n = 1 to 2, 3, 4, ... to visualize deeper convolutional layers.

import numpy as np
import matplotlib.pyplot as plt
import math

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


# =========================================================
# PART 0 - Select which conv layer to visualize
# =========================================================
# n = 1  -> block1_conv1   (first  conv layer in the model)
# n = 2  -> block1_conv2   (second conv layer)
# n = 3  -> block2_conv1
# n = 4  -> block2_conv2
# n = 5  -> block3_conv1
# ...
# n = 16 -> block5_conv4   (last conv layer in VGG19)
# =========================================================
n = 1        # <<< change this to 2, 3, 4, ... to pick a deeper layer


# =========================================================
# PART 1 - Load VGG19
# =========================================================
print("Loading VGG19 model...")
model = VGG19(weights="imagenet")


# =========================================================
# PART 2 - Enumerate all conv layers and map n -> layer name
# =========================================================
conv_layers = [l for l in model.layers if "conv" in l.name]

print("\n" + "=" * 70)
print("CONVOLUTIONAL LAYERS IN VGG19")
print("=" * 70)
for i, l in enumerate(conv_layers, start=1):
    f, b = l.get_weights()
    print(f"n = {i:2d}   {l.name:<15}  filters = {f.shape}")

if not (1 <= n <= len(conv_layers)):
    raise ValueError(f"n must be between 1 and {len(conv_layers)}")

target_layer = conv_layers[n - 1]
target_name  = target_layer.name
filters, biases = target_layer.get_weights()
s = filters.shape

print(f"\nSelected layer: n = {n}  ->  {target_name}")
print(f"Filter tensor shape: {s}")


# =========================================================
# PART 3 - Visualize ALL filters of the selected conv layer
# =========================================================
# NOTE: Only block1_conv1 (n = 1) has RGB 3-channel filters that can be
# shown as single grayscale images. For deeper layers the filters have
# many input channels, so we show only the *first* input channel as a
# representative slice.
# =========================================================

def visualize_filters(layer_name, max_show=64):
    layer = model.get_layer(layer_name)
    filters, _ = layer.get_weights()
    s = filters.shape

    # Normalize for display
    fmin, fmax = filters.min(), filters.max()
    filters_norm = (filters - fmin) / (fmax - fmin + 1e-8)

    n_filters = min(s[3], max_show)
    grid = int(math.ceil(math.sqrt(n_filters)))

    plt.figure(figsize=(grid * 1.6, grid * 1.6))

    for i in range(n_filters):
        f = filters_norm[:, :, :, i]        # (kh, kw, in_channels)

        # If 3 input channels (block1) collapse to grayscale;
        # otherwise show first input channel as a slice.
        if f.shape[-1] == 3:
            f_img = np.mean(f, axis=2)
        else:
            f_img = f[:, :, 0]

        ax = plt.subplot(grid, grid, i + 1)
        ax.imshow(f_img, cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{i}", fontsize=6)

    plt.suptitle(f"Filters – {layer_name}  ({n_filters} shown, "
                 f"kernel {s[0]}x{s[1]}, in_ch {s[2]}, out_ch {s[3]})")
    plt.tight_layout()
    plt.show()


visualize_filters(target_name, max_show=64)


# =========================================================
# PART 4 - Load an image
# =========================================================
image_path = "elephant.webp"       # <-- change to your own image

print("\nLoading image:", image_path)
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
print("Image shape:", image.shape)


# =========================================================
# PART 5 - Visualize feature maps of the SELECTED layer
# =========================================================
print(f"\nComputing feature maps of {target_name}...")

feature_model = Model(
    inputs=model.input,
    outputs=model.get_layer(target_name).output
)
fmap = feature_model.predict(image, verbose=0)[0]

print(f"{target_name} output shape: {fmap.shape}")

n_channels = min(fmap.shape[-1], 64)
grid = int(math.ceil(math.sqrt(n_channels)))

plt.figure(figsize=(12, 12))
for i in range(n_channels):
    ax = plt.subplot(grid, grid, i + 1)
    ax.imshow(fmap[:, :, i], cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(f"Feature maps – {target_name} "
             f"({fmap.shape[0]}x{fmap.shape[1]}, {fmap.shape[-1]} channels, "
             f"{n_channels} shown)")
plt.tight_layout()
plt.show()


# =========================================================
# PART 6 - Quantitative comparison for the selected layer
# =========================================================
mean_val = float(fmap.mean())
std_val  = float(fmap.std())
pct_zero = 100.0 * float((fmap == 0).mean())

print("\n" + "=" * 70)
print(f"STATISTICS FOR n = {n}  ({target_name})")
print("=" * 70)
print(f"Spatial size        : {fmap.shape[0]} x {fmap.shape[1]}")
print(f"Number of channels  : {fmap.shape[-1]}")
print(f"Mean activation     : {mean_val:.3f}")
print(f"Std of activations  : {std_val:.3f}")
print(f"Percent zeros (ReLU): {pct_zero:.1f}%")

print("\nVisualization completed!")