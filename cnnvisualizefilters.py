# Example 4.17
# CNN Layers Visualization
# VGG19 Filter and Feature Map Visualization

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


# =========================================================
# PART 1 - Load VGG19 Model
# =========================================================

print("Loading VGG19 model...")

model = VGG19(weights="imagenet")

print("\nVGG19 Model Summary:")
model.summary()


# =========================================================
# PART 2 - Display Layer Numbers and Names
# =========================================================

print("\n" + "=" * 70)
print("VGG19 LAYERS")
print("=" * 70)

n = 0

for layer in model.layers:
    print(n, layer.name)
    n += 1


# =========================================================
# PART 3 - Display Convolutional Layer Filter Shapes
# =========================================================

print("\n" + "=" * 70)
print("CONVOLUTIONAL LAYER FILTERS")
print("=" * 70)

n = 0

for layer in model.layers:

    if "conv" in layer.name:

        filters, biases = layer.get_weights()

        print(
            n,
            layer.name,
            "Filters:",
            filters.shape,
            "Biases:",
            biases.shape
        )

    n += 1


# =========================================================
# PART 4 - Visualize Filters from the First Conv Layer
# =========================================================

print("\n" + "=" * 70)
print("VISUALIZING FIRST CONVOLUTIONAL LAYER FILTERS")
print("=" * 70)

# First convolutional layer
layer = model.get_layer("block1_conv1")

filters, biases = layer.get_weights()

print("Original filter shape:", filters.shape)

# Normalize filter values to 0-1
f_min = filters.min()
f_max = filters.max()

filters = (filters - f_min) / (f_max - f_min)

print("Normalized filter shape:", filters.shape)


# Display the first 4 filters
n_filters = 4

plt.figure(figsize=(10, 10))

for i in range(n_filters):

    # Get one filter
    filter_image = filters[:, :, :, i]

    # Convert RGB filter to grayscale for easier visualization
    filter_image = np.mean(filter_image, axis=2)

    ax = plt.subplot(2, 2, i + 1)

    ax.imshow(
        filter_image,
        cmap="gray"
    )

    ax.set_title("Filter {}".format(i + 1))

    ax.axis("off")

plt.suptitle(
    "First 4 Filters - VGG19 block1_conv1"
)

plt.tight_layout()

plt.show()


# =========================================================
# PART 5 - Load an Image
# =========================================================

# ---------------------------------------------------------
# IMPORTANT:
# Change this to the location of your own image.
#
# Example:
# image_path = "bird.jpg"
# ---------------------------------------------------------

image_path = "elephant.webp"

print("\n" + "=" * 70)
print("LOADING IMAGE")
print("=" * 70)

# Load image and resize to VGG19 input size
image = load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to NumPy array
image = img_to_array(image)

# Add batch dimension
image = np.expand_dims(
    image,
    axis=0
)

# Preprocess image for VGG19
image = preprocess_input(image)

print("Image shape:", image.shape)


# =========================================================
# PART 6 - Visualize Feature Maps
# =========================================================

print("\n" + "=" * 70)
print("VISUALIZING FEATURE MAPS")
print("=" * 70)


# ---------------------------------------------------------
# Select convolutional layers
# ---------------------------------------------------------

layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]


# Create a model that outputs these layers
outputs = [
    model.get_layer(name).output
    for name in layer_names
]

feature_model = Model(
    inputs=model.input,
    outputs=outputs
)


# Generate feature maps
feature_maps = feature_model.predict(
    image,
    verbose=1
)


# ---------------------------------------------------------
# Display feature maps
# ---------------------------------------------------------

for layer_name, feature_map in zip(
    layer_names,
    feature_maps
):

    print(
        layer_name,
        "output shape:",
        feature_map.shape
    )

    # Number of feature maps to display
    n_features = min(
        feature_map.shape[-1],
        64
    )

    # Create an 8 x 8 grid
    square = 8

    plt.figure(
        figsize=(12, 12)
    )

    for i in range(n_features):

        ax = plt.subplot(
            square,
            square,
            i + 1
        )

        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(
            feature_map[0, :, :, i],
            cmap="gray"
        )

    plt.suptitle(
        "Feature Maps: " + layer_name
    )

    plt.tight_layout()

    plt.show()


print("\nVisualization completed!")