# Example 4.16
# The Autoencoder.py Program
# Autoencoder for MNIST handwritten digits

# ---------------------------------------------------------
# Load the MNIST handwritten digit images
# ---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(x_train.shape)
print(x_test.shape)


# ---------------------------------------------------------
# Build Autoencoder
# ---------------------------------------------------------

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input


def build_autoencoder(img_shape, code_size):

    # The encoder
    encoder = Sequential(name="encoder")

    encoder.add(Input(shape=img_shape))
    encoder.add(Flatten())

    # Encode the image into a smaller representation
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential(name="decoder")

    decoder.add(Input(shape=(code_size,)))

    # Decode back into the original image size
    decoder.add(
        Dense(
            int(np.prod(img_shape)),
            activation="sigmoid"
        )
    )

    decoder.add(Reshape(img_shape))

    return encoder, decoder


# ---------------------------------------------------------
# Build the encoder and decoder
# ---------------------------------------------------------

IMG_SHAPE = x_train[0].shape

# Size of the encoded representation
CODE_SIZE = 32

encoder, decoder = build_autoencoder(
    IMG_SHAPE,
    CODE_SIZE
)


# ---------------------------------------------------------
# Connect encoder and decoder
# ---------------------------------------------------------

inp = Input(shape=IMG_SHAPE)

code = encoder(inp)

reconstruction = decoder(code)

autoencoder = Model(
    inp,
    reconstruction
)


# ---------------------------------------------------------
# Compile Autoencoder
# ---------------------------------------------------------

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)


print(autoencoder.summary())


# ---------------------------------------------------------
# Train Autoencoder
# ---------------------------------------------------------

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=20,
    validation_data=(x_test, x_test)
)


# ---------------------------------------------------------
# Plot the training results
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Test Loss"
)

plt.title("Autoencoder Training")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()


# ---------------------------------------------------------
# Display the original image
# ---------------------------------------------------------

def show_image(x):

    plt.imshow(
        np.clip(x, 0, 1),
        cmap="gray"
    )

    plt.axis("off")


# ---------------------------------------------------------
# Visualize original, encoded and reconstructed images
# ---------------------------------------------------------

def visualize(img, encoder, decoder):

    # Encode the image
    code = encoder.predict(
        img[None],
        verbose=0
    )[0]

    # Decode the encoded image
    reco = decoder.predict(
        code[None],
        verbose=0
    )[0]

    # Create the figure
    plt.figure(figsize=(10, 4))

    # -----------------------------------------------------
    # Original
    # -----------------------------------------------------

    plt.subplot(1, 3, 1)

    show_image(img)

    plt.title("Original")

    # -----------------------------------------------------
    # Code
    # -----------------------------------------------------

    plt.subplot(1, 3, 2)

    # CODE_SIZE = 32
    # Display it as 16 x 2
    plt.imshow(
        code.reshape(16, 2),
        cmap="viridis",
        aspect="auto"
    )

    plt.title("Code")

    plt.colorbar()

    # -----------------------------------------------------
    # Reconstructed
    # -----------------------------------------------------

    plt.subplot(1, 3, 3)

    show_image(reco)

    plt.title("Reconstructed")

    plt.tight_layout()

    plt.show()


# ---------------------------------------------------------
# Find examples of specific digits
# ---------------------------------------------------------

# Find one digit 6
digit_6_index = np.where(y_test == 6)[0][0]

# Find one digit 2
digit_2_index = np.where(y_test == 2)[0][0]


# ---------------------------------------------------------
# Display digit 6
# ---------------------------------------------------------

visualize(
    x_test[digit_6_index],
    encoder,
    decoder
)


# ---------------------------------------------------------
# Display digit 2
# ---------------------------------------------------------

visualize(
    x_test[digit_2_index],
    encoder,
    decoder
)