# Example 4.5 AlexNet with Keras (Modified)
# Explore how the number of dense layer units affects the total parameter count

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input


def build_alexnet(dense1=4096, dense2=4096, dense3=1000,
                  num_classes=17, verbose=False):
    """Build an AlexNet-style model with configurable dense layer units."""
    model = Sequential()

    # Input Layer
    model.add(Input(shape=(224, 224, 3)))

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(11, 11), activation='relu',
                     strides=(4, 4), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), activation='relu',
                     strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu',
                     strides=(1, 1), padding='valid'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu',
                     strides=(1, 1), padding='valid'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 1st Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(dense1, activation='relu'))
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(dense2, activation='relu'))
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(dense3, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    if verbose:
        model.summary()
    return model


# ---- Compare several dense-layer configurations ------------------
configs = [
    (4096, 4096, 1000),   # original AlexNet
    (2048, 2048, 1000),   # halved
    (1024, 1024, 1000),   # quartered
    (512,  512,  500),    # much smaller
    (256,  256,  256),    # tiny
    (8192, 8192, 1000),   # doubled
]

# Compute the flatten size for AlexNet's architecture on 224x224 input
def conv_output(size, kernel, stride, padding=0):
    return (size - kernel + 2 * padding) // stride + 1

def pool_output(size, pool, stride):
    return (size - pool) // stride + 1

s = 224
s = conv_output(s, 11, 4)      # Conv1  -> 54
s = pool_output(s, 2, 2)       # Pool1  -> 27
s = conv_output(s, 11, 1)      # Conv2  -> 17
s = pool_output(s, 2, 2)       # Pool2  -> 8
s = conv_output(s, 3, 1)       # Conv3  -> 6
s = conv_output(s, 3, 1)       # Conv4  -> 4
s = conv_output(s, 3, 1)       # Conv5  -> 2
s = pool_output(s, 2, 2)       # Pool3  -> 1
flatten_size = s * s * 256     # = 256

print(f"Flatten size after convolutional layers: {flatten_size}\n")

# Fixed conv parameters (independent of dense configuration)
conv_params = (
    (11 * 11 * 3 + 1) * 96 +          # Conv1
    (11 * 11 * 96 + 1) * 256 +        # Conv2
    (3 * 3 * 256 + 1) * 384 +         # Conv3
    (3 * 3 * 384 + 1) * 384 +         # Conv4
    (3 * 3 * 384 + 1) * 256           # Conv5
)
print(f"Fixed convolution parameters: {conv_params:,}\n")

print(f"{'dense1':>8} | {'dense2':>8} | {'dense3':>8} | "
      f"{'dense params':>14} | {'total params':>14}")
print("-" * 72)

for d1, d2, d3 in configs:
    dense_params = (
        flatten_size * d1 + d1 +
        d1 * d2 + d2 +
        d2 * d3 + d3 +
        d3 * 17 + 17
    )
    total = conv_params + dense_params
    print(f"{d1:>8} | {d2:>8} | {d3:>8} | "
          f"{dense_params:>14,} | {total:>14,}")

# ---- Show the full summary for the original config ---------------
print("\n=== Detailed summary for original (4096, 4096, 1000) ===")
build_alexnet(4096, 4096, 1000, verbose=True)

print("\n=== Detailed summary for (512, 512, 500) ===")
build_alexnet(512, 512, 500, verbose=True)

# Compile the model (example for the original configuration)
model = build_alexnet(4096, 4096, 1000)
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

# Fit the model
# model.fit()

# Prediction with the model
# model.evaluate()