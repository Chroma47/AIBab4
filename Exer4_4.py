# Example 4.4 LeNet-5 Keras (Modified)
# Explore how the number of filters affects the total parameter count
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Input
from keras import optimizers

def build_lenet(filters1=6, filters2=16, input_shape=(32, 32, 1),
                dense1=120, dense2=84, num_classes=10, verbose=True):
    """Build a LeNet-5 style model with configurable filter counts."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(filters=filters1, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=filters2, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=dense1, activation='relu'))
    model.add(Dense(units=dense2, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    if verbose:
        model.summary()
    return model


# ---- Compare several filter configurations ----------------------
configs = [
    (6, 16),    # original LeNet-5
    (4, 12),    # fewer filters
    (8, 20),    # slightly more
    (12, 32),   # double-ish
    (16, 64),   # much larger
]

print(f"{'filters1':>8} | {'filters2':>8} | {'total params':>13} | "
      f"{'conv1 params':>12} | {'conv2 params':>12}")
print("-" * 72)

for f1, f2 in configs:
    model = build_lenet(filters1=f1, filters2=f2, verbose=False)
    total = model.count_params()

    # Manually compute conv layer params
    # Conv1: (k*k*in_channels + 1) * filters1
    conv1 = (3 * 3 * 1 + 1) * f1
    # Conv2 input channels = f1
    conv2 = (3 * 3 * f1 + 1) * f2

    print(f"{f1:>8} | {f2:>8} | {total:>13,} | {conv1:>12,} | {conv2:>12,}")

# ---- Show the full summary for the original config ---------------
print("\n=== Detailed summary for original (6, 16) ===")
build_lenet(filters1=6, filters2=16, verbose=True)

print("\n=== Detailed summary for (12, 32) ===")
build_lenet(filters1=12, filters2=32, verbose=True)