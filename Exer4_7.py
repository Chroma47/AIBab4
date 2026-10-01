# Example 4.9 (Modified)
# Compare performance for different filter counts (32 vs 64)
# and different kernel sizes ((5,5) vs (3,3))

import time
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.utils import to_categorical

# ---- Load data --------------------------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot 4 images as grayscale
plt.subplot(141); plt.imshow(X_train[0], cmap='gray')
plt.subplot(142); plt.imshow(X_train[1], cmap='gray')
plt.subplot(143); plt.imshow(X_train[2], cmap='gray')
plt.subplot(144); plt.imshow(X_train[3], cmap='gray')
plt.show()

# Reshape to [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test  = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train /= 255.0
X_test  /= 255.0

# One-hot encode outputs
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

num_classes = y_test.shape[1]


# ---- Model builder ---------------------------------------------
def build_model(filters=32, kernel_size=(5, 5)):
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# ---- Experiments ------------------------------------------------
configs = [
    (32, (5, 5)),   # original
    (64, (5, 5)),   # more filters
    (32, (3, 3)),   # smaller kernel
    (64, (3, 3)),   # more filters + smaller kernel
]

results = []

for filters, kernel in configs:
    print("\n" + "=" * 60)
    print(f"Training: filters={filters}, kernel_size={kernel}")
    print("=" * 60)

    model = build_model(filters=filters, kernel_size=kernel)
    model.summary()

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=200,
        verbose=0
    )
    train_time = time.time() - start

    scores = model.evaluate(X_test, y_test, verbose=0)
    error_pct = (1 - scores[1]) * 100

    results.append({
        'filters': filters,
        'kernel': kernel,
        'params': model.count_params(),
        'time': train_time,
        'test_acc': scores[1],
        'error': error_pct,
        'val_acc': history.history['val_accuracy'],
        'val_loss': history.history['val_loss'],
    })

    print(f"Test accuracy : {scores[1]*100:.2f}%")
    print(f"Error         : {error_pct:.2f}%")
    print(f"Params        : {model.count_params():,}")
    print(f"Training time : {train_time:.1f} s")


# ---- Summary table ---------------------------------------------
print("\n" + "=" * 78)
print(f"{'filters':>8} | {'kernel':>8} | {'params':>10} | "
      f"{'time(s)':>8} | {'test acc':>9} | {'error':>7}")
print("-" * 78)
for r in results:
    print(f"{r['filters']:>8} | {str(r['kernel']):>8} | "
          f"{r['params']:>10,} | {r['time']:>8.1f} | "
          f"{r['test_acc']*100:>8.2f}% | {r['error']:>6.2f}%")