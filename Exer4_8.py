# Example 4.9 (Modified)
# Added a second Conv2D + MaxPooling2D + Dropout block before Flatten
# Compares performance of the original (1 conv block) vs modified (2 conv blocks)

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


# ---- Original model: 1 conv block -------------------------------
def build_original():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    # Conv block 1
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# ---- Modified model: 2 conv blocks ------------------------------
def build_modified():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))

    # Conv block 1
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Conv block 2  <-- NEW
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# ---- Experiment runner ------------------------------------------
def run_experiment(name, builder):
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    model = builder()
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

    print(f"Test accuracy : {scores[1]*100:.2f}%")
    print(f"Error         : {error_pct:.2f}%")
    print(f"Params        : {model.count_params():,}")
    print(f"Training time : {train_time:.1f} s")

    return {
        'name': name,
        'params': model.count_params(),
        'time': train_time,
        'test_acc': scores[1],
        'error': error_pct,
        'val_acc': history.history['val_accuracy'],
        'val_loss': history.history['val_loss'],
    }


# ---- Run both configurations ------------------------------------
r_original = run_experiment("Original (1 conv block)", build_original)
r_modified = run_experiment("Modified (2 conv blocks)", build_modified)


# ---- Summary ----------------------------------------------------
print("\n" + "=" * 78)
print(f"{'model':<28} | {'params':>10} | {'time(s)':>8} | "
      f"{'test acc':>9} | {'error':>7}")
print("-" * 78)
for r in [r_original, r_modified]:
    print(f"{r['name']:<28} | {r['params']:>10,} | "
          f"{r['time']:>8.1f} | {r['test_acc']*100:>8.2f}% | "
          f"{r['error']:>6.2f}%")


# ---- Plot validation accuracy curves ----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(r_original['val_acc'], label='Original (1 block)')
plt.plot(r_modified['val_acc'], label='Modified (2 blocks)')
plt.title('Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r_original['val_loss'], label='Original (1 block)')
plt.plot(r_modified['val_loss'], label='Modified (2 blocks)')
plt.title('Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout()
plt.show()