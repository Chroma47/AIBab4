# Example 4.9

# Simple CNN for the MNIST Dataset

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Plot 4 images as grayscale
plt.subplot(141)
plt.imshow(X_train[0], cmap='gray')

plt.subplot(142)
plt.imshow(X_train[1], cmap='gray')

plt.subplot(143)
plt.imshow(X_train[2], cmap='gray')

plt.subplot(144)
plt.imshow(X_train[3], cmap='gray')

plt.show()

# Reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]

# Create model
model = Sequential()

# Input layer
model.add(Input(shape=(28, 28, 1)))

# Convolutional layer
model.add(Conv2D(32, (5, 5), activation='relu'))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Fit the model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200
)

# Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("CNN Error: %.2f%%" % (100 - scores[1] * 100))