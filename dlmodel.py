# Example 4.7

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

# Create model
model = Sequential()

# Input layer
model.add(Input(shape=(28, 28, 1)))

# Convolutional layer
model.add(Conv2D(
    32,
    (5, 5),
    activation='relu'
))

# Pooling layer
model.add(MaxPooling2D(
    pool_size=(2, 2)
))

# Dropout
model.add(Dropout(0.2))

# Flatten
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Display model summary
model.summary()