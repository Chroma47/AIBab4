# Example 4.19 SimpleRNN.py
# Updated for current TensorFlow / Keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Activation
from tensorflow.keras import optimizers


# Create the model
model = Sequential()

# Add SimpleRNN layer
model.add(
    SimpleRNN(
        50,
        input_shape=(49, 1),
        return_sequences=False
    )
)

# Add output layer
model.add(
    Dense(46)
)

# Add Softmax activation
model.add(
    Activation("softmax")
)


# Create Adam optimizer
# Modern Keras uses learning_rate instead of lr
adam = optimizers.Adam(
    learning_rate=0.001
)


# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=adam,
    metrics=["accuracy"]
)


# Display model structure
print(model.summary())