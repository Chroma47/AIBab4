# Example 4.20
# SimpleRNN3a.py
# Predicting airline passenger numbers with SimpleRNN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


# =========================================================
# Convert data into dataset matrix
# =========================================================

def convertToMatrix(data, step):
    X = []
    Y = []

    for i in range(len(data) - step):
        d = i + step

        X.append(data[i:d, :])
        Y.append(data[d, :])

    return np.array(X), np.array(Y)


# =========================================================
# Load airline passenger dataset
# =========================================================

# Original source mentioned in the book:
# https://www.kaggle.com/rakannimer/air-passengers

# Local version, if you have AirPassengers.csv:
# df = pd.read_csv(
#     "AirPassengers.csv",
#     usecols=[1],
#     engine="python"
# )

# Online dataset used by the book
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
    usecols=[1],
    engine="python"
)


# Display first few rows
print(df.head())


# Plot the original data
plt.figure(figsize=(10, 5))

plt.plot(df)

plt.title("Airline Passenger Data")
plt.xlabel("Month")
plt.ylabel("Passengers")

plt.show()


# =========================================================
# Set training and testing parameters
# =========================================================

step = 4

N = df.shape[0]

# Use 80% of the data for training
Tp = int(df.shape[0] * 0.8)


# =========================================================
# Split the data
# =========================================================

values = df.values

train = values[0:Tp, :]
test = values[Tp:N, :]


# =========================================================
# Add step elements into train and test
# =========================================================

test = np.append(
    test,
    np.repeat(test[-1, :], step)
)

train = np.append(
    train,
    np.repeat(train[-1, :], step)
)


# =========================================================
# Convert data into matrices
# =========================================================

trainX, trainY = convertToMatrix(
    train,
    step
)

testX, testY = convertToMatrix(
    test,
    step
)


# =========================================================
# Reshape input data
# =========================================================

trainX = np.reshape(
    trainX,
    (trainX.shape[0], 1, trainX.shape[1])
)

testX = np.reshape(
    testX,
    (testX.shape[0], 1, testX.shape[1])
)


print("\nTraining input shape:", trainX.shape)
print("Training output shape:", trainY.shape)

print("Testing input shape:", testX.shape)
print("Testing output shape:", testY.shape)


# =========================================================
# Build SimpleRNN model
# =========================================================

model = Sequential()

model.add(
    SimpleRNN(
        units=32,
        input_shape=(1, step),
        activation="relu"
    )
)

model.add(
    Dense(
        8,
        activation="relu"
    )
)

model.add(
    Dense(1)
)


# =========================================================
# Compile model
# =========================================================

model.compile(
    loss="mean_squared_error",
    optimizer="rmsprop"
)


# Display model structure
model.summary()


# =========================================================
# Train the model
# =========================================================

model.fit(
    trainX,
    trainY,
    epochs=100,
    batch_size=16,
    verbose=2
)


# =========================================================
# Make predictions
# =========================================================

trainPredict = model.predict(
    trainX,
    verbose=0
)

testPredict = model.predict(
    testX,
    verbose=0
)


# Combine predictions
predicted = np.concatenate(
    (trainPredict, testPredict),
    axis=0
)


# =========================================================
# Evaluate the model
# =========================================================

trainScore = model.evaluate(
    trainX,
    trainY,
    verbose=0
)

print("\nTraining loss:", trainScore)


# =========================================================
# Plot actual data and predictions
# =========================================================

index = df.index.values

plt.figure(figsize=(12, 6))

plt.plot(
    index,
    df.values,
    label="Data"
)

plt.plot(
    index,
    predicted[:len(index)],
    label="Prediction"
)

# Mark the training/testing boundary
plt.axvline(
    df.index[Tp],
    color="red",
    linestyle="--",
    label="Train/Test Split"
)

plt.xlabel("Time")
plt.ylabel("Passengers")

plt.title(
    "Airline Passenger Prediction Using SimpleRNN"
)

plt.legend()

plt.show()