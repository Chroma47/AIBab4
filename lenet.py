#Example 4.4 LeNet-5 Keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Input
from keras import optimizers

model = Sequential()

model.add(Input(shape=(32, 32, 1)))

model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()