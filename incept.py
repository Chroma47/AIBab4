#Example 4.6
from tensorflow.keras.applications import inception_v3

# init the models
model = inception_v3.InceptionV3(weights='imagenet')
print(model.summary())