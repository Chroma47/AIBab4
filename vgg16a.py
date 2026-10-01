# Example 4.11

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Load the VGG16 model
model = VGG16(weights='imagenet')

# Image path
img_path = 'elephant.webp'

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array
x = image.img_to_array(img)

# Add batch dimension
x = np.expand_dims(x, axis=0)

# Preprocess image
x = preprocess_input(x)

# Prediction
predictions = model.predict(x)

# Decode predictions
results = decode_predictions(predictions, top=5)

print(results)