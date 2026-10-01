# Example 4.10 (Modified)
# Load VGG19 instead of VGG16

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

# Load the VGG19 model pre-trained on ImageNet
model = VGG19(weights='imagenet')
print(model.summary())