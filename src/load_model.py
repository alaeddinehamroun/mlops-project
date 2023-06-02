import tensorflow as tf
import cv2
import numpy as np


new_model = tf.keras.models.load_model('./saved_model/my_model')
new_model.summary()

test_image = './PokemonData/Weezing/2a1a51667d764271958221642e59efab.jpg'
image = cv2.imread(test_image)
image = cv2.resize(image, (180, 180))  # Resize to match the model's input size
image = image / 255.0
image = np.expand_dims(image, axis=0)  # Add a batch dimension

print(new_model.predict(test_image))