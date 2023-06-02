from models.cnn_model import cnn_model
import tensorflow as tf
import pytest



def test_cnn_model():
    # Call the cnn_model() function
    model = cnn_model()

    # Perform assertions to check if the returned object is an instance of Sequential model
    assert isinstance(model, tf.keras.models.Sequential)

    # Check the number of layers in the model:
    assert len(model.layers) == 9