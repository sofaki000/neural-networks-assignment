import keras
import numpy as np


def get_transformed_mnist_dataset():
    from data_utilities.sre_dataset_utilities import get_transformed_data

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize the pixel values to the range of [0, 1].
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Add the channel dimension to the images.
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    #input shape: shape=(28, 28, 1)
    return x_train, y_train, x_test, y_test