"""
CNN model for image classification with data augmentation and rescaling.
"""

from tensorflow import keras
import data_loading


def create_model():
    """
    Create whole Sequential model stack and return it.
    """
    resize_and_rescale = keras.Sequential([
        keras.layers.Resizing(data_loading.IMG_WIDTH, data_loading.IMG_HEIGHT),
        keras.layers.Rescaling(1. / 255)
    ])

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
    ])

    model = keras.Sequential([
        resize_and_rescale,
        data_augmentation,
        keras.layers.Conv2D(64, 5, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    return model
