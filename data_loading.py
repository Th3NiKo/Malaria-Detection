"""
Module for preprocessing and loading malaria cell images.

Base is kaggle dataset "Malaria Cell Images Dataset"
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
"""

import os
import random
import shutil
from typing import Tuple
import tensorflow as tf


SEED = 2022
LABELS = ["Parasitized", "Uninfected"]
RAW_DATASET_PATH = "cell_images"
SPLIT_DATASET_PATH = "dataset"

VALIDATION_SPLIT = 0.15
BATCH_SIZE = 32
IMG_WIDTH, IMG_HEIGHT = (100, 100)

random.seed(SEED)

def train_test_split(actual_dataset_path: str, new_dataset_path: str,
                     split_value: float = 0.2):
    """
    Split images into two new folders "Train" and "Test".
    Each of them contains two folders "Parasitized" and "Uninfected".
    Function does not delete original dataset.

    Args:
        actual_dataset_path (str): path to dataset folder before train/test split.
            Should contain "Parasitized" and "Uninfected" folders.
        new_dataset_path (str): path to dataset folder after train/test split (result).
            If folder does not exist, will create it
        split_value (float): number from 0.01 to 0.99.
            Telling how much data to take for test split.

    Examples:
        >>> train_test_split("cell_images", "dataset")
    """
    train_path = os.path.join(new_dataset_path, "Train")
    test_path = os.path.join(new_dataset_path, "Test")
    paths = [train_path, test_path]

    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)

    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        for label in LABELS:
            label_dir = os.path.join(path, label)
            os.mkdir(label_dir)

    for label in LABELS:
        actual_dir = os.path.join(actual_dataset_path, label)
        images_names = os.listdir(actual_dir)
        random.shuffle(images_names)
        split_point = int(len(images_names) * split_value)

        # Train images
        for image_name in images_names[split_point:]:
            source = os.path.join(actual_dir, image_name)
            destination = os.path.join(train_path, label, image_name)
            shutil.copyfile(source, destination)

        # Test images
        for image_name in images_names[0:split_point]:
            source = os.path.join(actual_dir, image_name)
            destination = os.path.join(test_path, label, image_name)
            shutil.copyfile(source, destination)


def create_image_datasets(train_data_dir: str = os.path.join(SPLIT_DATASET_PATH, "Train"),
                         test_data_dir: str = os.path.join(SPLIT_DATASET_PATH, "Test"),
                         validation_split: float = 0.2,
                         image_size: Tuple = (IMG_WIDTH, IMG_HEIGHT),
                         batch_size = BATCH_SIZE) -> \
                         Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Creates tensorflow image datasets for train, validation and test sets.
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
                        train_data_dir,
                        validation_split=validation_split,
                        subset="training",
                        seed=SEED,
                        image_size=image_size,
                        batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
                            train_data_dir,
                            validation_split=validation_split,
                            subset="validation",
                            seed=SEED,
                            image_size=image_size,
                            batch_size=batch_size)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
                        test_data_dir,
                        seed=SEED,
                        image_size=image_size,
                        batch_size=batch_size)

    return train_dataset, validation_dataset, test_dataset
