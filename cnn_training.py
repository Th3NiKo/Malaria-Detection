"""
Basic CNN training pipeline.
"""
import tensorflow as tf
from tensorflow import keras
import data_loading
import model

EPOCHS = 10
INITIAL_LEARNING_RATE = 0.0001
MODEL_NAME = "CNN_BASELINE"


def training_pipeline(train_dataset: tf.data.Dataset,
                      validation_dataset: tf.data.Dataset,
                      test_dataset: tf.data.Dataset):
    """Whole training pipeline for basic CNN"""
    cnn_model = model.create_model()

    early_stopping = keras.callbacks.EarlyStopping(patience=3)
    optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)

    cnn_model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy', 'Precision', 'Recall']
    )

    cnn_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    cnn_model.evaluate(test_dataset)
    cnn_model.save(MODEL_NAME)

if __name__ == "__main__":
    train_data, validation_data, test_data = data_loading.create_image_datasets()
    training_pipeline(train_data, validation_data, test_data)
