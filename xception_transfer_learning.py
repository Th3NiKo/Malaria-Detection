"""
Xception model transfer learning.
"""
import tensorflow as tf
# Importing this way because of keras lazy loading (want to use autocomplete)
from tensorflow.python.keras.api import keras
import data_loading

EPOCHS = 3
LEARNING_RATE = 0.0001
MODEL_NAME = "XCEPTION"


def training_pipeline(train_dataset: tf.data.Dataset,
                      validation_dataset: tf.data.Dataset,
                      test_dataset: tf.data.Dataset):
    """Whole training pipeline for xception finetunning"""

    # Create xception model with 1 neuron on output
    base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(1, activation="sigmoid")(avg)
    xception_model = keras.Model(inputs=base_model.input, outputs=output)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode="max", patience=5)

    optimizer = keras.optimizers.Adam(LEARNING_RATE)
    loss = keras.losses.BinaryCrossentropy()

    xception_model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=["accuracy","Recall", "Precision"])

    xception_model.fit(train_dataset,
                       validation_data=validation_dataset,
                       epochs=EPOCHS,
                       callbacks=[early_stopping])

    xception_model.evaluate(test_dataset)
    xception_model.save(MODEL_NAME)

if __name__ == "__main__":
    train_data, validation_data, test_data = data_loading.create_image_datasets()
    training_pipeline(train_data, validation_data, test_data)
