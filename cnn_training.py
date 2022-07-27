"""
Basic training pipeline.
"""
from tensorflow import keras
import data_loading
import model

EPOCHS = 10
INITIAL_LEARNING_RATE = 0.0001
MODEL_NAME = "CNN_BASELINE"

train_data, validation_data, test_data = data_loading.create_image_datasets()

cnn_model = model.create_model()

early_stopping = keras.callbacks.EarlyStopping(patience=3)

optimizer = keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)

cnn_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=['accuracy', 'Precision', 'Recall']
)

history = cnn_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

cnn_model.evaluate(test_data)

cnn_model.save(MODEL_NAME)
