"""
BioVision AI — Animal Disease Classification Training Script
Uses MobileNetV2 transfer learning.

Usage (from project root):
    python training/train_animals.py
"""

import os
import sys
import json
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "animals")
OUTPUT_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "animal_model.h5")
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 30
LEARNING_RATE = 1e-4


def build_model(num_classes):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models, optimizers

    base = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def train():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from sklearn.metrics import classification_report, confusion_matrix
    from utils.charts import confusion_matrix_png, training_history_png

    logger.info("=== BioVision AI — Animal Disease Model Training ===")

    if not os.path.isdir(DATASET_DIR):
        logger.error(f"Dataset not found: {DATASET_DIR}. Run setup_datasets.py first.")
        return

    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    if not classes:
        logger.error("No class directories found.")
        return
    logger.info(f"Classes: {classes}")

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.75, 1.25],
        validation_split=0.2,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = train_gen.flow_from_directory(
        DATASET_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        DATASET_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", shuffle=False,
    )

    num_classes = len(train_data.class_indices)
    logger.info(f"Training: {train_data.samples} | Validation: {val_data.samples}")

    model, base_model = build_model(num_classes)

    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ]

    history = model.fit(
        train_data, validation_data=val_data,
        epochs=EPOCHS, callbacks=callbacks, verbose=1,
    )

    # Fine-tune top 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    from tensorflow.keras import optimizers
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_data, validation_data=val_data,
        epochs=EPOCHS, initial_epoch=history.epoch[-1] + 1,
        callbacks=callbacks, verbose=1,
    )

    merged = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        merged[key] = history.history.get(key, []) + history2.history.get(key, [])

    class_map_path = OUTPUT_MODEL.replace(".h5", "_classes.json")
    with open(class_map_path, "w") as f:
        json.dump(train_data.class_indices, f, indent=2)

    val_data.reset()
    preds = model.predict(val_data, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_data.classes[:len(y_pred)]
    idx2cls = {v: k for k, v in train_data.class_indices.items()}
    target_names = [idx2cls[i] for i in range(num_classes)]
    logger.info("\n" + classification_report(y_true, y_pred, target_names=target_names))

    charts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(charts_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    with open(os.path.join(charts_dir, "animal_training_history.png"), "wb") as f:
        f.write(training_history_png(merged))
    with open(os.path.join(charts_dir, "animal_confusion_matrix.png"), "wb") as f:
        f.write(confusion_matrix_png(cm, target_names))

    logger.info(f"Model saved: {OUTPUT_MODEL}")
    logger.info("=== Animal training complete! ===")


if __name__ == "__main__":
    train()
