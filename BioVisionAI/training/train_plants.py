"""
BioVision AI — Plant Disease Classification Training Script
Uses MobileNetV2 transfer learning with ImageDataGenerator augmentation.

Usage (from project root):
    python training/train_plants.py
"""

import os
import sys
import json
import logging
import numpy as np

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "plants")
OUTPUT_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "plant_model.h5")
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 30
LEARNING_RATE = 1e-4


def build_model(num_classes: int):
    """
    Build a MobileNetV2-based transfer learning model.

    Args:
        num_classes : Number of output classes.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models, optimizers

    # Load MobileNetV2 pre-trained on ImageNet; exclude top classifier
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze base initially

    # Custom classification head
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


def fine_tune_model(model, base_model, num_unfreeze: int = 30):
    """
    Unfreeze the last N layers of the base model for fine-tuning.
    """
    from tensorflow.keras import optimizers
    base_model.trainable = True
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    )
    from sklearn.metrics import classification_report, confusion_matrix
    from utils.charts import confusion_matrix_png, training_history_png

    logger.info("=== BioVision AI — Plant Model Training ===")

    # ── Verify dataset ──────────────────────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        logger.error(f"Dataset not found at: {DATASET_DIR}")
        logger.error("Please run setup_datasets.py first or place images manually.")
        return

    classes = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    if not classes:
        logger.error("No class subdirectories found in dataset.")
        return

    logger.info(f"Found {len(classes)} classes: {classes}")

    # ── Data generators ─────────────────────────────────────────────────────
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        validation_split=0.2,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = train_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = len(train_data.class_indices)
    logger.info(f"Training on {train_data.samples} images | Validating on {val_data.samples} images")

    # ── Build model ──────────────────────────────────────────────────────────
    model, base_model = build_model(num_classes)
    model.summary(print_fn=logger.info)

    # ── Callbacks ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ]

    # ── Phase 1: Train head only ─────────────────────────────────────────────
    logger.info("Phase 1: Training classification head...")
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS // 2,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Phase 2: Fine-tune ───────────────────────────────────────────────────
    logger.info("Phase 2: Fine-tuning top layers...")
    model = fine_tune_model(model, base_model)
    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        initial_epoch=history1.epoch[-1] + 1,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Merge histories ──────────────────────────────────────────────────────
    merged = {}
    for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
        merged[key] = (
            history1.history.get(key, []) + history2.history.get(key, [])
        )

    # ── Save class map ───────────────────────────────────────────────────────
    class_map_path = OUTPUT_MODEL.replace(".h5", "_classes.json")
    with open(class_map_path, "w") as f:
        json.dump(train_data.class_indices, f, indent=2)
    logger.info(f"Class map saved: {class_map_path}")

    # ── Evaluation ───────────────────────────────────────────────────────────
    logger.info("Evaluating on validation set...")
    val_data.reset()
    preds   = model.predict(val_data, verbose=0)
    y_pred  = np.argmax(preds, axis=1)
    y_true  = val_data.classes[:len(y_pred)]
    idx2cls = {v: k for k, v in train_data.class_indices.items()}
    target_names = [idx2cls[i] for i in range(num_classes)]

    report = classification_report(y_true, y_pred, target_names=target_names)
    logger.info(f"\nClassification Report:\n{report}")

    cm = confusion_matrix(y_true, y_pred)

    # ── Save chart PNGs ───────────────────────────────────────────────────────
    hist_png = training_history_png(merged)
    cm_png   = confusion_matrix_png(cm, target_names)
    charts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(charts_dir, exist_ok=True)
    with open(os.path.join(charts_dir, "plant_training_history.png"), "wb") as f:
        f.write(hist_png)
    with open(os.path.join(charts_dir, "plant_confusion_matrix.png"), "wb") as f:
        f.write(cm_png)

    logger.info(f"Model saved: {OUTPUT_MODEL}")
    logger.info("=== Training complete! ===")


if __name__ == "__main__":
    train()
