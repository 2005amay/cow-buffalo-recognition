import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
FINE_TUNE_LAYERS = 40

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_SAVE_PATH = MODEL_DIR / "model.keras"
CLASSES_SAVE_PATH = MODEL_DIR / "classes.json"
TRAINING_SUMMARY_PATH = MODEL_DIR / "training_summary.json"


def build_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )
    validation_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=VALIDATION_SPLIT,
    )

    generator_args = {
        "directory": str(DATASET_DIR),
        "target_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "class_mode": "categorical",
        "seed": RANDOM_SEED,
    }

    train_generator = train_datagen.flow_from_directory(
        subset="training",
        shuffle=True,
        **generator_args,
    )
    validation_generator = validation_datagen.flow_from_directory(
        subset="validation",
        shuffle=False,
        **generator_args,
    )

    return train_generator, validation_generator


def save_class_indices(class_indices):
    class_map = {str(index): name for name, index in class_indices.items()}
    with open(CLASSES_SAVE_PATH, "w", encoding="utf-8") as file:
        json.dump(class_map, file, indent=4)


def compute_class_weights(train_generator):
    class_counts = np.bincount(train_generator.classes, minlength=train_generator.num_classes)
    total_samples = float(np.sum(class_counts))
    class_weights = {
        index: float(total_samples / (train_generator.num_classes * count))
        for index, count in enumerate(class_counts)
        if count > 0
    }
    return class_counts.tolist(), class_weights


def build_model(num_classes):
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy", TopKCategoricalAccuracy(k=3, name="top_3_accuracy")],
    )


def build_callbacks():
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_SAVE_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )
    return [checkpoint, early_stop, reduce_lr]


def unfreeze_for_fine_tuning(base_model):
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - FINE_TUNE_LAYERS)

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def merge_histories(*histories):
    merged = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def save_training_summary(train_generator, validation_generator, class_counts, class_weights, history):
    class_map = {str(index): name for name, index in train_generator.class_indices.items()}
    summary = {
        "dataset_dir": str(DATASET_DIR),
        "model_path": str(MODEL_SAVE_PATH),
        "train_samples": train_generator.samples,
        "validation_samples": validation_generator.samples,
        "class_map": class_map,
        "class_counts": class_counts,
        "class_weights": class_weights,
        "best_val_accuracy": max(history.get("val_accuracy", [0.0])),
        "best_val_top_3_accuracy": max(history.get("val_top_3_accuracy", [0.0])),
        "best_val_loss": min(history.get("val_loss", [float("inf")])),
        "epochs_completed": len(history.get("loss", [])),
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def train_model():
    if not DATASET_DIR.exists():
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        return

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_generator, validation_generator = build_generators()
    save_class_indices(train_generator.class_indices)

    class_counts, class_weights = compute_class_weights(train_generator)
    print(f"Class weights: {class_weights}")

    model, base_model = build_model(train_generator.num_classes)
    callbacks = build_callbacks()

    print("\n--- Phase 1: Training classifier head ---")
    compile_model(model, learning_rate=1e-3)
    history_head = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    print("\n--- Phase 2: Fine-tuning EfficientNet ---")
    unfreeze_for_fine_tuning(base_model)
    compile_model(model, learning_rate=1e-5)
    history_fine = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=TOTAL_EPOCHS,
        initial_epoch=len(history_head.history.get("loss", [])),
        callbacks=callbacks,
        class_weight=class_weights,
    )

    full_history = merge_histories(history_head, history_fine)
    save_training_summary(
        train_generator,
        validation_generator,
        class_counts,
        class_weights,
        full_history,
    )

    print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")
    print(f"Training summary saved to {TRAINING_SUMMARY_PATH}")


if __name__ == "__main__":
    train_model()
