import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODEL_DIR)
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
MODEL_PATHS = [
    os.path.join(MODEL_DIR, "model.keras"),
    os.path.join(MODEL_DIR, "model.h5"),
]
IMG_SIZE = (224, 224)
TOP_K = 3
CLASSES = {}
CLASSES_MTIME = None
model = None
MODEL_MTIME = None


def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    if os.path.isdir(dataset_dir):
        class_names = sorted(
            name for name in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, name))
        )
        return {str(index): name for index, name in enumerate(class_names)}

    return {}


def load_model():
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                return tf.keras.models.load_model(model_path)
            except Exception as error:
                print(f"Error loading model '{model_path}': {error}")
    return None


def get_classes():
    global CLASSES, CLASSES_MTIME

    current_mtime = os.path.getmtime(CLASSES_PATH) if os.path.exists(CLASSES_PATH) else None
    if CLASSES and CLASSES_MTIME == current_mtime:
        return CLASSES

    CLASSES = load_classes()
    CLASSES_MTIME = current_mtime
    return CLASSES


def get_model():
    global model, MODEL_MTIME

    latest_path = next((path for path in MODEL_PATHS if os.path.exists(path)), None)
    current_mtime = os.path.getmtime(latest_path) if latest_path else None

    if model is not None and MODEL_MTIME == current_mtime:
        return model

    model = load_model()
    MODEL_MTIME = current_mtime
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def build_top_predictions(probabilities, classes):
    top_indices = np.argsort(probabilities)[::-1][:TOP_K]
    return [
        {
            "breed": classes.get(str(index), f"Class {index}"),
            "confidence": float(probabilities[index]),
        }
        for index in top_indices
    ]


def predict_breed(image_path):
    classes = get_classes()
    active_model = get_model()

    if not classes:
        return {
            "breed": "Model unavailable",
            "confidence": 0.0,
            "is_demo": True,
            "message": "No class metadata was found. Train the model again to regenerate model/classes.json.",
        }

    if active_model:
        try:
            processed_img = preprocess_image(image_path)
            predictions = active_model.predict(processed_img, verbose=0)[0]
            predicted_class_index = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            breed_name = classes.get(str(predicted_class_index), "Unknown")

            return {
                "breed": breed_name,
                "confidence": confidence,
                "is_demo": False,
                "top_predictions": build_top_predictions(predictions, classes),
            }
        except Exception as error:
            return {
                "breed": "Prediction failed",
                "confidence": 0.0,
                "is_demo": True,
                "message": f"Prediction failed: {error}",
            }

    return {
        "breed": "Model unavailable",
        "confidence": 0.0,
        "is_demo": True,
        "message": "No trained model was found. Train the model to get real predictions.",
    }
