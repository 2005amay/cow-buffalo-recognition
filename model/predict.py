import os
import json
import random
import tensorflow as tf
import numpy as np
from PIL import Image

# Load class names
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'classes.json')
with open(CLASSES_PATH, 'r') as f:
    CLASSES = json.load(f)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

model = load_model()

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_breed(image_path):
    global model
    
    # Reload model if it appeared (e.g. after training)
    if model is None:
        model = load_model()

    if model:
        # Real prediction
        try:
            processed_img = preprocess_image(image_path)
            predictions = model.predict(processed_img)
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            breed_name = CLASSES.get(str(predicted_class_index), "Unknown")
            
            return {
                'breed': breed_name,
                'confidence': confidence,
                'is_demo': False
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback to demo if prediction fails
            pass

    # Demo mode (Simulated)
    # In a real scenario, we would strictly fail, but for this demo request
    # we simulate a result if the model isn't trained yet.
    print("Model not found or failed. Using DEMO mode.")
    
    # Simulate a random breed for demonstration
    random_index = str(random.randint(0, len(CLASSES) - 1))
    breed_name = CLASSES[random_index]
    confidence = random.uniform(0.85, 0.99)
    
    return {
        'breed': breed_name,
        'confidence': confidence,
        'is_demo': True,
        'message': 'Running in DEMO mode. Train the model to get real predictions.'
    }
