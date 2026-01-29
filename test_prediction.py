
import os
import sys

# Add the project root to the path so we can import from model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.predict import predict_breed
import tensorflow as tf

def test_prediction():
    print("Testing prediction...")
    
    # Path to a real image from the dataset
    test_image_path = os.path.join("dataset", "Hariana", "Image_1.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        return

    print(f"Using test image: {test_image_path}")

    # Check if model exists (training might still be running or finished)
    model_path = os.path.join("model", "model.h5")
    if os.path.exists(model_path):
        print(f"Model found at {model_path}")
    else:
        print("Model file not found yet (training might be in progress). Using DEMO mode check.")

    try:
        result = predict_breed(test_image_path)
        print("Prediction Result:")
        print(result)
        
        if result:
            print("Prediction successful (can be demo or real).")
        else:
            print("Prediction returned None.")
            
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction()
