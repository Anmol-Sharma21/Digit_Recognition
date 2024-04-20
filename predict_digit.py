import cv2
import numpy as np
import tensorflow as tf

def predict_digit(image_path, model):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    image = np.expand_dims(image, axis=-1)  # Add a channel dimension (for grayscale)
    
    # Predict the digit using the model
    prediction = model.predict(image)
    
    # Get the predicted digit (index of the highest probability)
    predicted_digit = np.argmax(prediction)

    return predicted_digit

# Load the trained model
loaded_model = tf.keras.models.load_model('trained_model.keras')  

# Path to the image you want to predict
image_path = 'digit.png'  

# Predict the digit
predicted_digit = predict_digit(image_path, loaded_model)
print("Predicted Digit:", predicted_digit)
