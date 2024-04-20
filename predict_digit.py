import cv2 #type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore

def predict_digit(image_path, model):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  
    image = np.expand_dims(image, axis=-1)  
    
    # Predict the digit using the model
    prediction = model.predict(image)
    
    # Get the predicted digit (index of the highet probability)
    predicted_digit = np.argmax(prediction)

    return predicted_digit

# Load the trained model
loaded_model = tf.keras.models.load_model('trained_model.keras')  

# Path to the image you want to predict
image_path = 'digit_2.png'  

# Predict the digit
predicted_digit = predict_digit(image_path, loaded_model)
print("Predicted Digit:", predicted_digit)
