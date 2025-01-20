from sklearn.preprocessing import LabelEncoder
import polars as pl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib

# Load model
model = load_model('image_classification_model.h5')
class_names = ["architecure", "art_culture", "food_drinks", "travel_adventure"]


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    
    return predicted_class, predictions[0]

def main(img_path):
    predicted_class, probabilities = predict_image(img_path)

    print(f'Predicted class: {predicted_class}')
    print(f'Class probability: {probabilities}')
    return {"predicted class": predicted_class, "class probability": probabilities}


if __name__ == "__main__":
    img_path = 'example.jpg' 
    result = main(img_path)
    print(result)