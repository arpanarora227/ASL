from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import os
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = "/Users/gurpreetsingh/Desktop/Hackathon/asl_digit_classifier.keras"  # or "asl_digit_classifier.h5" if saved in H5 format
model = load_model(model_path)

# Function to preprocess and predict
def predict_image(img):
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return predicted_class

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the base64 image
    image_data = data["image"].split(",")[1]  # Split to remove the "data:image/jpeg;base64," part
    img = Image.open(BytesIO(base64.b64decode(image_data)))

    # Predict the class
    predicted_class = predict_image(img)

    # Return the prediction as JSON
    return jsonify({"prediction": int(predicted_class)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
